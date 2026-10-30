"""Reproducible, checkpointed three-model population-risk study.

risk-reconstruct reproduce --out results/paper
Each row is one independently sampled risk parameter; all other inputs are
fixed. No terminal-price Monte Carlo sampling is used in Tables 1 or 2.
"""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .gbm_risk import (
    GBM,
    CallGeometry,
    PopulationCallReconstructor,
    FAMILY_ORDER,
    FAMILY_NAMES,
    DEFAULT_INTERVALS,
    common_lipschitz_budget,
    population_call_risk,
)
from .heston_risk import Heston
from .finite_risk import FiniteModel, FiniteGeometry
from .risk_reconstruction import _integer
from .provenance import source_hash, runtime_versions, write_json, write_csv, check_config, run_lock
from .reports import write_tables


def parameter_draws(repetitions: int = 30, seed: int = 20261029) -> dict[str, np.ndarray]:
    repetitions = _integer(repetitions, "repetitions", 2)
    seed = _integer(seed, "seed", 0)
    streams = np.random.SeedSequence(seed).spawn(len(FAMILY_ORDER))
    return {
        f: np.random.default_rng(s).uniform(*DEFAULT_INTERVALS[f], repetitions)
        for f, s in zip(FAMILY_ORDER, streams)
    }


def strike_design(model, n_train: int = 100, seed: int = 13579) -> dict[str, np.ndarray]:
    n_train = _integer(n_train, "n_train", 20)
    if n_train % 20:
        raise ValueError("n_train must be a multiple of 20 for the 10% split.")
    n_test = n_train // 10
    if isinstance(model, FiniteModel):
        # Keep the original strike experiment fixed when changing stock support.
        # Training: [0.1,0.9]; OOD: [0,0.1) and (0.9,1].
        lo, hi, left, right = 0.1, 0.9, 0.0, 1.0
    else:
        lo, hi = model.quantile(np.array([0.1, 0.9]))
        width = 0.25 * (hi - lo)
        left, right = max(0.0, lo - width), hi + width
    rng = np.random.default_rng(_integer(seed, "seed", 0))
    train = np.linspace(lo, hi, n_train)
    interp = np.sort(rng.uniform(lo, hi, n_test))
    extrap = np.r_[
        np.sort(rng.uniform(left, lo, n_test // 2)), np.sort(rng.uniform(hi, right, n_test // 2))
    ]
    if np.intersect1d(train, interp).size or np.intersect1d(train, extrap).size:
        raise RuntimeError("Random strike collided with a training strike; choose another seed.")
    return {"train": train, "interpolation": interp, "extrapolation": extrap}


def mse_ratio(numerator, denominator):
    if denominator == 0:
        return None, "undefined_0_over_0" if numerator == 0 else "infinite"
    return numerator / denominator, "finite"


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("GBM", "Heston", "Finite"):
        for family in FAMILY_ORDER:
            group = metrics[(metrics.model == model) & (metrics.family == family)]
            if group.empty:
                continue
            row = {
                "model": model,
                "family": family,
                "risk_measure": FAMILY_NAMES[family],
                "repetitions": len(group),
                "undefined_ratio_count": int((group.ratio_status != "finite").sum()),
            }
            for col in ("train_mse", "test_mse", "extrap_mse", "benchmark_mse", "test_ratio"):
                x = group[col].to_numpy(dtype=float)
                # Do not silently drop undefined ratios from the summary.
                row[col + "_mean"] = float(np.mean(x))
                row[col + "_sd"] = float(np.std(x, ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def model_config(model):
    if isinstance(model, FiniteModel):
        return {"states": model.states.tolist(), "probabilities": model.probabilities.tolist()}
    return asdict(model)


def run_model(model, name, out, draws, n_train=100, strike_seed=13579):
    out = Path(out)
    checkpoint = out / "checkpoints" / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    design = strike_design(model, n_train, strike_seed)
    L = common_lipschitz_budget()
    config = {
        "model": name,
        "parameters": model_config(model),
        "strikes": {k: v.tolist() for k, v in design.items()},
        "draws": {k: v.tolist() for k, v in draws.items()},
        "L": L,
        "source_hash": source_hash(),
        "runtime": runtime_versions(),
    }
    check_config(checkpoint / "config.json", config)
    start = time.perf_counter()
    geometry = (
        FiniteGeometry(model, design["train"])
        if isinstance(model, FiniteModel)
        else CallGeometry(model, design["train"])
    )
    query = np.r_[design["interpolation"], design["extrapolation"]]
    query_coordinates, query_means = geometry.coordinates(query)
    all_strikes = np.r_[design["train"], query]
    n = len(design["train"])
    m = len(design["interpolation"])
    benchmark = population_call_risk(model, query, "es", 0.99).value
    baseline_seconds = time.perf_counter() - start
    all_metrics = []
    all_predictions = []
    for family in FAMILY_ORDER:
        for repetition, parameter in enumerate(draws[family]):
            stem = f"{family}_{repetition:02d}"
            record_path = checkpoint / (stem + ".json")
            pred_path = checkpoint / (stem + ".csv")
            if record_path.exists() and pred_path.exists():
                record = json.loads(record_path.read_text())
                predictions = pd.read_csv(pred_path, float_precision="round_trip")
            else:
                tic = time.perf_counter()
                risk = population_call_risk(model, all_strikes, family, float(parameter))
                fit = PopulationCallReconstructor(geometry, risk.value[:n], L)
                train = fit.certify_training()
                test = fit.predict_coordinates(query_coordinates, query_means)
                error = (test.value - risk.value[n:]) ** 2
                train_mse = float(np.mean((train.value - risk.value[:n]) ** 2))
                test_mse = float(np.mean(error[:m]))
                extrap_mse = float(np.mean(error[m:]))
                baseline_mse = float(np.mean((benchmark[m:] - risk.value[n + m :]) ** 2))
                ratio, status = mse_ratio(extrap_mse, baseline_mse)
                record = {
                    "model": name,
                    "family": family,
                    "repetition": repetition,
                    "parameter": float(parameter),
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "extrap_mse": extrap_mse,
                    "benchmark_mse": baseline_mse,
                    "test_ratio": ratio,
                    "ratio_status": status,
                    "max_train_certificate_gap": float(np.max(train.gap)),
                    "max_test_objective_gap": float(np.max(test.gap)),
                    "max_risk_error_indicator": float(np.max(risk.quadrature_error)),
                    "training_witness_norm": fit.training_witness_norm,
                    "seconds": time.perf_counter() - tic,
                }
                predictions = pd.DataFrame(
                    {
                        "model": name,
                        "family": family,
                        "repetition": repetition,
                        "parameter": float(parameter),
                        "split": ["train"] * n + ["interpolation"] * m + ["extrapolation"] * m,
                        "strike": all_strikes,
                        "truth": risk.value,
                        "prediction": np.r_[train.value, test.value],
                        "lower": np.r_[train.lower, test.lower],
                        "upper": np.r_[train.upper, test.upper],
                        "risk_error_indicator": risk.quadrature_error,
                        "solver_status": list(train.statuses) + list(test.statuses),
                        "es99": np.r_[np.full(n, np.nan), benchmark],
                    }
                )
                write_csv(pred_path, predictions)
                write_json(record_path, record)
            all_metrics.append(record)
            all_predictions.append(predictions)
        print(f"{name}: {family}, {len(draws[family])} repetitions complete", flush=True)
    diagnostics = {
        "model": name,
        "baseline_seconds": baseline_seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "gram_condition_number": geometry.condition_number
        if np.isfinite(geometry.condition_number)
        else None,
        "gram_rank": int(np.linalg.matrix_rank(geometry.gram)),
        "factor_residual": geometry.factor_residual,
        "residual_variance_corrections": geometry.negative_residual_corrections,
        "max_coordinate_value_error_bound": L
        * np.sqrt(max(geometry.negative_residual_corrections, default=0.0)),
    }
    if isinstance(model, Heston):
        logk = np.log(all_strikes)
        diagnostics["fourier_moment_refinement"] = {
            str(j): float(
                np.max(
                    np.abs(
                        model.truncated_moments(logk, j, True)
                        - model.truncated_moments(logk, j, False)
                    )
                )
            )
            for j in range(3)
        }
        diagnostics["heston_checks"] = model.inversion_checks
    write_json(checkpoint / "diagnostics.json", diagnostics)
    return (
        pd.DataFrame(all_metrics),
        pd.concat(all_predictions, ignore_index=True),
        config,
        diagnostics,
    )


def formatted_table(summary, model):
    sub = summary[summary.model == model]
    result = pd.DataFrame({"Risk measure": sub.risk_measure.to_numpy()})
    for col, label in [
        ("train_mse", "Train MSE"),
        ("test_mse", "Test MSE"),
        ("test_ratio", "Extrapolation TEST ratio"),
    ]:
        result[label] = [
            f"{mean:.3e} ± {sd:.3e}"
            if col != "test_ratio" or (0 < abs(mean) < 1e-3)
            else f"{mean:.3f} ± {sd:.3f}"
            for mean, sd in zip(sub[col + "_mean"], sub[col + "_sd"])
        ]
    return result


def run_study(
    out="results/paper", repetitions=30, n_train=100, seed=20261029, models=None, strike_seed=13579
):
    """Run or resume a single configuration; never mix sources or model sets."""
    models = (
        {"GBM": GBM(), "Heston": Heston(), "Finite": FiniteModel()} if models is None else models
    )
    classes = {"GBM": GBM, "Heston": Heston, "Finite": FiniteModel}
    if not models or any(
        name not in classes or not isinstance(model, classes[name])
        for name, model in models.items()
    ):
        raise ValueError("models must map GBM/Heston/Finite names to matching model objects.")
    draws = parameter_draws(repetitions, seed)
    with run_lock(out) as out:
        manifest = {
            "models": list(models),
            "repetitions": repetitions,
            "n_train": n_train,
            "parameter_seed": seed,
            "strike_seed": strike_seed,
            "source_hash": source_hash(),
            "runtime": runtime_versions(),
            "model_parameters": {name: model_config(model) for name, model in models.items()},
        }
        check_config(out / "run_manifest.json", manifest)
        metrics = []
        predictions = []
        configs = {}
        diagnostics = {}
        for name, model in models.items():
            raw, pred, config, diag = run_model(model, name, out, draws, n_train, strike_seed)
            metrics.append(raw)
            predictions.append(pred)
            configs[name] = config
            diagnostics[name] = diag
        raw = pd.concat(metrics, ignore_index=True)
        pred = pd.concat(predictions, ignore_index=True)
        summary = summarize(raw)
        write_csv(out / "replicate_metrics.csv", raw)
        write_csv(out / "predictions.csv", pred)
        write_csv(out / "summary.csv", summary)
        write_json(out / "config.json", configs)
        write_json(out / "diagnostics.json", diagnostics)
        write_json(out / "runtime_versions.json", runtime_versions())
        write_tables(summary, out, configs)
    return raw, summary, pred
