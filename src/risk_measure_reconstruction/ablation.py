"""Explicit, paired gamma ablation for the manuscript's ten-state law.

The manuscript does not fix alpha, seeds, repetitions, gamma grid or L.
The defaults here are disclosed choices; these are newly generated figures,
not a claimed bitwise reproduction of the draft's unprovided figure data.
"""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .risk_reconstruction import (
    ConicReconstructor,
    RandomDualReconstructor,
    _integer,
    _scalar,
    _real_array,
    es_reduced_lipschitz_bound,
    short_call_payoffs,
    quote_values,
    jensen_witness,
)
from .provenance import check_config, run_lock, runtime_versions, source_hash, write_csv, write_json

DEFAULT_ABLATION = {
    "repetitions": 30,
    "n_train": 100,
    "n_test": 10,
    "alpha": 0.75,
    "gammas": [i / 10 for i in range(11)],
    "seed": 20261029,
    "compare_random_dual": False,
    "random_directions": 100000,
}


def summarize_ablation(raw):
    metrics = ["train_mse", "test_mse", "max_solver_gap", "jensen_gap", "duplicate_label_gap"]
    result = raw.groupby(["gamma", "method"], sort=True)[metrics].agg(["mean", "std"])
    result.columns = [f"{a}_{b}" for a, b in result.columns]
    result = result.reset_index()
    result["repetitions"] = raw.groupby(["gamma", "method"], sort=True).size().to_numpy()
    return result


def plot_ablation(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    for split, filename in (("train", "gamma_vs_training_mse"), ("test", "gamma_vs_test_mse")):
        fig, ax = plt.subplots(figsize=(5.4, 3.5), layout="constrained")
        for method, block in summary.groupby("method", sort=False):
            block = block.sort_values("gamma")
            label = "Conic reconstruction" if method == "conic" else "Random dual approximation"
            ax.errorbar(
                block.gamma,
                block[f"{split}_mse_mean"],
                yerr=block[f"{split}_mse_std"],
                marker="o",
                markersize=3,
                capsize=2.5,
                linewidth=1.3,
                label=label,
            )
        ax.set(xlabel=r"Perturbation $\gamma$", ylabel=f"{split.capitalize()} MSE")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=9)
        for extension in ("pdf", "png"):
            fig.savefig(out / f"{filename}.{extension}", dpi=180)
        plt.close(fig)


def run_ablation(out="results/ablation", **options):
    unknown = set(options) - set(DEFAULT_ABLATION)
    if unknown:
        raise ValueError(f"Unknown ablation options: {sorted(unknown)}")
    config = DEFAULT_ABLATION | options
    for key, minimum in (
        ("repetitions", 2),
        ("n_train", 3),
        ("n_test", 1),
        ("seed", 0),
        ("random_directions", 0),
    ):
        config[key] = _integer(config[key], key, minimum)
    config["alpha"] = _scalar(config["alpha"], "alpha", 0, 1)
    if config["alpha"] == 1:
        raise ValueError("alpha must be less than one")
    gamma = _real_array(config["gammas"], "gammas", 1)
    if not len(gamma) or np.any((gamma < 0) | (gamma > 1)) or len(np.unique(gamma)) != len(gamma):
        raise ValueError("gammas must be distinct numbers in [0,1]")
    config["gammas"] = sorted(gamma.tolist())
    if not isinstance(config["compare_random_dual"], bool):
        raise ValueError("compare_random_dual must be boolean")
    states = np.arange(10, dtype=float) / 10
    p = np.full(10, 0.1)
    L = es_reduced_lipschitz_bound(p, config["alpha"]) * (1 + 1e-12)
    config |= {
        "states": states.tolist(),
        "probabilities": p.tolist(),
        "L": L,
        "random_sampling": "sphere_plus_zero",
        "source_hash": source_hash(),
        "runtime": runtime_versions(),
        "study_type": "ablation",
    }
    all_metrics, all_predictions = [], []
    with run_lock(out) as out:
        check_config(out / "run_manifest.json", config)
        streams = np.random.SeedSequence(config["seed"]).spawn(config["repetitions"])
        for repetition, stream in enumerate(streams):
            train_stream, test_stream, dual_stream = stream.spawn(3)
            train_k = np.sort(np.random.default_rng(train_stream).uniform(size=config["n_train"]))
            test_k = np.sort(np.random.default_rng(test_stream).uniform(size=config["n_test"]))
            dual_seed = int(dual_stream.generate_state(1)[0])
            k = np.r_[train_k, test_k]
            x = short_call_payoffs(k, states)
            n = config["n_train"]
            for index, gamma in enumerate(config["gammas"]):
                stem = out / "checkpoints" / f"draw_{repetition:03d}_gamma_{index:03d}"
                record_path, prediction_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
                if record_path.exists() and prediction_path.exists():
                    rows = json.loads(record_path.read_text())
                    points = pd.read_csv(prediction_path, float_precision="round_trip")
                else:
                    y = quote_values(k, x, p, gamma, config["alpha"])
                    witness = jensen_witness(train_k, y[:n], states, gamma)
                    inactive = train_k >= states[-1]
                    duplicate_gap = float(np.ptp(y[:n][inactive])) if inactive.sum() > 1 else 0.0
                    start = time.perf_counter()
                    fit = ConicReconstructor(x[:n], y[:n], p, L)
                    pred = fit.predict(x)
                    elapsed = time.perf_counter() - start
                    common = {
                        "repetition": repetition,
                        "gamma": gamma,
                        "alpha": config["alpha"],
                        "L": L,
                        "max_solver_gap": float(pred.gap.max()),
                        "min_solver_gap": float(pred.gap.min()),
                        "duplicate_label_gap": duplicate_gap,
                        **witness,
                    }
                    rows, point_frames = [], []
                    methods = [("conic", pred.value, elapsed)]
                    if config["compare_random_dual"]:
                        start = time.perf_counter()
                        approx = RandomDualReconstructor(
                            x[:n],
                            y[:n],
                            p,
                            L,
                            config["random_directions"],
                            dual_seed,
                            sampling="sphere",
                        ).predict(x)
                        methods.append(("random_dual", approx, time.perf_counter() - start))
                    for method, values, seconds in methods:
                        rows.append(
                            common
                            | {
                                "method": method,
                                "seconds": seconds,
                                "train_mse": float(np.mean((values[:n] - y[:n]) ** 2)),
                                "test_mse": float(np.mean((values[n:] - y[n:]) ** 2)),
                            }
                        )
                        point_frames.append(
                            pd.DataFrame(
                                {
                                    "repetition": repetition,
                                    "gamma": gamma,
                                    "method": method,
                                    "split": ["train"] * n + ["test"] * config["n_test"],
                                    "strike": k,
                                    "truth": y,
                                    "prediction": values,
                                    "conic_lower": pred.lower,
                                    "conic_upper": pred.upper,
                                }
                            )
                        )
                    points = pd.concat(point_frames, ignore_index=True)
                    write_csv(prediction_path, points)
                    write_json(record_path, rows)
                all_metrics.extend(rows)
                all_predictions.append(points)
            print(f"Ablation: draw {repetition + 1}/{config['repetitions']} complete", flush=True)
        raw = pd.DataFrame(all_metrics)
        points = pd.concat(all_predictions, ignore_index=True)
        summary = summarize_ablation(raw)
        write_csv(out / "replicate_metrics.csv", raw)
        write_csv(out / "predictions.csv", points)
        write_csv(out / "summary.csv", summary)
        write_json(out / "runtime_versions.json", runtime_versions())
        plot_ablation(summary, out)
        report = validate_ablation(out)
        write_json(out / "validation.json", report)
    return raw, summary, points


def validate_ablation(out):
    from .validation import _require

    out = Path(out)
    config = json.loads((out / "run_manifest.json").read_text())
    raw = pd.read_csv(out / "replicate_metrics.csv", float_precision="round_trip")
    points = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
    summary = pd.read_csv(out / "summary.csv", float_precision="round_trip")
    methods = ["conic", "random_dual"] if config["compare_random_dual"] else ["conic"]
    expected_keys = {
        (r, g, m) for r in range(config["repetitions"]) for g in config["gammas"] for m in methods
    }
    keys = ["repetition", "gamma", "method"]
    _require(not raw.duplicated(keys).any(), "duplicate ablation metrics")
    _require(
        set(raw[keys].itertuples(index=False, name=None)) == expected_keys, "ablation draw mismatch"
    )
    _require(
        set(points[keys].itertuples(index=False, name=None)) == expected_keys,
        "ablation prediction mismatch",
    )
    _require(
        np.isfinite(points.select_dtypes(include="number")).all().all(), "nonfinite ablation values"
    )
    gap = points.conic_upper - points.conic_lower
    tol = 2e-8 * (1 + np.maximum(abs(points.conic_upper), abs(points.conic_lower)))
    _require(((gap >= -tol) & (gap <= tol)).all(), "ablation solver bracket failure")
    for _, row in raw.iterrows():
        group = points[
            (points.repetition == row.repetition)
            & (points.gamma == row.gamma)
            & (points.method == row.method)
        ]
        for split, count in (("train", config["n_train"]), ("test", config["n_test"])):
            part = group[group.split == split]
            _require(len(part) == count, "ablation split size mismatch")
            expected = np.mean((part.prediction - part.truth) ** 2)
            _require(
                np.isclose(expected, row[f"{split}_mse"], rtol=2e-9, atol=1e-38),
                "ablation MSE mismatch",
            )
        if row.method == "random_dual":
            _require(
                (group.prediction <= group.conic_upper + 2e-8).all(),
                "random dual exceeds conic upper bound",
            )
    expected = summarize_ablation(raw).sort_values(["gamma", "method"]).reset_index(drop=True)
    summary = summary.sort_values(["gamma", "method"]).reset_index(drop=True)
    _require(len(expected) == len(summary), "ablation summary row mismatch")
    for column in expected.select_dtypes(include="number"):
        _require(
            np.allclose(expected[column], summary[column], rtol=2e-9, atol=1e-38),
            f"ablation summary {column} mismatch",
        )
    baseline = raw[(raw.gamma == 0) & (raw.method == "conic")]
    _require(
        (baseline.train_mse < 1e-14).all(), "ES baseline does not interpolate within tolerance"
    )
    return {
        "passed": True,
        "metric_rows": len(raw),
        "prediction_rows": len(points),
        "max_solver_gap": float(gap.max()),
        "positive_gamma_draws_with_incompatibility_witness": int(
            (
                (raw.gamma > 0)
                & (raw.method == "conic")
                & ((raw.jensen_gap > 1e-12) | (raw.duplicate_label_gap > 1e-12))
            ).sum()
        ),
    }
