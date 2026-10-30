"""Recompute reported errors from saved predictions and check certificates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .population_experiment import mse_ratio, summarize


def _require(condition, message):
    if not condition:
        raise ValueError(f"Validation failed: {message}")


def validate_results(directory):
    """Validate a complete main-study run; raise on corruption or missing draws.

    Solver brackets apply to the numerically represented reconstruction.
    Quadrature refinement indicators are reported separately, not interpreted
    as rigorous error enclosures for the continuous model.
    """
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    raw = pd.read_csv(directory / "replicate_metrics.csv", float_precision="round_trip")
    pred = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
    summary = pd.read_csv(directory / "summary.csv", float_precision="round_trip")
    _require(set(raw.model) == set(pred.model) == set(config), "model sets disagree")
    _require(not raw.duplicated(["model", "family", "repetition"]).any(), "duplicate metric rows")
    _require(
        not pred.duplicated(["model", "family", "repetition", "split", "strike"]).any(),
        "duplicate predictions",
    )
    for column in ("strike", "truth", "prediction", "lower", "upper", "risk_error_indicator"):
        _require(np.isfinite(pred[column]).all(), f"nonfinite {column}")
    _require(
        np.array_equal(pred.prediction, pred.upper), "predictions differ from primal upper values"
    )
    gap = pred.upper - pred.lower
    tolerance = 2e-10 * (1 + np.maximum(np.abs(pred.upper), np.abs(pred.lower)))
    _require(((gap >= -tolerance) & (gap <= tolerance)).all(), "objective gaps exceed tolerance")
    _require((pred.risk_error_indicator >= 0).all(), "negative quadrature error indicator")
    expected_keys = set()
    max_cash_residual = 0.0
    for model, model_config in config.items():
        for family, draws in model_config["draws"].items():
            for rep, parameter in enumerate(draws):
                key = (model, family, rep)
                expected_keys.add(key)
                block = pred[
                    (pred.model == model) & (pred.family == family) & (pred.repetition == rep)
                ]
                metric = raw[
                    (raw.model == model) & (raw.family == family) & (raw.repetition == rep)
                ]
                _require(len(metric) == 1, f"missing metric {key}")
                row = metric.iloc[0]
                _require(
                    row.parameter == parameter and (block.parameter == parameter).all(),
                    f"parameter mismatch {key}",
                )
                _require(
                    set(block.split) == {"train", "interpolation", "extrapolation"},
                    f"split mismatch {key}",
                )
                errors = {}
                for split, strikes in model_config["strikes"].items():
                    points = block[block.split == split].sort_values("strike")
                    _require(
                        np.array_equal(points.strike.to_numpy(), np.sort(strikes)),
                        f"strike mismatch {key} {split}",
                    )
                    errors[split] = float(np.mean((points.prediction - points.truth) ** 2))
                ood = block[block.split == "extrapolation"]
                _require(np.isfinite(ood.es99).all(), f"missing ES99 benchmark {key}")
                benchmark = float(np.mean((ood.es99 - ood.truth) ** 2))
                ratio, status = mse_ratio(errors["extrapolation"], benchmark)
                _require(row.ratio_status == status, f"ratio status mismatch {key}")
                expected = {
                    "train_mse": errors["train"],
                    "test_mse": errors["interpolation"],
                    "extrap_mse": errors["extrapolation"],
                    "benchmark_mse": benchmark,
                    "test_ratio": np.nan if ratio is None else ratio,
                    "max_train_certificate_gap": float(
                        gap.loc[block[block.split == "train"].index].max()
                    ),
                    "max_test_objective_gap": float(
                        gap.loc[block[block.split != "train"].index].max()
                    ),
                    "max_risk_error_indicator": float(block.risk_error_indicator.max()),
                }
                for name, value in expected.items():
                    _require(
                        np.isclose(row[name], value, rtol=2e-9, atol=1e-38, equal_nan=True),
                        f"{name} is inconsistent with predictions for {key}",
                    )
                if (
                    model == "Finite"
                    and min(model_config["parameters"]["states"]) >= block.strike.max()
                ):
                    for column in ("truth", "prediction"):
                        values = block[column] + block.strike
                        max_cash_residual = max(max_cash_residual, float(np.ptp(values)))
    actual_keys = set(raw[["model", "family", "repetition"]].itertuples(index=False, name=None))
    pred_keys = set(pred[["model", "family", "repetition"]].itertuples(index=False, name=None))
    _require(expected_keys == actual_keys == pred_keys, "missing or unexpected draws")
    recomputed = summarize(raw)
    keys = ["model", "family"]
    left, right = summary.set_index(keys).sort_index(), recomputed.set_index(keys).sort_index()
    _require(left.index.equals(right.index), "summary rows disagree")
    for column in right.select_dtypes(include="number"):
        _require(
            np.allclose(left[column], right[column], rtol=2e-9, atol=1e-38, equal_nan=True),
            f"summary {column} is inconsistent",
        )
    _require(max_cash_residual < 1e-10, "finite cash-additivity residual exceeds tolerance")
    return {
        "passed": True,
        "draw_count": len(raw),
        "prediction_count": len(pred),
        "max_objective_gap": float(gap.max()),
        "min_objective_gap": float(gap.min()),
        "minimum_benchmark_mse": float(raw.benchmark_mse.min()),
        "nonfinite_ratio_count": int((raw.ratio_status != "finite").sum()),
        "max_risk_error_indicator": float(pred.risk_error_indicator.max()),
        "finite_cash_additivity_residual": max_cash_residual,
        "checks": [
            "draws and strikes",
            "per-draw MSEs and ratios",
            "sample SDs (ddof=1)",
            "primal/dual objective gaps",
            "finite cash-additivity",
        ],
    }
