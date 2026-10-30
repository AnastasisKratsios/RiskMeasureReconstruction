"""Check a validated main run against the supplied manuscript's rounded tables."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from risk_measure_reconstruction.validation import validate_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference/paper_reported.csv",
    )
    args = parser.parse_args()
    validate_results(args.input)
    actual = pd.read_csv(args.input / "summary.csv", float_precision="round_trip")
    expected = pd.read_csv(args.reference, float_precision="round_trip")
    if not (actual.repetitions == 30).all():
        raise ValueError("Paper comparison requires 30 parameter draws per family.")
    config = json.loads((args.input / "config.json").read_text())
    if any(len(model["strikes"]["train"]) != 100 for model in config.values()):
        raise ValueError("Paper comparison requires 100 training strikes.")
    combined = actual.merge(
        expected, on=["model", "family"], suffixes=("_actual", "_paper"), validate="one_to_one"
    )
    if len(combined) != len(actual):
        raise ValueError("Unmatched model/family in the reference.")
    rows = []
    for _, row in combined.iterrows():
        for metric in (
            "train_mse_mean",
            "train_mse_sd",
            "test_mse_mean",
            "test_mse_sd",
            "test_ratio_mean",
            "test_ratio_sd",
        ):
            value, reference = float(row[metric + "_actual"]), float(row[metric + "_paper"])
            if row.model == "Finite":
                tolerance, check = 1e-24, "absolute roundoff scale"
                matches = abs(value) <= tolerance
            elif reference == 0:
                tolerance, check = 1e-18, "training interpolation tolerance"
                matches = abs(value) <= tolerance
            elif "ratio" in metric:
                tolerance, check = 0.00050000001, "manuscript rounding interval"
                matches = abs(value - reference) <= tolerance
            else:
                tolerance = 0.00050000001 * 10 ** math.floor(math.log10(abs(reference)))
                check = "manuscript rounding interval"
                matches = abs(value - reference) <= tolerance
            rows.append(
                {
                    "model": row.model,
                    "family": row.family,
                    "metric": metric,
                    "paper_value": reference,
                    "reproduced_value": value,
                    "absolute_difference": abs(value - reference),
                    "tolerance": tolerance,
                    "check": check,
                    "passed": bool(matches),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.input / "paper_comparison.csv", index=False)
    report = {
        "passed": bool(comparison.passed.all()),
        "checks": len(comparison),
        "failed_checks": int((~comparison.passed).sum()),
        "finite_policy": "Absolute threshold 1e-24; tiny machine residuals are not compared relatively.",
    }
    (args.input / "paper_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print(comparison[~comparison.passed].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
