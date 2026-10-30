"""Command-line entry points. Run ``risk-reconstruct --help``."""

import argparse
import json
import os
from pathlib import Path
import sys

PAPER_DEFAULTS = {
    "models": ["GBM", "Heston", "Finite"],
    "repetitions": 30,
    "n_train": 100,
    "parameter_seed": 20261029,
    "strike_seed": 13579,
}


def _load_config(path, allowed):
    if path is None:
        return {}
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or set(config) - set(allowed):
        raise ValueError(f"Configuration must be an object with keys from {sorted(allowed)}")
    return config


def main(argv=None):
    # Respect explicit user choices; use one BLAS thread by default.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    parser = argparse.ArgumentParser(
        description="Numerical companion to Reconstructing risk preferences from data"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("reproduce", "Run the paper design (30 draws per family by default)"),
        ("smoke", "Small end-to-end run (2 draws, 20 training strikes)"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument(
            "--config", type=Path, help="JSON settings; explicit flags override them"
        )
        command.add_argument("--models", nargs="+", choices=["GBM", "Heston", "Finite"])
        command.add_argument("--repetitions", type=int)
        command.add_argument("--n-train", type=int)
        command.add_argument("--parameter-seed", type=int)
        command.add_argument("--strike-seed", type=int)
        command.add_argument(
            "--out",
            type=Path,
            default=Path("results") / ("paper" if name == "reproduce" else "smoke"),
        )
    ablation = sub.add_parser(
        "ablation", help="Generate the documented paired gamma study and figures"
    )
    ablation.add_argument("--config", type=Path)
    ablation.add_argument("--out", type=Path, default=Path("results/ablation"))
    ablation.add_argument("--repetitions", type=int)
    ablation.add_argument("--gammas", nargs="+", type=float)
    ablation.add_argument("--seed", type=int)
    ablation.add_argument(
        "--compare-random-dual", action=argparse.BooleanOptionalAction, default=None
    )
    ablation.add_argument("--random-directions", type=int)
    for name, description in (
        ("validate", "Recompute metrics and verify saved numerical checks"),
        ("tables", "Regenerate the main-study LaTeX tables from a validated run"),
    ):
        command = sub.add_parser(name, help=description)
        command.add_argument("--input", type=Path, required=True)
    sub.add_parser("doctor", help="Print numerical dependency versions and thread settings")
    args = parser.parse_args(argv)
    try:
        from .provenance import runtime_versions, write_json

        if args.command == "doctor":
            print(json.dumps(runtime_versions(), indent=2))
        elif args.command in ("reproduce", "smoke"):
            from . import GBM, Heston, FiniteModel
            from .population_experiment import run_study, formatted_table
            from .validation import validate_results

            config = PAPER_DEFAULTS.copy()
            if args.command == "smoke":
                config |= {"repetitions": 2, "n_train": 20}
            config |= _load_config(args.config, PAPER_DEFAULTS)
            config |= {
                key: getattr(args, key) for key in PAPER_DEFAULTS if getattr(args, key) is not None
            }
            classes = {"GBM": GBM, "Heston": Heston, "Finite": FiniteModel}
            selected = config.pop("models")
            if (
                not isinstance(selected, list)
                or not selected
                or any(name not in classes for name in selected)
            ):
                raise ValueError("models must be a nonempty list drawn from GBM, Heston, Finite")
            if len(set(selected)) != len(selected):
                raise ValueError("models must not contain duplicates")
            _, summary, _ = run_study(
                args.out,
                models={name: classes[name]() for name in selected},
                seed=config.pop("parameter_seed"),
                **config,
            )
            report = validate_results(args.out)
            write_json(args.out / "validation.json", report)
            for name in selected:
                print(f"\n{name}\n{formatted_table(summary, name).to_string(index=False)}")
            print(f"\nValidated {report['draw_count']} draws; outputs: {args.out.resolve()}")
        elif args.command == "ablation":
            from .ablation import DEFAULT_ABLATION, run_ablation

            config = _load_config(args.config, DEFAULT_ABLATION)
            for key in (
                "repetitions",
                "gammas",
                "seed",
                "compare_random_dual",
                "random_directions",
            ):
                if getattr(args, key) is not None:
                    config[key] = getattr(args, key)
            run_ablation(args.out, **config)
            print(f"Validated ablation and figures: {args.out.resolve()}")
        else:
            manifest_path = args.input / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if manifest.get("study_type") == "ablation":
                from .ablation import validate_ablation

                if args.command == "tables":
                    raise ValueError(
                        "tables exports the main study; ablation produces figures and summary.csv"
                    )
                report = validate_ablation(args.input)
            else:
                from .validation import validate_results

                report = validate_results(args.input)
            if args.command == "tables":
                import pandas as pd
                from .reports import write_tables

                summary = pd.read_csv(args.input / "summary.csv", float_precision="round_trip")
                config = json.loads((args.input / "config.json").read_text())
                write_tables(summary, args.input, config)
            print(json.dumps(report, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
