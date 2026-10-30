import json
import subprocess
import sys

import numpy as np
import pytest

from risk_measure_reconstruction import FiniteModel, GBM
from risk_measure_reconstruction.ablation import run_ablation, validate_ablation
from risk_measure_reconstruction.cli import main
from risk_measure_reconstruction.population_experiment import (
    parameter_draws,
    run_study,
    strike_design,
    mse_ratio,
)
from risk_measure_reconstruction.validation import validate_results


def test_seed_pairing_and_fixed_finite_strikes():
    small, full = parameter_draws(2), parameter_draws(30)
    for family in small:
        np.testing.assert_array_equal(small[family], full[family][:2])
    finite = strike_design(FiniteModel())
    np.testing.assert_array_equal(finite["train"], np.linspace(0.1, 0.9, 100))
    assert len(finite["interpolation"]) == len(finite["extrapolation"]) == 10
    assert np.sum(finite["extrapolation"] < 0.1) == 5
    assert np.sum(finite["extrapolation"] > 0.9) == 5
    continuous = strike_design(GBM())
    np.testing.assert_allclose(continuous["train"][[0, -1]], GBM().quantile([0.1, 0.9]))
    with pytest.raises(ValueError, match="multiple of 20"):
        strike_design(FiniteModel(), n_train=21)
    assert mse_ratio(0.0, 0.0) == (None, "undefined_0_over_0")
    assert mse_ratio(1.0, 0.0) == (None, "infinite")
    assert mse_ratio(1e-30, 2.0) == (5e-31, "finite")


def test_complete_run_resume_and_corruption_detection(tmp_path):
    out = tmp_path / "study"
    raw, summary, pred = run_study(out, repetitions=2, n_train=20, models={"Finite": FiniteModel()})
    report = validate_results(out)
    assert report["draw_count"] == 8
    assert report["prediction_count"] == 8 * 24
    assert report["minimum_benchmark_mse"] > 0
    for _, row in summary.iterrows():
        values = raw[raw.family == row.family].test_ratio
        assert row.test_ratio_sd == pytest.approx(np.std(values, ddof=1))
    before = (out / "checkpoints/Finite/es_00.json").stat().st_mtime_ns
    run_study(out, repetitions=2, n_train=20, models={"Finite": FiniteModel()})
    assert (out / "checkpoints/Finite/es_00.json").stat().st_mtime_ns == before
    assert validate_results(out)["passed"]
    with pytest.raises(ValueError, match="new output directory"):
        run_study(out, repetitions=3, n_train=20, models={"Finite": FiniteModel()})
    assert not (out / ".run.lock").exists()
    raw.loc[0, "test_mse"] = 42
    raw.to_csv(out / "replicate_metrics.csv", index=False)
    with pytest.raises(ValueError, match="test_mse"):
        validate_results(out)


def test_table_uses_actual_repetition_count_and_cli_errors(tmp_path):
    out = tmp_path / "cli"
    assert main(["smoke", "--models", "Finite", "--out", str(out)]) == 0
    text = (out / "table_finite.tex").read_text()
    assert "over 2 parameter draws" in text
    assert r"\begin{table}" in text and r"\pm" in text
    assert main(["tables", "--input", str(out)]) == 0
    assert main(["validate", "--input", str(out)]) == 0
    assert (
        main(["reproduce", "--models", "Finite", "--n-train", "21", "--out", str(tmp_path / "bad")])
        == 2
    )
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps({"repetitons": 2}))
    assert main(["smoke", "--config", str(bad_config), "--out", str(tmp_path / "bad2")]) == 2


def test_ablation_paper_states_and_real_incompatibility(tmp_path):
    raw, summary, points = run_ablation(
        tmp_path,
        repetitions=2,
        n_train=30,
        n_test=3,
        gammas=[0, 0.5, 1.0],
        compare_random_dual=True,
        random_directions=50,
    )
    config = json.loads((tmp_path / "run_manifest.json").read_text())
    np.testing.assert_array_equal(config["states"], np.arange(10) / 10)
    assert raw.L.nunique() == 1
    assert raw[(raw.gamma == 0) & (raw.method == "conic")].train_mse.max() < 1e-14
    positive = raw[(raw.gamma > 0) & (raw.method == "conic")]
    assert ((positive.jensen_gap > 1e-12) | (positive.duplicate_label_gap > 1e-12)).all()
    assert validate_ablation(tmp_path)["passed"]
    assert (tmp_path / "gamma_vs_training_mse.pdf").is_file()
    assert (tmp_path / "gamma_vs_test_mse.png").is_file()


def test_installed_module_entry_point_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "risk_measure_reconstruction", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "reproduce" in result.stdout
