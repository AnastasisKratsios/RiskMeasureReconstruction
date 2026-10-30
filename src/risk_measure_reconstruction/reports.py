"""Paper-ready LaTeX from raw mean/sample-standard-deviation statistics."""

import math
from pathlib import Path

from .provenance import write_text, write_csv


def latex_number(value, ratio=False):
    if not math.isfinite(value):
        return r"\mathrm{undefined}"
    if value == 0:
        return "0"
    if ratio and abs(value) >= 1e-3:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.3e}".split("e")
    return mantissa + r"\times 10^{" + str(int(exponent)) + "}"


def write_tables(summary, out, configs=None):
    from .population_experiment import formatted_table

    out = Path(out)
    for model in summary.model.unique():
        sub = summary[summary.model == model]
        if len(set(sub.repetitions)) != 1:
            raise ValueError("Each family must have the same repetition count for a paper table.")
        repetitions = int(sub.repetitions.iloc[0])
        lines = [
            r"\begin{table}[ht]",
            r"    \centering",
            r"    \begin{tabular}{lccc}",
            r"        \toprule",
            r"        Risk measure & Train MSE & Test MSE & OOD Test Ratio \\",
            r"        \midrule",
        ]
        for _, row in sub.iterrows():
            entries = []
            for key in ("train_mse", "test_mse", "test_ratio"):
                entries.append(
                    "$"
                    + latex_number(row[key + "_mean"], key == "test_ratio")
                    + r"\pm "
                    + latex_number(row[key + "_sd"], key == "test_ratio")
                    + "$"
                )
            lines.append("        " + row.risk_measure + " & " + " & ".join(entries) + r" \\")
        caption = f"{model}: mean $\\pm$ one sample standard deviation over {repetitions} parameter draws."
        finite = (configs or {}).get("Finite", {})
        states = finite.get("parameters", {}).get("states", [])
        probabilities = finite.get("parameters", {}).get("probabilities", [])
        paper_finite = (
            len(states) == 1000
            and states[0] == 1
            and states[-1] == 10
            and all(abs(p - 0.001) < 1e-15 for p in probabilities)
            and len(probabilities) == 1000
        )
        if model == "Finite" and paper_finite:
            caption += (
                r" $S_1$ is uniform over $10^3$ equally spaced points in $[1,10]$."
                r" Training strikes lie in $[0.1,0.9]$ and OOD strikes in $[0,0.1)\cup(0.9,1]$."
                r" Errors are at floating-point roundoff; cash-additivity gives zero error in exact arithmetic."
            )
        lines += [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{" + caption + "}",
            r"    \label{tab:risk-" + model.lower() + "}",
            r"\end{table}",
        ]
        write_text(out / f"table_{model.lower()}.tex", "\n".join(lines) + "\n")
        write_csv(out / f"table_{model.lower()}.csv", formatted_table(summary, model))
