# Reconstructing risk preferences from data

Numerical companion to **Reconstructing risk preferences from data**, by
Anastasis Kratsios, Silvana M. Pesenti, and Philipp Zimmermann.

Reconstruct a convex, cash-additive risk functional from finitely many option
risk quotes, then evaluate it at unseen strikes. This repository includes
the GBM, Heston, and discrete experiments; four latent risk families;
checked conic reconstruction; a separately labelled random-dual approximation;
CSV results; LaTeX tables; and a reproducible perturbation study.

## Start here

Use **Python 3.12** for the recorded reference environment. The package supports
Python 3.11 and later; CI is configured for 3.11–3.13. No GPU, proprietary solver,
market data, API key, or external dataset is required.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e '.[dev]'
risk-reconstruct smoke --models Finite --out results/first-run
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
For a regular installation without the reference version pins, use
`python -m pip install -e '.[dev]'` after activating the environment.

The smoke run is intentionally small: two parameter draws per risk family and
20 training strikes. It validates the saved predictions and produces
`results/first-run/table_finite.tex`, `summary.csv`, and `validation.json`.

## Reproduce the paper experiments

```bash
# All three models, four families, 30 draws per family, 100 training strikes.
risk-reconstruct reproduce --config configs/paper.json --out results/paper

# Or run just the updated discrete experiment.
risk-reconstruct reproduce --models Finite --out results/finite

# Recompute metrics from saved per-strike predictions and check solver brackets.
risk-reconstruct validate --input results/paper

# Regenerate the paper-ready LaTeX tables; requires \usepackage{booktabs}.
risk-reconstruct tables --input results/paper

# Compare with the supplied manuscript's rounded table values.
python scripts/compare_paper.py --input results/paper
```

Run the identical command again to resume its checkpoints. A changed source,
configuration, dependency version, or model set requires a **new output
directory**. Commands work from any directory after installation; paths to
JSON configuration files are relative to the working directory.

`python -m risk_measure_reconstruction ...` is equivalent to `risk-reconstruct ...`.
Use `risk-reconstruct --help` or `risk-reconstruct doctor` for command help and
runtime versions. The CLI defaults BLAS thread counts to one unless already set.

## What is being measured?

The observations are short-call **payoffs**
$X_K=-(S_1-K)_+$ and labels $y_K=\rho(X_K)$. Risk increases with the **loss**
$-X_K$. The reconstruction uses a fixed common Lipschitz budget and the original
simplex formula, evaluated using conic optimization with independent primal and
dual objective checks. GBM and Heston use their full terminal laws with
deterministic numerical integration; they are not replaced by sampled stocks.

| Model | Terminal stock law | Training strikes |
| --- | --- | --- |
| GBM | $S_0=1$, drift $0.01$, volatility $0.01$, horizon $1$ | 100 equally spaced points between the 10th and 90th terminal quantiles |
| Heston | $(S_0,v_0)=(1,0.04)$, drift $0.01$, $\kappa=2$, $\theta=0.04$, $\xi=0.3$, correlation $-0.7$, horizon $1$ | Same quantile rule under the Heston law |
| Discrete | Uniform on $\{1+9i/999:i=0,\ldots,999\}$ | 100 equally spaced points in $[0.1,0.9]$ |

Each family uses the same parameter draws across the three stock models.
Each model holds its training, interpolation, and extrapolation strike sets
fixed across all draws. Test MSE uses 10 interpolation strikes. OOD Test Ratio
is calculated **within each draw** as the extrapolation MSE divided by the
ES-at-0.99 benchmark MSE, then summarized by its mean and sample standard
deviation. There is no denominator floor or clipping of small errors.

**Interpret the discrete case carefully.** Its strikes lie in $[0,1]$ while
$S_1\ge1$, so $X_K=K-S_1$. All these payoffs differ only by cash. Once one quote
is interpolated, cash additivity gives exact reconstruction throughout the
queried range. Errors near $10^{-30}$ are floating-point roundoff; this case
checks cash additivity and does not demonstrate extrapolation across an
option payoff kink. [A separate example](examples/finite_state.py) uses strikes
crossing the stock support.

## Perturbation study

```bash
risk-reconstruct ablation --config configs/ablation.json --out results/ablation
```

This writes `gamma_vs_training_mse.pdf` and `gamma_vs_test_mse.pdf`, PNG previews,
per-draw metrics, predictions, and compatibility witnesses. Copy the PDFs to the
paper's `Graphics/Ablation/` directory. The ablation uses the manuscript's
separate ten-state law $S_1\in\{i/10:i=0,\ldots,9\}$.

The supplied draft leaves several ablation settings unspecified. This repo
chooses and records them in `configs/ablation.json`; the figures are a new,
fully specified run, not a claim to recover the draft's unavailable plot data.
See [paper alignment](docs/paper_alignment.md) before replacing those figures.

To compare with the appendix's approximation, add `--compare-random-dual`.
This samples 100,000 sphere directions plus zero by default, and reports its
errors separately from the conic solution.

## Use the reconstruction in Python

```python
import numpy as np
from risk_measure_reconstruction import ConicReconstructor, weighted_es_loss

# Rows are payoff observations; columns are terminal states.
p = np.array([0.2, 0.3, 0.5])
X = np.array([[-1.0, 0.0, 1.0], [-2.0, -1.0, 2.0]])
y = weighted_es_loss(-X, p, alpha=0.75)
fit = ConicReconstructor(X, y, p, L=2.0)
prediction = fit.predict(np.array([[-0.5, 0.2, 0.8]]))
print(prediction.value, prediction.lower, prediction.upper)
```

The general reconstruction is convex, cash-additive, and Lipschitz. Exact
interpolation requires compatible labels and a sufficient budget. The package
does not apply the paper's separate monotone-hull construction or enforce
normalization. An arbitrary output must not be described as automatically
monotone or coherent.

For a notebook walkthrough, install `python -m pip install -e '.[notebook]'`
and open [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb).

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/risk_measure_reconstruction/` | Installable numerical package and CLI |
| `configs/` | Explicit paper, smoke, and ablation settings |
| `tests/` | Independent numerical identities, solver comparisons, and workflow checks |
| `reference/` | Supplied paper values and fresh, verified repository runs |
| `docs/method.md` | Formula, sign conventions, risk families, and numerical limits |
| `docs/reproduction.md` | Output schema, seeds, checkpoints, verification, and publishing |
| `docs/paper_alignment.md` | Draft inconsistencies and proposed clarifications |
| `examples/finite_state.py` | Small example with strikes crossing the stock support |
| `.github/workflows/tests.yml` | Tests, all-model smoke run, and package build in CI |

Run `pytest -q`, `ruff check .`, `ruff format --check .`, and `python -m build`
to check the repository. See [verification](docs/verification.md) for the checks
actually performed for this bundle and the resulting numerical tolerances.

## Citation

Please cite the accompanying paper by Kratsios, Pesenti, and Zimmermann.
`CITATION.cff` records the supplied manuscript title and authors; no DOI or
publication venue has been invented. The paper source itself is not bundled.
See [release notes](docs/reproduction.md#publishing-and-license) for publishing
and licensing details.

