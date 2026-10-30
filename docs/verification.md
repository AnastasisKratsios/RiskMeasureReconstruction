# Verification of this repository bundle

Verified locally with Python 3.12 and the numerical versions recorded in
`requirements-lock.txt`. This is evidence from local execution; GitHub Actions
is configured but has not been run on a published remote by this workflow.

| Check | Result |
| --- | --- |
| Installation in a fresh virtual environment using the documented commands | Passed |
| `pip check` | No broken requirements |
| Test suite | **28 passed** |
| Ruff lint and formatting | Passed |
| Main experiment | **360 fits, 43,200 predictions**, all three models and all four families |
| Main output validation | Passed: raw MSEs, per-draw ratios, sample SDs, draws, strikes, solver brackets |
| Comparison with supplied manuscript | **72 checks passed**; continuous statistics agree at displayed precision |
| Discrete experiment | Reported table reproduced; maximum cash-translation residual about $7.11\times10^{-15}$ |
| All-model CLI smoke run | Passed: 24 fits, 576 predictions |
| Gamma study | **330 conic fits, 36,300 predictions**, outputs and figures generated and validated |
| Positive-gamma incompatibility witnesses | Present for all **300** positive-gamma conic fits in the recorded run |
| Optional random-dual path | Exercised separately by the workflow tests; lower-approximation property checked |
| Nontrivial finite-state example | Executed with strikes crossing the stock support |
| Notebook | All **4** ordinary Python cells executed; stored outputs and notebook schema validated |
| Package build | Source distribution and wheel built successfully |
| Installed wheel outside the source tree | Finite CLI smoke run passed |
| Figures | Both PNG previews visually inspected; PDF exports generated from the same figures |

The main run's largest recorded primal–dual objective gap was approximately
$1.13\times10^{-11}$; the smallest was $-5.33\times10^{-15}$, retained as
floating-point roundoff. Its largest risk-integration error indicator was
approximately $1.06\times10^{-10}$. All 360 benchmark MSEs were positive and
all ratios finite. The continuous covariance matrices had numerical rank
100, while the finite cash-translation design had rank 1.

The tests include finite ES compared with an independent linear program,
the conic solver compared with a separate one-dimensional primal optimizer,
cash additivity and convexity, integrated nonuniform atom masses, independent
GBM integration, the Heston zero-volatility-of-volatility limit, checkpoint
resumption, corrupted-output rejection, and actual payoff-based incompatibility
witnesses. These test mathematical behavior rather than only code paths.

The notebook was executed using `scripts/check_notebook.py`, which evaluates
its ordinary Python cells sequentially and records their real outputs. A
networked Jupyter kernel could not start in the verification host because
local socket/interface access was restricted. An interactive Jupyter front end
was therefore not verified here. The notebook has no magics or widgets; its
code also runs through the documented socket-free checker.

The ablation does **not** reproduce the error scale claimed for the draft's
unspecified original figure settings. Its actual settings and numerical
differences are explicitly documented in [paper alignment](paper_alignment.md).
Quadrature indicators are not formal interval-arithmetic certificates.
See [the mathematical contract](method.md) for the scope of all numerical checks.

To repeat the checks from a checkout:

```bash
pytest -q
ruff check .
ruff format --check .
python scripts/check_notebook.py
risk-reconstruct smoke --out results/smoke-check
risk-reconstruct reproduce --config configs/paper.json --out results/paper-check
python scripts/compare_paper.py --input results/paper-check
risk-reconstruct ablation --config configs/ablation.json --out results/ablation-check
python -m build
```

