# Reproduction and outputs

## Main study

Run from a clean checkout with the setup in the root README. All input data are
generated from the explicit stock laws and seeds; there are no downloads at
experiment runtime. `configs/paper.json` selects 30 repetitions, 100 training
strikes, all three models, parameter seed `20261029`, and strike seed `13579`.

Four NumPy `SeedSequence` child streams generate independent parameters for ES,
power, exponential, and Wang, in that order. The same arrays are passed to
each stock model. The first two draws of a smoke run agree with the first two
draws of a paper run. Increasing repetitions extends each stream without
changing earlier parameters.

For GBM and Heston, training endpoints are the 10th and 90th terminal
quantiles. The OOD intervals extend each side by one quarter of the training
width, with the lower endpoint clamped at zero. The finite experiment keeps
the stock-independent strike bands: training $[0.1,0.9]$, left OOD $[0,0.1)$,
and right OOD $(0.9,1]$. Interior and OOD strikes are drawn once from the fixed
strike seed and reused. Numerical uniform sampling uses NumPy's half-open
interval convention; the endpoint differences have probability zero under
the continuous strike design.

`n_train` must be a positive multiple of 20. Each held-out split has
`n_train / 10` points, half of the extrapolation split on each side.
The complete default run contains 360 fits and 43,200 prediction rows.

For each draw, the reference is the latent family's risk at the same query
strike. The benchmark is $\mathrm{ES}_{0.99}$ of the **call loss**. Denoting
the extrapolation strikes by $\mathcal K$, the OOD ratio is

$$
\frac{|\mathcal K|^{-1}\sum_{K\in\mathcal K}
(\widehat\rho(X_K)-\rho(X_K))^2}
{|\mathcal K|^{-1}\sum_{K\in\mathcal K}
(\mathrm{ES}_{0.99}(-(S_1-K)_+)-\rho(X_K))^2},
$$

where the ES notation in this display follows the **payoff-risk convention**.
Equivalently its denominator uses upper-tail ES of the positive loss
$(S_1-K)_+$. The final table reports the arithmetic mean and sample SD
(`ddof=1`) of the per-draw ratios, not a ratio of aggregate MSEs.
For a zero denominator the record contains a null ratio with status
`undefined_0_over_0` or `infinite`. Such draws are not dropped, and their
aggregate ratio is marked undefined. The default study has positive benchmark
denominators.

## Files produced by a run

| File | Contents |
| --- | --- |
| `run_manifest.json` | Run identity, selected models, seeds, model parameters, source hash, runtime |
| `config.json` | Full model parameters, exact strike arrays, parameter draws, shared budget |
| `replicate_metrics.csv` | One row per model/family/draw, MSEs, ratio, status, error indicators, timing |
| `predictions.csv` | One row per model/family/draw/strike, latent label, predicted risk, lower/upper objectives, solver status, ES99 |
| `summary.csv` | Mean and sample SD for each model/family; explicit undefined-ratio count |
| `table_<model>.tex` | Complete `table` environment; requires `booktabs` |
| `table_<model>.csv` | Human-readable formatted table |
| `diagnostics.json` | Geometry rank/condition, residual corrections, integration checks, timings |
| `runtime_versions.json` | Python, platform, numerical package versions, thread settings |
| `validation.json` | Result of recomputing metrics and checking numerical identities |
| `checkpoints/<model>/` | Atomic per-draw records and predictions for resuming |

Training `es99` entries are intentionally blank: the benchmark is needed only
on held-out queries. Model names are `GBM`, `Heston`, `Finite`; risk-family keys
are `es`, `power`, `exponential`, `wang`; main-study split names are `train`,
`interpolation`, `extrapolation`.

`validate` recomputes MSEs, ratios, means and sample SDs from the prediction
rows, verifies the recorded parameters and strike arrays, checks solver
objective gaps, and checks the finite cash-translation identity. It raises
an error when a file is incomplete or inconsistent. It does not treat a
successful solver status as sufficient on its own.

## Checkpoints and reproducibility

Use separate output directories for different studies. The manifest prevents
reuse after a source, configuration, model set, or numerical environment
change. Source hashing covers the installed package's Python modules.
Completed pairs of per-draw JSON/CSV checkpoints are reused; an incomplete
pair is recomputed. Writes use a temporary file followed by replacement.

One process owns an output directory at a time. A `.run.lock` file prevents
concurrent writers. If a process is forcibly killed, verify it has stopped
before deleting that lock and resuming the same command. A normal exception
or interruption releases the lock automatically.

Rerunning the same model with different libraries can change the last digits,
especially when an exact-arithmetic error is zero. The pinned environment
helps reproduce the continuous table values; discrete errors around $10^{-30}$
are machine-level residuals, not statistically meaningful target values.
Use absolute tolerances for this case instead of a relative comparison of
two tiny floating-point MSEs.

## Perturbation settings

`configs/ablation.json` fixes alpha at 0.75, gamma on `0,0.1,...,1`, 30 paired
repetitions, 100 random training strikes, 10 random test strikes, and seed
`20261029`. Stocks are uniform on `0,0.1,...,0.9`. A draw has independent
training/test streams, reused across gamma. The optional random-dual stream
is also reused across gamma for that draw.

One common $L=\sqrt3(1+10^{-12})$ is a valid reduced bound for the ES baseline
at alpha 0.75. It stays fixed across gamma and draws. The positive-gamma data
are deliberately incompatible; the repo does not estimate a nonexistent
Lipschitz interpolant from pairwise slopes. The plots show mean MSE with one
sample SD. Raw training errors, including baseline solver roundoff, are kept.

`--compare-random-dual` adds separately named approximation rows and curves.
`--random-directions` changes their count. It can be expensive at 100,000
directions; reduce it for exploratory runs and use a fresh output directory.

## Publishing and license

This bundle is a complete repository working tree, ready to initialize or
copy into the paper's intended GitHub repository. It was not published by
this bundle creation workflow. The address in the draft is
`AnastasisKratsios/RiskMeasureReconstruction`; its remote contents were not
available for comparison. Do not replace an existing checkout blindly.

For a new empty repository, after reviewing the files:

```bash
git init -b main
git add .
git commit -m "Add reproducible risk reconstruction experiments"
git remote add origin https://github.com/AnastasisKratsios/RiskMeasureReconstruction.git
git push -u origin main
```

If the remote already has content, clone it and incorporate this tree on a
branch, preserving existing history and files. The `.github` workflow is
ready for GitHub Actions; local checks do not imply a remote CI run has occurred.

No software license was specified in the supplied material. This bundle
therefore does not assign a license on the authors' behalf. Add the authors'
chosen license before distributing it as licensed open-source software.
Publication identifiers can be added to `CITATION.cff` when available.

