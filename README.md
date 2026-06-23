# Toy Implementation: Universal Risk-Measure Reconstruction

This notebook implements the toy experiment from **Universal Risk Measures: Near-Optimal Reconstruction of Cash-Additive Convex Risk Measures from Finite Data**.

The experiment studies finite-data reconstruction of a cash-additive convex risk measure from elicited option-risk quotes.  We work on a finite state space, query short European call positions, generate synthetic user responses from strike-dependent Expected Shortfall levels, and reconstruct a globally defined risk functional using the paper’s finite-state reconstruction formula.

## What the notebook does

The notebook:

1. builds a finite state space $\Omega=\{0,1/(B-1),\ldots,1\}$;
2. queries short-call payoffs $X_K=-(S_1-K)_+$;
3. generates hidden strike-dependent risk preferences through $q_\Lambda(K)$;
4. computes synthetic quotes $y_K=\operatorname{ES}_{q_\Lambda(K)}(-X_K)$;
5. removes the cash direction via $z=x-(p^\top x)\mathbf 1$;
6. reconstructs the reduced convex risk functional by solving the simplex problem;
7. lifts the reconstruction back to a cash-additive risk measure;
8. runs ablations in $\Lambda$, $B$, and $N_{\mathrm{train}}$;
9. saves figures, CSV summaries, and paper-ready LaTeX tables.

## How to run

Open the notebook in Google Colab and run all cells.

For a quick smoke test, keep

`RUN_QUICK_TEST=True`.

For fuller experiments, set

`RUN_QUICK_TEST=False`.

The main outputs are written to `OUTPUT_DIR`, including:

- PDF figures in `figures/`;
- raw ablation CSV files;
- summarized CSV files;
- LaTeX tables;
- a bundled zip archive `toy_es_reconstruction_outputs.zip`.

## Main outputs

The notebook produces the figures and tables used to illustrate that the reconstruction remains stable across:

- increasing heterogeneity of the hidden Expected Shortfall level;
- increasing finite-state dimension;
- increasing number of elicited training quotes.

The key qualitative message is that the formula acts as a stable post-processing map: from any internally consistent finite quote dictionary, it produces a globally defined cash-additive convex reconstruction with controlled regularity.

## Dependencies

The notebook uses:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `tqdm`

No deep-learning framework is required.
