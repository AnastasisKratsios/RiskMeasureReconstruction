# Alignment with the supplied manuscript

This companion follows the supplied manuscript titled **Reconstructing risk
preferences from data**, including its updated 1,000-state discrete model.
The draft source was used for reference and has not been silently edited or
included in the repository. The following points matter when interpreting
or updating its numerical section.

## Main tables and solver description

The main text describes conic optimization with primal–dual certificates,
matching the supplied implementation and this repository's main tables.
The computational appendix still says all experiments use a randomized
approximation with 100,000 directions. These are different algorithms.

The repository defaults to the conic reconstruction. Its optional random-dual
implementation uses sphere directions plus zero and reports separate results.
The appendix can retain the approximation as an optional computational
variant, but should not attribute the main tables to it. Suggested replacement:

> The reported main experiments evaluate the conic dual and independently
> check feasible primal and dual objective values. The finite collection of
> supporting hyperplanes described below is an optional lower approximation;
> it is not used to generate Tables 1–3 and need not interpolate training data.

## Discrete stock support and strike support

The updated stock law is exactly
$\{1+9i/999:i=0,\ldots,999\}$ with equal probabilities. The original strike
design is retained: training $[0.1,0.9]$, interpolation inside it, and OOD
$[0,0.1)\cup(0.9,1]$. Therefore every queried short-call payoff is
$X_K=K-S_1$, including the endpoint $S_1=K=1$.

The discrete errors are zero in real arithmetic as a direct consequence of
cash additivity and interpolation. The table records unrounded residual MSEs
of order $10^{-30}$. Suggested explanatory sentence for the manuscript:

> In the discrete model all queried strikes satisfy $K\le S_1$ almost surely,
> so $X_K=K-S_1$. The observed test errors are at floating-point roundoff:
> cash additivity and interpolation imply exact recovery throughout these
> strike intervals in real arithmetic.

Changing the stock support alone changes the economic experiment. Moving
both stocks and strikes by the same positive affine map would leave the
relative MSE ratio invariant when the reconstruction is scaled consistently.
The updated finite design should not be used as evidence of learning
extrapolation across payoff kinks. GBM and Heston keep their original design.

## Short-call sign convention

The main numerical section correctly uses $X_K=-(S_1-K)_+$. The ablation
paragraph later calls $(S_1-K)_+$ a short-position payoff, omitting the minus
sign, and ES is also written on the positive loss in some prose. This repo
uses one convention throughout: short **payoff** $X_K=-(S_1-K)_+$, positive
**loss** $-X_K=(S_1-K)_+$, and `weighted_es_loss` evaluates upper-tail loss ES.
Under the abstract payoff-risk convention, write the benchmark as
$\operatorname{ES}_{0.99}(X_K)$; if writing ES on the positive call, explicitly
identify it as a loss functional.

## Ablation: what is specified and what is new

The draft specifies stocks $\{i/10:i=0,\ldots,9\}$, uniform random strikes in
$(0,1)$, and quotes
$y_K=\operatorname{ES}_{\alpha}(X_K)+\gamma(1-K^2)/2$
using the payoff-risk convention. It does not specify a reproducible alpha,
gamma grid, repetition count, random seeds, or reconstruction budget for the
displayed plots. The provided implementation's older generic helper uses
`linspace(0,1,10)`, which is a different stock law.

The repository's `ablation` command uses the manuscript's `arange(10)/10`
law and declares the missing settings in `configs/ablation.json` and
[the reproduction guide](reproduction.md). Its two figures are **new results
under those explicit choices**. The original figure files and underlying
values were not supplied, so equality with the draft's quoted error orders
is not asserted. Review the new summaries before updating that paragraph.

In the included run, the mean train/test MSE at $\gamma=1$ is approximately
$0.09151/0.09579$, rather than the draft's quoted $10^{-7}/10^{-6}$ scale.
The baseline $\gamma=0$ has train MSE at roundoff and mean test MSE about
$1.697\times10^{-7}$. These results reflect the explicitly chosen fixed
budget and the conflicting zero-payoff labels above; they should not be
presented as reproducing the draft's unspecified ablation settings.

A concave perturbation in strike alone is not a general certificate that
arbitrary sampled payoff quotes violate convexity. Here a concrete witness
is available: within any interval between adjacent stock states, $X_K$ is
affine in $K$. Three sampled strikes $a<k<b$ in that interval satisfy a
payoff convex-combination identity, while the quote's excess over the endpoint
chord is $\gamma(k-a)(b-k)/2>0$. The code records actual sampled witnesses.

There is an additional incompatibility for this particular ten-state law:
for $K>0.9$ the payoff is identically zero, but the positive-gamma quote
$\gamma(1-K^2)/2$ varies with $K$. Two such training strikes give conflicting
values for the same payoff. `duplicate_label_gap` records this failure.
The repo does not claim that every conceivable small random dataset has
one of these witnesses; the observed counts are reported.

A suitable caption for the new plots is:

> Mean training and test MSE, with one sample standard deviation over 30
> paired strike draws, for quadratic perturbations of ES-at-0.75 call-risk
> quotes. The stock is uniform on $\{i/10:i=0,\ldots,9\}$, with 100 training
> and 10 test strikes drawn uniformly in $(0,1)$ and a fixed reduced budget
> $L=\sqrt3(1+10^{-12})$. The conic reconstruction is evaluated with
> primal–dual objective checks; positive-gamma incompatibility is assessed
> using sampled payoff identities and repeated-payoff labels.

