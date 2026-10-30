# Mathematical and numerical contract

## Payoffs, losses, and the reconstruction

`X` denotes a payoff and `-X` its loss. Cash additivity means
$\rho(X+c)=\rho(X)-c$. For a short call,
$X_K=-(S_1-K)_+$, so a positive option loss receives a positive risk quote.
No probability-normal or Gaussian assumption is imposed on an arbitrary payoff.

For training pairs $(X_i,y_i)$, set $\bar X_i=X_i-\mathbb E X_i$ and
$\tilde y_i=y_i+\mathbb E X_i$. The implemented reconstruction is

$$
\widehat\rho(X)=\min_{\ell\in\Delta_N}
\left\{\sum_{i=1}^N\ell_i\tilde y_i+
L\left\|X-\mathbb E X-\sum_{i=1}^N\ell_i\bar X_i\right\|_{L^2}\right\}
-\mathbb E X.
$$

Here $L$ is the Lipschitz budget **on the centered, cash-quotient space**.
A full-space bound is $\sqrt{1+L^2}$. The formula is convex, cash-additive,
and Lipschitz for every finite dataset, but it interpolates only if the labels
are compatible with this budget. Monotonicity, positive homogeneity, and
normalization are not enforced. The separate monotone-hull construction in
the paper is outside this package.

For a finite law with probabilities $p_b>0$, the Euclidean coordinates are
$w_i=\sqrt p\odot\bar X_i$ and $a=\sqrt p\odot(X-\mathbb E X)$.
The reduced dual is

$$
\max_{u,t}\{a^\top u+t:\|u\|_2\le L,\quad w_i^\top u+t\le\tilde y_i\text{ for all }i\}.
$$

Clarabel solves this second-order cone program. The implementation repairs
its candidate simplex weights to a feasible simplex point and its candidate
dual vector to the norm ball, then independently evaluates the original
primal and dual objectives. `Prediction.value` is the feasible primal upper
value; `lower`, `upper`, `gap`, and `statuses` expose the check. Unacceptable
gaps raise an exception. A slightly negative gap within floating-point
tolerance is retained in the output instead of replaced by zero.

For call-family spectral labels, a common affine dual witness can certify all
training vertices simultaneously. `certify_training()` checks that witness's
norm and the objective gaps before returning the matching primal vertex
values. Thus the reported exact training zeros are certificate evaluations.
Query values still evaluate the reconstruction; latent test labels are never
substituted for predictions.

## Latent risk families

For $Y=-X$, the spectral definition is
$\rho(X)=\int_0^1F_Y^{-1}(u)\phi(u)\,du$. The four densities and reduced
Lipschitz bounds used by the experiment are:

| Family | Loss-quantile density $\phi(u)$ | Reduced bound |
| --- | --- | --- |
| ES | $\mathbf1_{[\alpha,1]}(u)/(1-\alpha)$ | $\sqrt{\alpha/(1-\alpha)}$ |
| Power | $r u^{r-1}$ | $(r-1)/\sqrt{2r-1}$ |
| Exponential | $\beta e^{\beta u}/(e^\beta-1)$ | $\sqrt{\beta/(2\tanh(\beta/2))-1}$ |
| Wang | $\exp(\lambda\Phi^{-1}(u)-\lambda^2/2)$ | $\sqrt{e^{\lambda^2}-1}$ |

`Phi` in Wang's distortion is the standard normal CDF **as part of the
definition of the risk functional**; it does not assert that the stock or
payoff is normally distributed. The power survival distortion is
$1-(1-v)^r$, the exponential one is
$(1-e^{-\beta v})/(1-e^{-\beta})$, and Wang's is
$\Phi(\Phi^{-1}(v)+\lambda)$.

The parameter intervals are ES $[0.5,0.9)$, power $[1.1,10)$, exponential
$[0.5,10)$, and Wang $[0.1,2)$. One common budget dominates all four boxes:
$L=\sqrt{e^4-1}(1+10^{-12})\approx7.321075742898133$.
It is not estimated separately for each draw.

Finite ES fills the upper tail of the loss distribution with exactly
$1-\alpha$ probability mass, including a fractional boundary atom.
For the other three families, integrated spectral masses are computed on the
true cumulative atom intervals. A quantile-threshold average that drops or
double-counts the boundary atom would give different answers.

## Population geometry and integration

GBM uses full-law lognormal moments. Call covariance is evaluated with
50-digit intermediate arithmetic before converting to float64, limiting
cancellation for the narrow-volatility experiment. The training covariance
matrix defines a Cholesky coordinate system. Each query includes its own
orthogonal residual norm; the construction preserves that query's norm and
inner products with the training claims. It is not a common joint embedding
of all query residuals. No stock grid or terminal-price Monte Carlo sample
replaces either continuous law.

GBM ES and Wang call risks use closed forms. Power and exponential risks use
adaptive, infinite-interval survival integration. SciPy's returned error
estimate is checked against absolute and relative tolerances.

Heston uses damped characteristic-function inversion for truncated moments,
survival probabilities, and terminal quantiles. Base Fourier quadrature has
768 nodes and cutoff 256; refinement uses twice the nodes and cutoff 320.
The payoff quadrature has 192 nodes, doubled in the refinement. A controlled
upper-tail truncation uses the eighth moment and the default spectral box.
Recorded diagnostics include refinement differences, probability corrections,
and the tail tolerance. Heston requires the Feller condition and the explicit
sufficient moment conditions enforced in `Heston.__post_init__`; arbitrary
Heston parameters are not supported by this inversion configuration.

`risk_error_indicator` is a quadrature/refinement indicator, **not a rigorous
interval enclosure** of every continuous-law numerical error. Solver brackets
bound the reconstruction in the computed geometry, separately from its
integration and moment errors. A zero indicator for an analytic expression
means no quadrature estimator is needed, not that floating-point rounding is
impossible. The Gram matrix is not repaired with diagonal jitter. Tiny negative
query residual variances are handled only within a scale-dependent roundoff
tolerance, with their magnitudes recorded.

## Optional random supporting planes

`RandomDualReconstructor` samples directions with norm $L$, includes the zero
direction, and computes the minimum admissible intercept for each one.
Its maximum over these affine functions is convex and cash-additive after
lifting, and is a lower approximation to the exact conic reconstruction.
It need not interpolate training labels. A legacy uniform-radius variant is
available explicitly with `sampling="radius"`; it is not uniform on the ball.
Neither variant generates the main tables.

Instances reuse a solver object and are not thread-safe. Independent instances
can be used for separate jobs. The CLI runs sequentially with deterministic
seeds and records the numerical environment.

