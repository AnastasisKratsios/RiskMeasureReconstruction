"""Population GBM call risk and exact-in-real-arithmetic L2 geometry.

No empirical scenario law is substituted for GBM. See docs/method.md.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import warnings

import clarabel
import mpmath as mp
import numpy as np
from scipy import sparse
from scipy.integrate import IntegrationWarning, quad
from scipy.linalg import cholesky, solve_triangular
from scipy.special import log_ndtr, ndtr, ndtri

from .risk_reconstruction import ConicReconstructor, Prediction, _real_array, _scalar, _integer


FAMILY_ORDER = ("es", "power", "exponential", "wang")
FAMILY_NAMES = {
    "es": "Expected Shortfall",
    "power": "Power spectral",
    "exponential": "Exponential spectral",
    "wang": "Wang",
}
PARAMETER_NAMES = {"es": "alpha", "power": "r", "exponential": "beta", "wang": "lambda"}
DEFAULT_INTERVALS = {
    "es": (0.5, 0.9),
    "power": (1.1, 10.0),
    "exponential": (0.5, 10.0),
    "wang": (0.1, 2.0),
}


@dataclass(frozen=True)
class GBM:
    """dS=mu*S*dt+sigma*S*dW; mu is price drift, not log-price drift."""

    s0: float = 1.0
    mu: float = 0.01
    sigma: float = 0.01
    horizon: float = 1.0
    moment_digits: int = 50

    def __post_init__(self):
        for name in ("s0", "sigma", "horizon"):
            value = _scalar(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be strictly positive.")
        _scalar(self.mu, "mu", -np.inf)
        _integer(self.moment_digits, "moment_digits", 30)

    @property
    def eta(self):
        return math.log(self.s0) + (self.mu - 0.5 * self.sigma**2) * self.horizon

    @property
    def v(self):
        return self.sigma * math.sqrt(self.horizon)

    @property
    def mean(self):
        return self.s0 * math.exp(self.mu * self.horizon)

    @property
    def std(self):
        return self.mean * math.sqrt(math.expm1(self.v**2))

    def sample(self, size, seed=123):
        """Exact GBM terminal sampling, provided for diagnostics, not risk labels."""
        size = _integer(size, "size", 1)
        seed = _integer(seed, "seed", 0)
        with np.errstate(over="raise", invalid="raise"):
            values = np.exp(self.eta + self.v * np.random.default_rng(seed).normal(size=size))
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise FloatingPointError("GBM samples overflowed or underflowed.")
        return values

    def quantile(self, probability):
        u = _real_array(probability, "probability")
        if np.any((u <= 0) | (u >= 1)):
            raise ValueError("Quantile probabilities must lie strictly in (0,1).")
        with np.errstate(over="raise", invalid="raise"):
            result = np.exp(self.eta + self.v * ndtri(u))
        if not np.all(np.isfinite(result)) or np.any(result <= 0):
            raise FloatingPointError("GBM quantiles overflowed or underflowed.")
        return result

    def _mp_parameters(self):
        s0, mu, sigma, t = map(mp.mpf, map(str, (self.s0, self.mu, self.sigma, self.horizon)))
        return mp.log(s0) + (mu - sigma * sigma / 2) * t, sigma * mp.sqrt(t)

    def _moment_mp(self, order, strike):
        eta, v = self._mp_parameters()
        tail = (
            mp.mpf(1)
            if strike == 0
            else mp.erfc(-(eta + order * v * v - mp.log(strike)) / (v * mp.sqrt(2))) / 2
        )
        return mp.exp(order * eta + order * order * v * v / 2) * tail

    def truncated_moment(self, order, strike):
        order = _integer(order, "order", 0)
        strike = _scalar(strike, "strike")
        with mp.workdps(self.moment_digits):
            return float(self._moment_mp(order, mp.mpf(str(strike))))

    @lru_cache(maxsize=4096)
    def call_mean(self, strike):
        strike = _scalar(strike, "strike")
        with mp.workdps(self.moment_digits):
            k = mp.mpf(str(strike))
            return float(self._moment_mp(1, k) - k * self._moment_mp(0, k))

    @lru_cache(maxsize=32768)
    def call_covariance(self, strike_a, strike_b):
        ka = _scalar(strike_a, "strike_a")
        kb = _scalar(strike_b, "strike_b")
        with mp.workdps(self.moment_digits):
            a, b = mp.mpf(str(ka)), mp.mpf(str(kb))
            h = max(a, b)
            cross = (
                self._moment_mp(2, h)
                - (a + b) * self._moment_mp(1, h)
                + a * b * self._moment_mp(0, h)
            )
            ma = self._moment_mp(1, a) - a * self._moment_mp(0, a)
            mb = self._moment_mp(1, b) - b * self._moment_mp(0, b)
            return float(cross - ma * mb)


def validate_parameter(family, parameter):
    if family not in FAMILY_ORDER:
        raise ValueError(f"Unknown family {family!r}; choose {FAMILY_ORDER}.")
    p = _scalar(parameter, "parameter")
    if family == "es" and not p < 1:
        raise ValueError("ES alpha must lie in [0,1).")
    if family == "power" and p < 1:
        raise ValueError("Power r must be at least one.")
    if family == "exponential" and p <= 0:
        raise ValueError("Exponential beta must be positive.")
    return p


def spectrum(family, parameter, u):
    """Loss-quantile density; open-unit-interval arguments for Wang."""
    p = validate_parameter(family, parameter)
    u = _real_array(u, "u")
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("Spectrum evaluation requires 0<u<1.")
    if family == "es":
        return (u >= p).astype(float) / (1 - p)
    if family == "power":
        return p * u ** (p - 1)
    if family == "exponential":
        return p * np.exp(p * (u - 1)) / (-np.expm1(-p))
    return np.exp(p * ndtri(u) - 0.5 * p * p)


def reduced_lipschitz_bound(family, parameter):
    p = validate_parameter(family, parameter)
    if family == "es":
        square = p / (1 - p)
    elif family == "power":
        square = (p - 1) ** 2 / (2 * p - 1)
    elif family == "exponential":
        # Taylor expansion is used only to avoid cancellation near zero;
        # the omitted O(p^8) term is below float64 roundoff for this branch.
        square = (
            p * p / 12 - p**4 / 720 + p**6 / 30240 if p < 1e-3 else p / (2 * np.tanh(p / 2)) - 1
        )
    else:
        square = math.expm1(p * p)
    return math.sqrt(square)


def common_lipschitz_budget(intervals=None):
    intervals = DEFAULT_INTERVALS if intervals is None else intervals
    if set(intervals) != set(FAMILY_ORDER):
        raise ValueError("Supply a parameter interval for every risk family.")
    bounds = []
    for family in FAMILY_ORDER:
        pair = _real_array(intervals[family], f"{family} interval", 1)
        if pair.shape != (2,) or pair[0] >= pair[1]:
            raise ValueError("Parameter intervals must have two increasing endpoints.")
        bounds.extend(reduced_lipschitz_bound(family, x) for x in pair)
    return max(bounds) * (1 + 1e-12)


@dataclass
class RiskValues:
    value: np.ndarray
    quadrature_error: np.ndarray


def population_call_risk(gbm, strikes, family, parameter, *, epsabs=2e-13, epsrel=2e-11):
    """Risk of X_K=-(S_T-K)+ under the full GBM law.

    ES/Wang use closed forms; power/exponential use infinite-interval
    quadrature of distorted survival probabilities. Error estimates are
    checked and returned. Entropic is deliberately outside this finite API.
    """
    k = _real_array(strikes, "strikes", 1)
    if np.any(k < 0):
        raise ValueError("Strikes must be nonnegative.")
    p = validate_parameter(family, parameter)
    if hasattr(gbm, "spectral_call_risk"):
        return gbm.spectral_call_risk(k, family, p)
    epsabs = _scalar(epsabs, "epsabs")
    epsrel = _scalar(epsrel, "epsrel")
    if epsabs == 0 and epsrel == 0:
        raise ValueError("At least one quadrature tolerance must be positive.")
    result, errors = [], []
    for strike in k:
        if family == "es":
            with mp.workdps(gbm.moment_digits):
                eta, v = gbm._mp_parameters()
                kk, alpha = mp.mpf(str(float(strike))), mp.mpf(str(p))
                cutoff = (
                    mp.exp(eta + v * mp.sqrt(2) * mp.erfinv(2 * alpha - 1)) if alpha else mp.mpf(0)
                )
                h = max(kk, cutoff)
                value = float((gbm._moment_mp(1, h) - kk * gbm._moment_mp(0, h)) / (1 - alpha))
            error = 0.0
        elif family == "wang":
            shifted = GBM(
                gbm.s0,
                gbm.mu + p * gbm.sigma / math.sqrt(gbm.horizon),
                gbm.sigma,
                gbm.horizon,
                gbm.moment_digits,
            )
            value, error = shifted.call_mean(float(strike)), 0.0
        else:
            lower = -np.inf if strike == 0 else (math.log(strike) - gbm.eta) / gbm.v

            def integrand(z):
                if family == "power":
                    survival = -math.expm1(p * float(log_ndtr(z)))
                else:
                    survival = -math.expm1(-p * float(ndtr(-z))) / (-math.expm1(-p))
                if survival == 0:
                    return 0.0
                return math.exp(math.log(gbm.v) + gbm.eta + gbm.v * z + math.log(survival))

            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                value, error = quad(
                    integrand, lower, np.inf, epsabs=epsabs, epsrel=epsrel, limit=250
                )
            if not np.isfinite(value + error) or error > max(epsabs, epsrel * abs(value)):
                raise RuntimeError(f"Quadrature tolerance failed: value={value}, estimate={error}.")
        if not np.isfinite(value) or value < 0:
            raise RuntimeError("Nonfinite or negative population call risk.")
        result.append(value)
        errors.append(error)
    return RiskValues(np.asarray(result), np.asarray(errors))


class CallGeometry:
    """Training-span coordinates from population call covariance, with no grid."""

    def __init__(self, gbm, train_strikes):
        self.gbm = gbm
        self.strikes = _real_array(train_strikes, "train_strikes", 1).copy()
        if len(self.strikes) < 1 or np.any(self.strikes < 0) or np.any(np.diff(self.strikes) <= 0):
            raise ValueError("Training strikes must be nonnegative and strictly increasing.")
        n = len(self.strikes)
        self.gram = np.empty((n, n))
        for i, a in enumerate(self.strikes):
            for j in range(i + 1):
                self.gram[i, j] = self.gram[j, i] = gbm.call_covariance(
                    float(a), float(self.strikes[j])
                )
        try:
            self.factor = cholesky(self.gram, lower=True)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Call Gram matrix is not numerically positive definite; no jitter was added."
            ) from exc
        self.train_coordinates = np.column_stack([self.factor, np.zeros(n)])
        self.train_means = -np.array([gbm.call_mean(float(k)) for k in self.strikes])
        self.factor_residual = float(np.max(np.abs(self.factor @ self.factor.T - self.gram)))
        self.condition_number = float(np.linalg.cond(self.gram))
        self.negative_residual_corrections = []

    def coordinates(self, strikes):
        k = _real_array(strikes, "strikes", 1)
        if np.any(k < 0):
            raise ValueError("Strikes must be nonnegative.")
        coordinates, means = [], []
        for strike in k:
            exact = np.flatnonzero(self.strikes == strike)
            if len(exact):
                idx = int(exact[0])
                coordinates.append(self.train_coordinates[idx].copy())
                means.append(self.train_means[idx])
                continue
            c = np.array([self.gbm.call_covariance(float(x), float(strike)) for x in self.strikes])
            a = solve_triangular(self.factor, c, lower=True)
            variance = self.gbm.call_covariance(float(strike), float(strike))
            residual = float(variance - a @ a)
            tolerance = (
                100
                * len(a)
                * np.finfo(float).eps
                * max(variance, float(a @ a), np.finfo(float).tiny)
            )
            if residual < -tolerance:
                raise RuntimeError(
                    f"Negative query residual variance {residual:g}; tolerance {tolerance:g}."
                )
            if residual < 0:
                self.negative_residual_corrections.append(-residual)
                residual = 0.0
            coordinates.append(np.r_[a, math.sqrt(residual)])
            means.append(-self.gbm.call_mean(float(strike)))
        return np.asarray(coordinates).reshape((-1, len(self.strikes) + 1)), np.asarray(means)


class PopulationCallReconstructor(ConicReconstructor):
    """The existing conic reconstruction in population Hilbert coordinates.

    Reuses the original independent primal/dual objective checks. Scaling by
    GBM standard deviation is an invertible change of units, not a surrogate.
    """

    def __init__(self, geometry, y, L, gap_atol=2e-10, gap_rtol=2e-10):
        self.geometry = geometry
        y = _real_array(y, "y", 1)
        if y.shape != geometry.train_means.shape:
            raise ValueError("One risk quote is required per training strike.")
        self.L = _scalar(L, "L")
        self.scale = geometry.gbm.std
        self.gap_atol = _scalar(gap_atol, "gap_atol") / self.scale
        self.gap_rtol = _scalar(gap_rtol, "gap_rtol")
        self.N, self.B = geometry.train_coordinates.shape
        self.training_values = y.copy()
        self.W = (geometry.train_coordinates / self.scale).T.copy()
        self.y_tilde = (y + geometry.train_means) / self.scale
        self._solver = None
        if self.L > 0:
            self._build_solver()

    def _build_solver(self):
        n, b = self.N, self.B
        inequality = sparse.hstack([sparse.csc_matrix(self.W.T), np.ones((n, 1))])
        soc = sparse.vstack(
            [
                sparse.csc_matrix((1, b + 1)),
                sparse.hstack([-sparse.eye(b), sparse.csc_matrix((b, 1))]),
            ]
        )
        matrix = sparse.vstack([inequality, soc], format="csc")
        self._q = np.r_[np.zeros(b), -1.0]
        rhs = np.r_[self.y_tilde, self.L, np.zeros(b)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 250
        settings.max_threads = 1
        settings.tol_gap_abs = 1e-11
        settings.tol_gap_rel = 1e-11
        settings.tol_feas = 1e-11
        if b < n:
            # Finite-state quotes are highly redundant. Tighten the linear
            # refinement/solver target; retain the same independent gap test.
            settings.tol_gap_abs = 1e-12
            settings.tol_gap_rel = 1e-12
            settings.tol_feas = 1e-12
            settings.iterative_refinement_abstol = 1e-14
            settings.iterative_refinement_reltol = 1e-14
            settings.iterative_refinement_max_iter = 30
        settings.presolve_enable = False
        self._solver = clarabel.DefaultSolver(
            sparse.csc_matrix((b + 1, b + 1)),
            self._q,
            matrix,
            rhs,
            [clarabel.NonnegativeConeT(n), clarabel.SecondOrderConeT(b + 1)],
            settings,
        )

    def predict_coordinates(self, coordinates, means):
        a = _real_array(coordinates, "coordinates", 2)
        means = _real_array(means, "means", 1)
        if a.shape != (len(means), self.B):
            raise ValueError("Query coordinate and mean shapes do not match the geometry.")
        lows, ups, statuses = [], [], []
        for row, m in zip(a, means):
            lo, hi, status = self._solve(row / self.scale)
            lows.append(lo * self.scale - m)
            ups.append(hi * self.scale - m)
            statuses.append(status)
        lows, ups = np.asarray(lows), np.asarray(ups)
        return Prediction(ups.copy(), lows, ups, ups - lows, tuple(statuses))

    def _solve(self, a):
        # Independent feasible witnesses can resolve flat, redundant finite
        # faces more accurately than an interior-point stopping criterion.
        # These are bounds on the ORIGINAL simplex problem, not new quotes.
        if self.L > 0 and hasattr(self, "_training_dual"):
            delta = a[:, None] - self.W
            distance = np.linalg.norm(delta, axis=0)
            vertex = self.y_tilde + self.L * distance
            j = int(np.argmin(vertex))
            upper = float(vertex[j])
            u, t = self._training_dual
            lower = float(a @ u + t)
            if distance[j] > 0:
                supporting = self.L * (1 - 4 * np.finfo(float).eps) * delta[:, j] / distance[j]
                lower = max(
                    lower, float(a @ supporting + np.min(self.y_tilde - supporting @ self.W))
                )
            if self.N > 1:
                edge = self.W[:, 1:] - self.W[:, :-1]
                square = np.sum(edge * edge, axis=0)
                fraction = np.divide(
                    np.sum((a[:, None] - self.W[:, :-1]) * edge, axis=0),
                    square,
                    out=np.zeros_like(square),
                    where=square > 0,
                )
                fraction = np.clip(
                    fraction, 0, 1
                )  # simplex-feasible candidate, not an altered payoff
                point = self.W[:, :-1] + edge * fraction
                labels = (1 - fraction) * self.y_tilde[:-1] + fraction * self.y_tilde[1:]
                upper = min(
                    upper,
                    float(np.min(labels + self.L * np.linalg.norm(a[:, None] - point, axis=0))),
                )
            gap = upper - lower
            tolerance = self.gap_atol + self.gap_rtol * max(abs(upper), abs(lower))
            if -tolerance <= gap <= tolerance:
                return lower, upper, "PrimalDualCertificate"
        return super()._solve(a)

    def predict(self, strikes: np.ndarray) -> Prediction:
        return self.predict_coordinates(*self.geometry.coordinates(strikes))

    def certify_training(self) -> Prediction:
        """Evaluate training values by matching feasible primal/dual witnesses.

        Calls are comonotone, so each spectral risk is a single linear
        expectation on this family. A common affine dual witness can certify
        every vertex value at once. This is checked, not presumed for arbitrary
        labels. It evaluates the same reconstruction formula without N solves.
        """
        u, _, _, _ = np.linalg.lstsq(self.W.T, self.y_tilde, rcond=None)
        norm = float(np.linalg.norm(u))
        if norm > self.L * (1 + 1e-10):
            raise RuntimeError("Training affine witness exceeds the Lipschitz budget.")
        if norm > self.L:
            u *= self.L / norm
        u *= 1 - 4 * np.finfo(float).eps
        intercept = float(np.min(self.y_tilde - self.W.T @ u))
        self._training_dual = (u.copy(), intercept)
        lower = (self.W.T @ u + intercept) * self.scale - self.geometry.train_means
        upper = self.training_values.copy()
        gap = upper - lower
        if gap.min() < -2e-12 or gap.max() > 2e-10:
            raise RuntimeError("Affine training certificate does not bracket the labels closely.")
        self.training_witness_norm = norm
        return Prediction(
            upper.copy(), lower, upper, gap, tuple("AffineCertificate" for _ in upper)
        )
