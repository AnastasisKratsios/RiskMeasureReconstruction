"""Continuous Heston terminal law by damped characteristic-function inversion.

This is deterministic population quadrature, not a Monte Carlo scenario law.
Both Fourier and payoff quadratures have independent refinement diagnostics.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, roots_legendre

from .risk_reconstruction import _real_array, _scalar, _integer
from .gbm_risk import RiskValues, validate_parameter


@dataclass(frozen=True)
class Heston:
    s0: float = 1.0
    mu: float = 0.01
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    horizon: float = 1.0
    fourier_nodes: int = 768
    fourier_cutoff: float = 256.0
    payoff_nodes: int = 192
    tail_tolerance: float = 2e-13

    def __post_init__(self):
        for name in ("s0", "kappa", "theta", "v0", "horizon", "fourier_cutoff", "tail_tolerance"):
            if _scalar(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive.")
        _scalar(self.xi, "xi")
        _scalar(self.mu, "mu", -np.inf)
        if not -1 < _scalar(self.rho, "rho", -1, 1) < 1:
            raise ValueError("rho must lie strictly between -1 and 1.")
        _integer(self.fourier_nodes, "fourier_nodes", 64)
        _integer(self.payoff_nodes, "payoff_nodes", 32)
        if self.xi > 0 and 2 * self.kappa * self.theta < self.xi**2:
            raise ValueError(
                "This implementation requires the Feller condition 2*kappa*theta>=xi^2."
            )
        if self.xi:
            for q in (-1, 8):
                b = self.rho * self.xi * q - self.kappa
                if b >= 0 or b * b - self.xi**2 * (q * q - q) <= 0:
                    raise ValueError(
                        "Parameters fail a sufficient moment condition for the Fourier contours."
                    )

    def characteristic(self, u):
        """E exp(i*u*log(S_T)), stable negative-exponential Heston form."""
        u = np.asarray(u, dtype=complex)
        if self.xi == 0:
            integrated = (
                self.theta * self.horizon
                + (self.v0 - self.theta) * (-np.expm1(-self.kappa * self.horizon)) / self.kappa
            )
            return np.exp(
                1j * u * (math.log(self.s0) + self.mu * self.horizon)
                - 0.5 * (u * u + 1j * u) * integrated
            )
        b = self.kappa - self.rho * self.xi * 1j * u
        d = np.sqrt(b * b + self.xi**2 * (u * u + 1j * u))
        g = (b - d) / (b + d)
        decay = np.exp(-d * self.horizon)
        c = 1j * u * (
            math.log(self.s0) + self.mu * self.horizon
        ) + self.kappa * self.theta / self.xi**2 * (
            (b - d) * self.horizon - 2 * (np.log1p(-g * decay) - np.log1p(-g))
        )
        dv = (b - d) / self.xi**2 * (-np.expm1(-d * self.horizon)) / (1 - g * decay)
        return np.exp(c + dv * self.v0)

    def moment(self, order):
        if not isinstance(order, (int, np.integer)) or not 0 <= order <= 8:
            raise ValueError("Only integer moments from zero through eight are supported.")
        if self.xi:
            b = self.rho * self.xi * 8 - self.kappa
            if b >= 0 or b * b - self.xi**2 * (64 - 8) <= 0:
                raise ValueError("Parameters fail the sufficient eighth-moment condition.")
        value = complex(self.characteristic(-1j * order))
        if (
            not np.isfinite(value.real + value.imag)
            or abs(value.imag) > 1e-10 * abs(value.real)
            or value.real <= 0
        ):
            raise ValueError(
                "Requested Heston moment is not finite on the implemented transform branch."
            )
        return value.real

    @property
    def mean(self):
        return self.s0 * math.exp(self.mu * self.horizon)

    @property
    def std(self):
        return math.sqrt(self.moment(2) - self.mean**2)

    @cached_property
    def inversion_checks(self):
        # An eighth moment gives a population L2 tail bound for all spectra.
        # This condition also avoids using an invalid damped Fourier contour.
        q = 8.0
        b = self.rho * self.xi * q - self.kappa
        if self.xi and (b >= 0 or b * b - self.xi**2 * (q * q - q) <= 0):
            raise ValueError(
                "Parameters do not meet the sufficient all-time eighth-moment condition."
            )
        return {
            "eighth_moment": self.moment(8),
            "max_probability_projection": 0.0,
            "max_payoff_refinement": 0.0,
        }

    @lru_cache(maxsize=16)
    def _fourier_rule(self, q, refined):
        self.inversion_checks
        n = self.fourier_nodes * (2 if refined else 1)
        cutoff = self.fourier_cutoff * (1.25 if refined else 1)
        x, w = roots_legendre(n)
        t = (x + 1) * cutoff / 2
        weights = w * cutoff / (2 * np.pi)
        return t, weights * self.characteristic(t - 1j * q)

    def truncated_moments(self, log_strikes, order=0, refined=True):
        """E[S^order 1_{S>K}], using q=8 on the right and q=-1 on the left."""
        if not isinstance(order, (int, np.integer)) or not 0 <= order <= 2:
            raise ValueError("Truncated moments support orders zero, one, and two.")
        x = _real_array(log_strikes, "log_strikes")
        shape = x.shape
        x = x.ravel()
        out = np.empty_like(x)
        for right, q in ((True, 8.0), (False, -1.0)):
            ids = np.flatnonzero((x >= math.log(self.s0)) == right)
            if not len(ids):
                continue
            t, w = self._fourier_rule(q, refined)
            coefficient = w / (q - order + 1j * t)
            for start in range(0, len(ids), 256):
                ix = ids[start : start + 256]
                integral = np.real(np.exp(-1j * x[ix, None] * t) @ coefficient)
                tail = np.exp((order - q) * x[ix]) * integral
                out[ix] = tail if right else self.moment(order) + tail
        return out.reshape(shape)

    def survival(self, log_strikes, refined=True):
        raw = self.truncated_moments(log_strikes, 0, refined)
        fixed = np.clip(raw, 0.0, 1.0)
        correction = float(np.max(np.abs(fixed - raw))) if raw.size else 0.0
        self.inversion_checks["max_probability_projection"] = max(
            self.inversion_checks["max_probability_projection"], correction
        )
        if correction > 2e-10:
            raise RuntimeError(f"Fourier probabilities outside [0,1] by {correction:g}.")
        return fixed

    def quantile(self, probabilities, refined=True):
        probabilities = _real_array(probabilities, "probabilities")
        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValueError("Quantile probabilities must lie strictly in (0,1).")
        out = []
        center = math.log(self.s0) + self.mu * self.horizon
        for u in probabilities.ravel():

            def objective(x):
                return float(self.survival(np.array(x), refined)) - (1 - float(u))

            a, b = center - 1.0, center + 1.0
            for _ in range(10):
                if objective(a) >= 0 and objective(b) <= 0:
                    break
                a -= 1.0
                b += 1.0
            else:
                raise RuntimeError("Could not bracket the Heston quantile.")
            out.append(math.exp(brentq(objective, a, b, xtol=2e-13, rtol=2e-13)))
        return np.asarray(out).reshape(probabilities.shape)

    @lru_cache(maxsize=4096)
    def _moments_at(self, strike):
        strike = _scalar(strike, "strike")
        if strike == 0:
            return tuple(self.moment(j) for j in range(3))
        return tuple(float(self.truncated_moments(np.array(math.log(strike)), j)) for j in range(3))

    @lru_cache(maxsize=4096)
    def call_mean(self, strike):
        m0, m1, _ = self._moments_at(strike)
        return m1 - strike * m0

    @lru_cache(maxsize=32768)
    def call_covariance(self, strike_a, strike_b):
        a, b = float(strike_a), float(strike_b)
        m0, m1, m2 = self._moments_at(max(a, b))
        return m2 - (a + b) * m1 + a * b * m0 - self.call_mean(a) * self.call_mean(b)

    @cached_property
    def log_tail_cutoff(self):
        # ||phi||_2 <= exp(2) on the default parameter box. Requiring the
        # endpoint <=2 for Wang is also enforced in spectral_call_risk below.
        return (
            math.log(
                math.exp(2)
                * math.sqrt(self.inversion_checks["eighth_moment"])
                / self.tail_tolerance
            )
            / 3
        )

    @lru_cache(maxsize=12)
    def _risk_grid(self, strikes_tuple, refined):
        k = np.asarray(strikes_tuple)
        if np.any(k <= 0):
            raise ValueError("Heston spectral quadrature currently requires positive strikes.")
        n = self.payoff_nodes * (2 if refined else 1)
        x, w = roots_legendre(n)
        low = np.log(k)
        high = self.log_tail_cutoff
        if np.any(low >= high):
            raise ValueError("Strike exceeds the controlled Heston quadrature domain.")
        nodes = low[:, None] + (x + 1) * (high - low[:, None]) / 2
        weights = w * (high - low[:, None]) / 2 * np.exp(nodes)
        probabilities = self.survival(nodes, refined=refined)
        return weights, probabilities

    def spectral_call_risk(self, strikes, family, parameter):
        p = validate_parameter(family, parameter)
        if family == "es":
            values = []
            for refined in (False, True):
                q = float(self.quantile(np.array(p), refined)) if p else 0.0
                h = np.maximum(strikes, q)
                positive = h > 0
                m0 = np.ones(len(h))
                m1 = np.full(len(h), self.mean)
                for order, destination in ((0, m0), (1, m1)):
                    destination[positive] = self.truncated_moments(
                        np.log(h[positive]), order, refined
                    )
                values.append((m1 - strikes * m0) / (1 - p))
            difference = np.abs(values[1] - values[0])
            if np.any(difference > 2e-9):
                raise RuntimeError("Heston ES refinement discrepancy exceeds 2e-9.")
            return RiskValues(values[1], difference)
        if (family == "wang" and p > 2) or (family in ("power", "exponential") and p > 10):
            raise ValueError("This tail bound is validated for Wang<=2 and power/exponential<=10.")
        values = []
        for refined in (False, True):
            weights, survival = self._risk_grid(tuple(map(float, strikes)), refined)
            with np.errstate(divide="ignore", invalid="raise"):
                if family == "power":
                    distorted = -np.expm1(p * np.log1p(-survival))
                elif family == "exponential":
                    distorted = -np.expm1(-p * survival) / (-np.expm1(-p))
                else:
                    distorted = ndtr(ndtri(survival) + p)
            values.append(np.sum(weights * distorted, axis=1))
        difference = np.abs(values[1] - values[0])
        largest = float(difference.max()) if difference.size else 0.0
        self.inversion_checks["max_payoff_refinement"] = max(
            self.inversion_checks["max_payoff_refinement"], largest
        )
        if largest > 2e-9:
            raise RuntimeError(
                f"Heston payoff/Fourier refinement difference too large: {largest:g}."
            )
        return RiskValues(values[1], difference + self.tail_tolerance)
