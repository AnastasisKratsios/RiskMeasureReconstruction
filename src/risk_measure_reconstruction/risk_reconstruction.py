"""Finite-state call-quote experiments and the paper's reconstruction formula.

All numerical work is CPU float64. The conic reference solves the mathematical
formula to a checked tolerance; random duals are a separately labelled
approximation. See docs/method.md for assumptions, signs, and objective brackets.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional
import time

import clarabel
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist


def _real_array(value, name, ndim=None):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers, not {a.dtype}.")
    a = np.asarray(a, dtype=np.float64)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite.")
    return a


def _scalar(value, name, lo=0.0, hi=np.inf):
    a = _real_array(value, name, 0)
    result = float(a)
    if not lo <= result <= hi:
        raise ValueError(f"{name} must lie in [{lo}, {hi}].")
    return result


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _probabilities(p, B):
    p = _real_array(p, "p", 1)
    if p.shape != (B,) or np.any(p <= 0):
        raise ValueError("p must have B strictly positive entries.")
    if not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("p must sum to one; probabilities are not silently normalized.")
    return p


def quadratic_quote(K):
    """g(K)=(1-K^2)/2 on [0,1], evaluated in a stable factored form."""
    K = _real_array(K, "K")
    if np.any((K < 0) | (K > 1)):
        raise ValueError("Strikes K must lie in [0,1].")
    return 0.5 * (1.0 - K) * (1.0 + K)


def short_call_payoffs(K, states):
    """Return X[n,b]=-(states[b]-K[n])+; K shape (N,), states shape (B,)."""
    K = _real_array(K, "K", 1)
    quadratic_quote(K)  # domain validation
    states = _real_array(states, "states", 1)
    if states.size < 2 or np.any((states < 0) | (states > 1)):
        raise ValueError("At least two states in [0,1] are required.")
    if np.any(np.diff(states) <= 0):
        raise ValueError("States must be strictly increasing.")
    return -np.maximum(states[None, :] - K[:, None], 0.0)


def weighted_es_loss(losses, p, alpha):
    """Upper-tail ES for a finite distribution, including fractional atoms.

    losses has shape (..., B); p is a strictly positive probability vector.
    A tail mass 1-alpha is filled starting at the largest loss. This is the
    exact finite-distribution ES definition in real arithmetic, not a quantile
    threshold approximation that drops or duplicates boundary mass.
    """
    losses = _real_array(losses, "losses")
    if losses.ndim < 1:
        raise ValueError("losses must have a state axis.")
    p = _probabilities(p, losses.shape[-1])
    alpha = _scalar(alpha, "alpha", 0.0, 1.0)
    if alpha == 1.0:
        raise ValueError("alpha must be strictly less than one.")
    order = np.argsort(-losses, axis=-1, kind="stable")
    sorted_loss = np.take_along_axis(losses, order, axis=-1)
    sorted_p = np.take_along_axis(np.broadcast_to(p, losses.shape), order, axis=-1)
    before = np.concatenate(
        [np.zeros_like(sorted_p[..., :1]), np.cumsum(sorted_p, axis=-1)[..., :-1]], axis=-1
    )
    tail_mass = 1.0 - alpha
    mass = np.minimum(sorted_p, np.maximum(tail_mass - before, 0.0))
    return np.sum((mass / tail_mass) * sorted_loss, axis=-1)


def es_reduced_lipschitz_bound(p, alpha):
    """Valid upper bound on Lip(ES_alpha(-X) restricted to the cash quotient)."""
    p = _real_array(p, "p", 1)
    p = _probabilities(p, len(p))
    alpha = _scalar(alpha, "alpha", 0.0, 1.0)
    if alpha == 1.0:
        raise ValueError("alpha must be strictly less than one.")
    return float(np.sqrt(min(alpha / (1.0 - alpha), 1.0 / np.min(p) - 1.0)))


def quote_values(K, X, p, gamma, alpha=0.75):
    """Call-family quotes, not a global latent functional on arbitrary X."""
    gamma = _scalar(gamma, "gamma", 0.0, 1.0)
    K = _real_array(K, "K", 1)
    X = _real_array(X, "X", 2)
    if X.shape[0] != K.size:
        raise ValueError("X and K must have the same number of rows.")
    p = _probabilities(p, X.shape[1])
    return weighted_es_loss(-X, p, alpha) + gamma * quadratic_quote(K)


def sample_call_data(N_train=100, N_test=10, B=10, seed=123):
    N_train = _integer(N_train, "N_train", 1)
    N_test = _integer(N_test, "N_test", 1)
    B = _integer(B, "B", 2)
    seed = _integer(seed, "seed", 0)
    train_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
    K_train = np.sort(np.random.default_rng(train_seed).uniform(size=N_train))
    K_test = np.sort(np.random.default_rng(test_seed).uniform(size=N_test))
    states = np.linspace(0.0, 1.0, B)
    p = np.full(B, 1.0 / B)
    return {
        "K_train": K_train,
        "K_test": K_test,
        "states": states,
        "p": p,
        "X_train": short_call_payoffs(K_train, states),
        "X_test": short_call_payoffs(K_test, states),
    }


def empirical_pairwise_L(X, y, p):
    """Necessary pairwise bound only; NOT a risk-field certificate."""
    X = _real_array(X, "X", 2)
    y = _real_array(y, "y", 1)
    p = _probabilities(p, X.shape[1])
    if y.shape != (len(X),) or len(X) == 0:
        raise ValueError("Require N>=1 payoff rows and N labels.")
    mean = X @ p
    W = (X - mean[:, None]) * np.sqrt(p)
    d = pdist(W)
    dy = pdist((y + mean)[:, None], metric="cityblock")
    if np.any((d == 0.0) & (dy != 0.0)):
        raise ValueError("Identical reduced payoffs have different reduced labels.")
    nonzero = d > 0.0
    return float(np.max(dy[nonzero] / d[nonzero])) if np.any(nonzero) else 0.0


def fixed_reference_L(data, alphas=(0.75,), inflation=1.0):
    """One L across ALL gamma and alpha values in the experiment.

    It dominates a theorem-backed ES baseline bound and every gamma=1 pairwise
    bound. This certifies compatibility at gamma=0; positive gamma is checked
    independently. Pairwise bounds alone are not risk-field certificates.
    """
    inflation = _scalar(inflation, "inflation", 1.0)
    alphas = _real_array(alphas, "alphas", 1)
    if not len(alphas):
        raise ValueError("At least one alpha is required.")
    bound = 0.0
    for alpha in alphas:
        y1 = quote_values(data["K_train"], data["X_train"], data["p"], 1.0, alpha)
        bound = max(
            bound,
            es_reduced_lipschitz_bound(data["p"], alpha),
            empirical_pairwise_L(data["X_train"], y1, data["p"]),
        )
    return inflation * bound + 1e-8


@dataclass
class Prediction:
    value: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    gap: np.ndarray
    statuses: tuple


class ConicReconstructor:
    """Numerical SOCP implementation of the exact reconstruction formula.

    Each query returns independently evaluated feasible primal/dual objectives.
    An unacceptable objective gap raises RuntimeError instead of hiding a
    failed solve. The solver object is reused, so instances are not thread-safe.
    """

    def __init__(self, X, y, p, L, gap_atol=2e-8, gap_rtol=2e-8):
        X = _real_array(X, "X", 2)
        y = _real_array(y, "y", 1)
        if X.shape[0] == 0 or X.shape[1] < 2 or y.shape != (len(X),):
            raise ValueError("Require X shape (N,B), N>=1, B>=2, and y shape (N,).")
        self.p = _probabilities(p, X.shape[1]).copy()
        self.L = _scalar(L, "L")
        self.gap_atol = _scalar(gap_atol, "gap_atol")
        self.gap_rtol = _scalar(gap_rtol, "gap_rtol")
        self.N, self.B = X.shape
        self.sqrt_p = np.sqrt(self.p)
        means = X @ self.p
        self.W = ((X - means[:, None]) * self.sqrt_p).T.copy()
        self.y_tilde = (y + means).copy()
        self._solver = None
        if self.L > 0.0:
            self._build_solver()

    def _build_solver(self):
        N, B = self.N, self.B
        # Solve min_{u,t} -a.u-t, W_n.u+t <= y_tilde_n, ||u|| <= L.
        # Its value is -phi(a). Nonnegative-constraint duals are simplex
        # weights, giving an independent feasible ORIGINAL primal objective.
        Aineq = sparse.hstack([sparse.csc_matrix(self.W.T), np.ones((N, 1))])
        Asoc = sparse.vstack(
            [
                sparse.csc_matrix((1, B + 1)),
                sparse.hstack([-sparse.eye(B), sparse.csc_matrix((B, 1))]),
            ]
        )
        A = sparse.vstack([Aineq, Asoc], format="csc")
        self._q = np.r_[np.zeros(B), -1.0]
        b = np.r_[self.y_tilde, self.L, np.zeros(B)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 250
        settings.tol_gap_abs = 1e-12
        settings.tol_gap_rel = 1e-12
        settings.tol_feas = 1e-12
        settings.presolve_enable = False
        self._solver = clarabel.DefaultSolver(
            sparse.csc_matrix((B + 1, B + 1)),
            self._q,
            A,
            b,
            [clarabel.NonnegativeConeT(N), clarabel.SecondOrderConeT(B + 1)],
            settings,
        )

    def _solve(self, a):
        if self.L == 0.0:
            value = float(np.min(self.y_tilde))
            return value, value, "AnalyticL0"
        self._q[: self.B] = -a
        self._solver.update(q=self._q)
        sol = self._solver.solve()
        status = str(sol.status)
        if status not in ("Solved", "AlmostSolved"):
            raise RuntimeError(f"Conic solve failed: {status}.")

        # Feasible primal candidate in the ORIGINAL (non-epigraph) formula.
        lam = np.maximum(np.asarray(sol.z[: self.N]), 0.0)
        mass = lam.sum()
        if not np.isfinite(mass) or mass <= 0:
            raise RuntimeError("Solver returned unusable simplex weights.")
        lam /= mass
        upper = float(self.y_tilde @ lam + self.L * np.linalg.norm(a - self.W @ lam))
        # Every simplex vertex is another valid upper candidate.
        vertices = self.y_tilde + self.L * np.linalg.norm(self.W - a[:, None], axis=0)
        upper = min(upper, float(vertices.min()))

        # The conic primal vector u yields a global supporting affine function.
        u = np.asarray(sol.x[: self.B], dtype=float)
        norm = np.linalg.norm(u)
        if norm > self.L:
            u *= self.L / norm
        # Tiny inward factor avoids radial rounding just beyond the ball.
        u *= 1.0 - 4.0 * np.finfo(float).eps
        intercept = float(np.min(self.y_tilde - u @ self.W))
        lower = float(a @ u + intercept)
        gap = upper - lower
        tolerance = self.gap_atol + self.gap_rtol * max(abs(upper), abs(lower))
        if not np.isfinite(gap) or gap < -tolerance or gap > tolerance:
            raise RuntimeError(
                f"Unacceptable objective bracket: lower={lower:.12g}, "
                f"upper={upper:.12g}, gap={gap:.3g}, tolerance={tolerance:.3g}."
            )
        # A reversed bracket within roundoff is retained, not silently certified.
        return lower, upper, status

    def predict(self, X):
        X = _real_array(X, "X_query", 2)
        if X.shape[1] != self.B:
            raise ValueError(f"Query payoffs must have {self.B} state coordinates.")
        mean = X @ self.p
        Wq = (X - mean[:, None]) * self.sqrt_p
        lower, upper, statuses = [], [], []
        for a, m in zip(Wq, mean):
            lo, hi, status = self._solve(a)
            lower.append(lo - m)
            upper.append(hi - m)
            statuses.append(status)
        lower, upper = np.asarray(lower), np.asarray(upper)
        return Prediction(upper.copy(), lower, upper, upper - lower, tuple(statuses))


class RandomDualReconstructor:
    """Finite supporting-plane approximation, with the zero direction.

    Sphere sampling matches the paper appendix. The legacy uniform-radius
    variant is available explicitly as sampling="radius". Neither is an exact
    solver or guaranteed to interpolate the training labels.
    """

    def __init__(self, X, y, p, L, K_dual=2000, seed=123, sampling="sphere"):
        X = _real_array(X, "X", 2)
        y = _real_array(y, "y", 1)
        if len(X) == 0 or y.shape != (len(X),):
            raise ValueError("Require N>=1 payoff rows and N labels.")
        self.p = _probabilities(p, X.shape[1]).copy()
        L = _scalar(L, "L")
        K_dual = _integer(K_dual, "K_dual", 0)
        seed = _integer(seed, "seed", 0)
        self.B = X.shape[1]
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(K_dual, self.B))
        if sampling not in ("sphere", "radius"):
            raise ValueError('sampling must be "sphere" or "radius".')
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        radius = np.full(K_dual, L) if sampling == "sphere" else L * rng.uniform(size=K_dual)
        U *= radius[:, None] * (1 - 4 * np.finfo(float).eps)
        self.U = np.vstack([np.zeros((1, self.B)), U])
        means = X @ self.p
        W = (X - means[:, None]) * np.sqrt(self.p)
        self.intercepts = np.min((y + means)[None, :] - self.U @ W.T, axis=1)

    def predict(self, X, chunk_size=1024):
        X = _real_array(X, "X_query", 2)
        chunk_size = _integer(chunk_size, "chunk_size", 1)
        if X.shape[1] != self.B:
            raise ValueError(f"Query payoffs must have {self.B} state coordinates.")
        out = np.empty(len(X), dtype=float)
        for start in range(0, len(X), chunk_size):
            batch = X[start : start + chunk_size]
            means = batch @ self.p
            W = (batch - means[:, None]) * np.sqrt(self.p)
            out[start : start + len(batch)] = np.max(W @ self.U.T + self.intercepts, axis=1) - means
        return out


def jensen_witness(K, y, states, gamma):
    """Largest within-gap endpoint-chord violation among sampled strikes.

    For this specific quadratic, extreme sampled endpoints maximize the
    same-gap violation at each interior strike. Zero is not a general
    certificate of compatibility; positive excess is a certificate of failure.
    """
    K, y = _real_array(K, "K", 1), _real_array(y, "y", 1)
    states = _real_array(states, "states", 1)
    gamma = _scalar(gamma, "gamma", 0.0, 1.0)
    short_call_payoffs(K, states)  # validate strike/state domains
    if K.shape != y.shape:
        raise ValueError("K and y must have the same shape.")
    order = np.argsort(K)
    K, y = K[order], y[order]
    breakpoints = np.unique(np.r_[0.0, states, 1.0])
    group = np.clip(np.searchsorted(breakpoints, K, side="right") - 1, 0, len(breakpoints) - 2)
    best, predicted, triples = 0.0, 0.0, 0
    max_identity_residual = 0.0
    for j in range(len(breakpoints) - 1):
        ids = np.flatnonzero(group == j)
        if len(ids) < 3:
            continue
        a, b = K[ids[0]], K[ids[-1]]
        if b <= a:
            continue
        inside = ids[(K[ids] > a) & (K[ids] < b)]
        t = (K[inside] - a) / (b - a)
        excess = y[inside] - ((1.0 - t) * y[ids[0]] + t * y[ids[-1]])
        exact = 0.5 * gamma * (K[inside] - a) * (b - K[inside])
        triples += len(inside)
        if len(inside):
            max_identity_residual = max(
                max_identity_residual, float(np.max(np.abs(excess - exact)))
            )
            idx = int(np.argmax(excess))
            if excess[idx] > best:
                best, predicted = float(excess[idx]), float(exact[idx])
    return dict(
        jensen_gap=best,
        jensen_gap_formula=predicted,
        jensen_identity_residual=max_identity_residual,
        witness_count=triples,
    )


def run_general_experiment(
    N_train=100,
    N_test=10,
    B=10,
    gamma=0.5,
    inflation=1.0,
    seed=123,
    K_dual=2000,
    compare_random_dual=True,
    L: Optional[float] = None,
    alpha=0.75,
):
    """Run one paired draw; return aggregate rows and every training prediction."""
    data = sample_call_data(N_train, N_test, B, seed)
    gamma = _scalar(gamma, "gamma", 0.0, 1.0)
    if L is None:
        L = fixed_reference_L(data, [alpha], inflation)
    L = _scalar(L, "L")
    p, X, Xt = data["p"], data["X_train"], data["X_test"]
    y = quote_values(data["K_train"], X, p, gamma, alpha)
    yt = quote_values(data["K_test"], Xt, p, gamma, alpha)
    witness = jensen_witness(data["K_train"], y, data["states"], gamma)
    start = time.perf_counter()
    model = ConicReconstructor(X, y, p, L)
    pred = model.predict(np.vstack([X, Xt]))
    reference_seconds = time.perf_counter() - start
    ref_train, ref_test = pred.value[:N_train], pred.value[N_train:]
    residual_lower = np.maximum(y - pred.upper[:N_train], 0.0)
    residual_upper = np.maximum(y - pred.lower[:N_train], 0.0)
    common = dict(
        B=B,
        N_train=N_train,
        N_test=N_test,
        gamma=gamma,
        alpha=alpha,
        seed=seed,
        L_recon=L,
        inflation=inflation,
        **witness,
        risk_field_max_lower=float(residual_lower.max()),
        risk_field_max_upper=float(residual_upper.max()),
        reference_train_MSE_upper=float(np.mean(residual_upper**2)),
        max_solver_gap=float(np.max(pred.gap)),
        min_solver_gap=float(np.min(pred.gap)),
    )
    rows = []

    def metrics(method, train, test, elapsed):
        return dict(
            common,
            method=method,
            K_dual=(K_dual if method == "random_dual" else 0),
            train_MSE=float(np.mean((train - y) ** 2)),
            test_MSE=float(np.mean((test - yt) ** 2)),
            train_MAE=float(np.mean(np.abs(train - y))),
            test_MAE=float(np.mean(np.abs(test - yt))),
            train_MaxAE=float(np.max(np.abs(train - y))),
            test_MaxAE=float(np.max(np.abs(test - yt))),
            seconds=elapsed,
        )

    rows.append(metrics("conic_reference", ref_train, ref_test, reference_seconds))
    points = pd.DataFrame(
        dict(
            seed=seed,
            B=B,
            N_train=N_train,
            gamma=gamma,
            alpha=alpha,
            K=data["K_train"],
            y=y,
            reconstruction=ref_train,
            reconstruction_lower=pred.lower[:N_train],
            solver_gap=pred.gap[:N_train],
        )
    )
    if compare_random_dual:
        start = time.perf_counter()
        approx = RandomDualReconstructor(X, y, p, L, K_dual, seed)
        ap = approx.predict(np.vstack([X, Xt]))
        rows.append(metrics("random_dual", ap[:N_train], ap[N_train:], time.perf_counter() - start))
        points["random_dual"] = ap[:N_train]
    return pd.DataFrame(rows), points


def summarize(raw, x_col):
    metrics = [
        "train_MSE",
        "test_MSE",
        "train_MAE",
        "test_MAE",
        "train_MaxAE",
        "test_MaxAE",
        "L_recon",
        "jensen_gap",
        "risk_field_max_lower",
        "risk_field_max_upper",
        "max_solver_gap",
        "seconds",
    ]
    grouped = raw.groupby([x_col, "method"], sort=True)
    result = grouped[metrics].agg(["mean", "std"]).reset_index()
    result.columns = [a if not b else f"{a}_{b}" for a, b in result.columns]
    for col in result:
        if col.endswith("_std"):
            result[col] = result[col].fillna(0.0)
    result["n_draws"] = grouped.size().to_numpy()
    return result
