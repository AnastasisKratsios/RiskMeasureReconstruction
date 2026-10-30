import numpy as np
import pytest
from scipy.optimize import linprog, minimize_scalar

from risk_measure_reconstruction import (
    ConicReconstructor,
    RandomDualReconstructor,
    weighted_es_loss,
)
from risk_measure_reconstruction.risk_reconstruction import (
    jensen_witness,
    quote_values,
    short_call_payoffs,
)


def test_es_fractional_atoms_matches_independent_dual_linear_program():
    losses = np.array([-2.0, 1.0, 4.0, 8.0])
    p = np.array([0.17, 0.31, 0.39, 0.13])
    for alpha in (0.0, 0.55, 0.89):
        result = linprog(
            -losses,
            A_eq=np.ones((1, 4)),
            b_eq=[1],
            bounds=list(zip(np.zeros(4), p / (1 - alpha))),
            method="highs",
        )
        assert result.success
        assert weighted_es_loss(losses, p, alpha) == pytest.approx(-result.fun, abs=2e-14)


def test_socp_agrees_with_independent_two_quote_primal():
    p = np.array([0.2, 0.3, 0.5])
    x = np.array([[-1.0, 0.3, 2.0], [1.3, -0.5, 0.1]])
    y, L = np.array([0.7, 1.1]), 1.8
    query = np.array([[0.1, 1.4, -0.4]])
    means = x @ p
    center = (x - means[:, None]) * np.sqrt(p)
    qm = (query @ p)[0]
    a = (query[0] - qm) * np.sqrt(p)

    def objective(t):
        return (
            (y + means) @ np.array([t, 1 - t])
            + L * np.linalg.norm(a - t * center[0] - (1 - t) * center[1])
            - qm
        )

    optimum = min(
        objective(0),
        objective(1),
        minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-14}).fun,
    )
    result = ConicReconstructor(x, y, p, L).predict(query)
    assert result.lower[0] - 1e-10 <= optimum <= result.upper[0] + 1e-10
    assert result.value[0] == pytest.approx(optimum, abs=2e-8)


def test_cash_additivity_convexity_and_l0():
    p = np.full(3, 1 / 3)
    x = np.array([[-1.0, 0.0, 1.0], [-2.0, -1.0, 2.0], [1.0, 0.0, 2.0]])
    y = weighted_es_loss(-x, p, 0.5)
    fit = ConicReconstructor(x, y, p, 2.0)
    a, b = np.array([0.3, -0.7, 1.0]), np.array([-0.8, 0.4, 0.9])
    query = np.stack([a, a + 2.4, b, 0.3 * a + 0.7 * b])
    result = fit.predict(query).value
    assert result[1] == pytest.approx(result[0] - 2.4, abs=5e-8)
    assert result[3] <= 0.3 * result[0] + 0.7 * result[2] + 5e-8
    zero = ConicReconstructor(x, y, p, 0).predict(query)
    np.testing.assert_allclose(zero.value, np.min(y + x @ p) - query @ p, atol=1e-15)
    assert np.all(zero.gap == 0)


def test_random_dual_is_lower_approximation():
    x = np.array([[-1.0, 0.0, 1.0], [-2.0, -1.0, 2.0]])
    p = np.full(3, 1 / 3)
    y = weighted_es_loss(-x, p, 0.5)
    query = np.array([[0.2, -0.3, 0.7], [-0.8, 0.1, 1.3]])
    conic = ConicReconstructor(x, y, p, 2).predict(query)
    random = RandomDualReconstructor(x, y, p, 2, K_dual=100, seed=12)
    assert np.all(np.linalg.norm(random.U, axis=1) <= 2)
    assert np.all(random.predict(query) <= conic.upper + 1e-12)
    np.testing.assert_allclose(random.predict(query + 1.7), random.predict(query) - 1.7, atol=1e-14)


def test_actual_jensen_witness_on_affine_payoff_segment():
    states = np.arange(10) / 10
    k = np.array([0.21, 0.24, 0.29])
    p = np.full(10, 0.1)
    x = short_call_payoffs(k, states)
    y = quote_values(k, x, p, gamma=0.6)
    witness = jensen_witness(k, y, states, gamma=0.6)
    assert witness["jensen_gap"] > 0
    assert witness["jensen_identity_residual"] < 1e-14
    t = (k[1] - k[0]) / (k[2] - k[0])
    np.testing.assert_allclose(x[1], (1 - t) * x[0] + t * x[2], atol=1e-15)


@pytest.mark.parametrize("p,alpha", [([0.2, 0.2], 0.5), ([0.0, 1.0], 0.5), ([0.5, 0.5], 1.0)])
def test_invalid_es_inputs_are_rejected(p, alpha):
    with pytest.raises(ValueError):
        weighted_es_loss([1, 2], p, alpha)
