import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

from risk_measure_reconstruction import (
    FiniteModel,
    FiniteGeometry,
    GBM,
    Heston,
    CallGeometry,
    PopulationCallReconstructor,
    common_lipschitz_budget,
    population_call_risk,
)
from risk_measure_reconstruction.gbm_risk import spectrum, reduced_lipschitz_bound

FAMILIES = [("es", 0.73), ("power", 2.1), ("exponential", 3.2), ("wang", 0.8)]


@pytest.mark.parametrize("family,parameter", FAMILIES)
def test_nonuniform_atoms_against_integrated_spectrum(family, parameter):
    model = FiniteModel([0.0, 0.8, 1.9, 3.0], [0.12, 0.34, 0.28, 0.26])
    edges = np.r_[0, np.cumsum(model.probabilities)]
    mass = []
    for a, b in zip(edges[:-1], edges[1:]):
        if family == "es":
            mass.append(max(0, b - max(a, parameter)) / (1 - parameter))
        else:
            mass.append(
                quad(lambda u: float(spectrum(family, parameter, np.array(u))), a, b, epsabs=1e-11)[
                    0
                ]
            )
    k = np.array([0.4, 1.0, 2.5])
    expected = -model.payoffs(k) @ mass
    np.testing.assert_allclose(
        population_call_risk(model, k, family, parameter).value, expected, atol=2e-10
    )


@pytest.mark.parametrize("family,parameter", FAMILIES)
def test_finite_paper_cash_translation(family, parameter):
    model = FiniteModel()
    strikes = np.linspace(0.1, 0.9, 20)
    labels = population_call_risk(model, strikes, family, parameter).value
    fit = PopulationCallReconstructor(
        FiniteGeometry(model, strikes), labels, common_lipschitz_budget()
    )
    training = fit.certify_training()
    query = np.array([0.0, 0.03, 0.43, 0.97, 1.0])
    result = fit.predict(query)
    truth = population_call_risk(model, query, family, parameter).value
    np.testing.assert_allclose(result.value, truth, rtol=0, atol=3e-13)
    np.testing.assert_allclose(result.value + query, labels[0] + strikes[0], rtol=0, atol=3e-13)
    assert training.gap.max() < 3e-13


def test_gbm_moments_geometry_and_risk_limits():
    model = GBM(sigma=0.2)
    assert model.truncated_moment(1, 0) == pytest.approx(model.mean, rel=1e-14)
    k = np.array([0.7, 1.0, 1.3])
    expected = np.array([model.call_mean(x) for x in k])
    for family, parameter in [("es", 0.0), ("power", 1.0), ("wang", 0.0)]:
        np.testing.assert_allclose(
            population_call_risk(model, k, family, parameter).value, expected, atol=1e-12
        )
    geometry = CallGeometry(model, k)
    a, m = geometry.coordinates([0.85])
    assert (a @ a.T)[0, 0] == pytest.approx(model.call_covariance(0.85, 0.85), rel=1e-12)
    assert m[0] == pytest.approx(-model.call_mean(0.85))
    np.testing.assert_allclose(
        a @ geometry.train_coordinates.T, [[model.call_covariance(0.85, x) for x in k]], atol=1e-13
    )


@pytest.mark.parametrize("family,parameter", FAMILIES)
def test_gbm_risk_against_independent_normal_density_integral(family, parameter):
    model = GBM(sigma=0.2)
    k = 0.97

    def integrand(z):
        u = ndtr(z)
        if not 0 < u < 1:
            return 0.0
        loss = max(np.exp(model.eta + model.v * z) - k, 0)
        return (
            loss * float(spectrum(family, parameter, u)) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        )

    points = [(np.log(k) - model.eta) / model.v]
    if family == "es":
        points.append(ndtri(parameter))
    reference = quad(integrand, -8, 8, points=points, epsabs=2e-10, limit=150)[0]
    assert population_call_risk(model, [k], family, parameter).value[0] == pytest.approx(
        reference, abs=2e-8
    )


@pytest.mark.slow
def test_heston_zero_vol_of_vol_matches_gbm():
    heston = Heston(xi=0, v0=0.04, theta=0.04)
    gbm = GBM(sigma=0.2)
    assert heston.moment(1) == pytest.approx(gbm.mean, rel=1e-13)
    strikes = np.array([0.8, 1.0, 1.2])
    for order in range(3):
        expected = [gbm.truncated_moment(order, k) for k in strikes]
        np.testing.assert_allclose(
            heston.truncated_moments(np.log(strikes), order), expected, atol=2e-10
        )
    for family, parameter in FAMILIES:
        np.testing.assert_allclose(
            population_call_risk(heston, strikes, family, parameter).value,
            population_call_risk(gbm, strikes, family, parameter).value,
            atol=2e-9,
        )


def test_heston_domain_guards_and_common_budget():
    with pytest.raises(ValueError, match="Feller"):
        Heston(xi=1.0)
    assert common_lipschitz_budget() == pytest.approx(np.sqrt(np.expm1(4)), rel=2e-12)
    assert reduced_lipschitz_bound("power", 1) == 0
    assert reduced_lipschitz_bound("wang", 0) == 0
