"""Configurable finite-state spectral call risks, preserving exact atom masses."""

import math
import numpy as np
from scipy.special import ndtr, ndtri
from .risk_reconstruction import _real_array, _probabilities, weighted_es_loss
from .gbm_risk import RiskValues, validate_parameter


class FiniteModel:
    def __init__(self, states=None, probabilities=None):
        self.states = _real_array(
            np.linspace(1, 10, 1000) if states is None else states, "states", 1
        ).copy()
        if len(self.states) < 2 or self.states[0] < 0 or np.any(np.diff(self.states) <= 0):
            raise ValueError("Require B>=2 strictly increasing nonnegative states.")
        p = (
            np.full(len(self.states), 1 / len(self.states))
            if probabilities is None
            else probabilities
        )
        self.probabilities = _probabilities(p, len(self.states)).copy()

    @property
    def mean(self):
        return float(self.states @ self.probabilities)

    @property
    def std(self):
        return math.sqrt(float((self.states - self.mean) ** 2 @ self.probabilities))

    def payoffs(self, strikes):
        k = _real_array(strikes, "strikes", 1)
        if np.any(k < 0):
            raise ValueError("Strikes must be nonnegative.")
        return -np.maximum(self.states[None, :] - k[:, None], 0.0)

    def spectral_call_risk(self, strikes, family, parameter):
        p = validate_parameter(family, parameter)
        losses = -self.payoffs(strikes)
        if family == "es":
            values = weighted_es_loss(losses, self.probabilities, p)
        else:
            edges = np.r_[0.0, np.cumsum(self.probabilities)]
            edges[-1] = 1.0
            if family == "power":
                cdf = edges**p
            elif family == "exponential":
                cdf = np.exp(p * (edges - 1)) * (-np.expm1(-p * edges)) / (-np.expm1(-p))
            else:
                cdf = ndtr(ndtri(edges) - p)
            weights = np.diff(cdf)
            if np.any(weights < 0) or abs(weights.sum() - 1) > 2e-14:
                raise RuntimeError("Invalid integrated spectral weights.")
            values = losses @ weights
        return RiskValues(values, np.zeros(len(strikes)))


class FiniteGeometry:
    def __init__(self, model, train_strikes):
        self.gbm = model
        self.strikes = _real_array(train_strikes, "train_strikes", 1).copy()
        if len(self.strikes) < 1 or np.any(np.diff(self.strikes) <= 0):
            raise ValueError("Training strikes must be strictly increasing.")
        self.train_coordinates, self.train_means = self.coordinates(self.strikes)
        self.gram = self.train_coordinates @ self.train_coordinates.T
        self.factor_residual = 0.0
        self.condition_number = float("inf")
        self.negative_residual_corrections = []

    def coordinates(self, strikes):
        x = self.gbm.payoffs(strikes)
        mean = x @ self.gbm.probabilities
        return (x - mean[:, None]) * np.sqrt(self.gbm.probabilities), mean
