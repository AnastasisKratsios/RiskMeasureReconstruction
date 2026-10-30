"""Nontrivial finite calls crossing the stock support: python examples/finite_state.py."""

import numpy as np
from risk_measure_reconstruction import (
    FiniteModel,
    FiniteGeometry,
    PopulationCallReconstructor,
    common_lipschitz_budget,
    population_call_risk,
)

# A separate example, distinct from the paper's all-in-the-money finite case.
stock = FiniteModel(states=np.linspace(0.0, 2.0, 25))
training_strikes = np.linspace(0.1, 1.9, 20)
labels = population_call_risk(stock, training_strikes, "es", 0.75).value
geometry = FiniteGeometry(stock, training_strikes)
fit = PopulationCallReconstructor(geometry, labels, common_lipschitz_budget())
fit.certify_training()

query = np.array([0.05, 0.55, 1.05, 1.95])
prediction = fit.predict(query)
truth = population_call_risk(stock, query, "es", 0.75).value
for strike, target, value, gap in zip(query, truth, prediction.value, prediction.gap):
    print(f"K={strike:.2f}  latent={target:.8f}  reconstructed={value:.8f}  gap={gap:.2e}")
