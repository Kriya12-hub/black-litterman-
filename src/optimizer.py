import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mean_variance_optimizer(expected_returns, cov_matrix):
    n = len(expected_returns)

    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    )

    bounds = tuple((0, 1) for _ in range(n))
    initial_weights = np.ones(n) / n

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    weights = pd.Series(result.x, index=expected_returns.index)
    return weights
