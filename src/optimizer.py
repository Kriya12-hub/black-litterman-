import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mean_variance_optimizer(expected_returns, cov_matrix, risk_free_rate=0.0):
    n = len(expected_returns)

    def negative_sharpe(weights):
        port_return = weights @ expected_returns
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -(port_return - risk_free_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    init = np.ones(n) / n

    result = minimize(
        negative_sharpe,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return pd.Series(result.x, index=expected_returns.index)
