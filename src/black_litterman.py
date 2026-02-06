import numpy as np
import pandas as pd

def black_litterman_posterior(cov_matrix, pi, P, Q, omega, tau=0.05):
    """
    Compute Black–Litterman posterior expected returns.
    """
    cov = cov_matrix.values
    pi = pi.values.reshape(-1, 1)

    middle = np.linalg.inv(P @ (tau * cov) @ P.T + omega)
    adj = tau * cov @ P.T @ middle @ (Q - P @ pi)

    posterior_returns = pi + adj
    return pd.Series(posterior_returns.flatten(), index=cov_matrix.index)
