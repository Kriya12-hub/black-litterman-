import numpy as np
import pandas as pd

def black_litterman_posterior(cov_matrix, pi, P, Q, omega, tau=0.05):
    """
    Compute Black–Litterman posterior expected returns.
    Works safely with pandas inputs.
    """
    cov = cov_matrix.values
    pi = np.asarray(pi).reshape(-1, 1)
    P = np.asarray(P)
    Q = np.asarray(Q)
    omega = np.asarray(omega)

    middle = np.linalg.inv(P @ (tau * cov) @ P.T + omega)
    adjustment = tau * cov @ P.T @ middle @ (Q - P @ pi)

    posterior = pi + adjustment

    return pd.Series(posterior.ravel(), index=cov_matrix.index)
