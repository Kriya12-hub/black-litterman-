import pandas as pd
import numpy as np

def implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion=2.5):
    """
    Compute market-implied equilibrium returns.
    """
    pi = risk_aversion * cov_matrix.dot(market_weights)
    return pi
