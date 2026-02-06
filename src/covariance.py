import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()

def shrinkage_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf()
    lw.fit(returns.values)
    cov = pd.DataFrame(
        lw.covariance_,
        index=returns.columns,
        columns=returns.columns
    )
    return cov
