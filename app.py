import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

from src.data_loader import load_price_data, compute_returns
from src.covariance import shrinkage_covariance
from src.optimizer import mean_variance_optimizer
from src.equilibrium import implied_equilibrium_returns
from src.black_litterman import black_litterman_posterior

st.set_page_config(page_title="Black–Litterman Portfolio Optimizer", layout="wide")

st.title("Black–Litterman Portfolio Optimizer")
st.write(
    """
    This application compares traditional Mean–Variance optimization
    with the Black–Litterman model, which blends market equilibrium
    with investor views.
    """
)

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
start_date = "2019-01-01"
end_date = "2024-12-31"

@st.cache_data
def load_data():
    prices = load_price_data(tickers, start_date, end_date)
    returns = compute_returns(prices)
    cov_matrix = shrinkage_covariance(returns)
    return returns, cov_matrix

returns, cov_matrix = load_data()

historical_returns = returns.mean() * 252
weights_mv = mean_variance_optimizer(historical_returns, cov_matrix)

weights_mv = mean_variance_optimizer(historical_returns, cov_matrix)

st.sidebar.header("Investor View")

asset_long = st.sidebar.selectbox("Asset expected to outperform", tickers, index=1)
asset_short = st.sidebar.selectbox("Asset expected to underperform", tickers, index=2)

view_return = st.sidebar.slider(
    "Expected Outperformance (%)",
    min_value=0.0,
    max_value=25.0,
    value=10.0,
    step=1.0
) / 100


confidence = st.sidebar.slider(
    "Confidence Level",
    min_value=1,
    max_value=100,
    value=70
)

n = len(tickers)
market_weights = pd.Series([1/n] * n, index=tickers)

pi = implied_equilibrium_returns(cov_matrix, market_weights)

P = [[0]*n]
P[0][tickers.index(asset_long)] = 1
P[0][tickers.index(asset_short)] = -1

Q = [[view_return]]
confidence_scaled = confidence / 100  # 0.01 → 1.0
omega = [[(1 - confidence_scaled) * 0.05]]


posterior_returns = black_litterman_posterior(
    cov_matrix,
    pi,
    P,
    Q,
    omega
)

weights_bl = mean_variance_optimizer(posterior_returns, cov_matrix)
st.write("Posterior returns:", posterior_returns)


st.subheader("Portfolio Weights Comparison")

comparison = pd.DataFrame({
    "Mean–Variance": weights_mv,
    "Black–Litterman": weights_bl
})

st.bar_chart(comparison)
st.dataframe(comparison)

returns, cov_matrix = load_data()
