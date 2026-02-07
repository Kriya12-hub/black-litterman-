import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AlphaStack",
    layout="wide"
)

# -------------------------------------------------
# LOGO + TITLE + MOVING BANNER (BRANDING ONLY)
# -------------------------------------------------
st.markdown("""
<style>
.alphastack-header {
    padding: 10px 0 22px 0;
}

/* Logo row */
.logo-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Logo icon */
.logo-icon {
    font-size: 42px;
}

/* Logo text */
.logo-text {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

/* Subtitle */
.logo-sub {
    font-size: 14px;
    opacity: 0.8;
    margin-left: 52px;
}

/* Moving banner */
.marquee {
    overflow: hidden;
    white-space: nowrap;
    margin-top: 12px;
}

.marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 14s linear infinite;
    font-size: 13px;
    opacity: 0.75;
}

@keyframes marquee {
    0% { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
</style>

<div class="alphastack-header">
    <div class="logo-row">
        <div class="logo-icon">📊</div>
        <div class="logo-text">AlphaStack</div>
    </div>

    <div class="logo-sub">
        Black–Litterman Portfolio Optimizer
    </div>

    <div class="marquee">
        <span>
            Market Equilibrium × Investor Views × Confidence-Weighted Allocation × Scenario-Based Portfolio Intelligence
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.write(
    "This application compares traditional Mean–Variance optimization with the "
    "Black–Litterman model, which blends market equilibrium with investor views."
)

# -------------------------------------------------
# SIDEBAR – INVESTOR VIEW
# -------------------------------------------------
st.sidebar.header("Investor View")

tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]

asset_long = st.sidebar.selectbox(
    "Asset expected to outperform",
    tickers,
    index=0
)

asset_short = st.sidebar.selectbox(
    "Asset expected to underperform",
    tickers,
    index=2
)

view_return = st.sidebar.slider(
    "Expected Outperformance (%)",
    0.0, 25.0, 8.0, step=1.0
) / 100

confidence = st.sidebar.slider(
    "Confidence Level",
    1, 100, 90
)

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
@st.cache_data
def load_data(tickers):
    data = yf.download(tickers, start="2019-01-01", progress=False)
    prices = data["Adj Close"] if "Adj Close" in data else data["Close"]
    returns = prices.pct_change().dropna()
    return returns

returns = load_data(tickers)
cov_matrix = returns.cov() * 252

# -------------------------------------------------
# MEAN–VARIANCE OPTIMIZATION
# -------------------------------------------------
def mean_variance_optimizer(expected_returns, cov_matrix):
    n = len(expected_returns)

    def neg_sharpe(weights):
        ret = weights @ expected_returns
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -ret / vol

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n
    init = np.ones(n) / n

    result = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=expected_returns.index)

historical_returns = returns.mean() * 252
weights_mv = mean_variance_optimizer(historical_returns, cov_matrix)

# -------------------------------------------------
# BLACK–LITTERMAN MODEL
# -------------------------------------------------
def black_litterman_posterior(cov_matrix, pi, P, Q, omega, tau=0.05):
    cov = cov_matrix.values
    pi = np.asarray(pi).reshape(-1, 1)
    middle = np.linalg.inv(P @ (tau * cov) @ P.T + omega)
    posterior = pi + tau * cov @ P.T @ middle @ (Q - P @ pi)
    return pd.Series(posterior.flatten(), index=cov_matrix.index)

n = len(tickers)
market_weights = pd.Series([1/n] * n, index=tickers)
pi = cov_matrix @ market_weights

P = np.zeros((1, n))
P[0, tickers.index(asset_long)] = 1
P[0, tickers.index(asset_short)] = -1

Q = np.array([[view_return]])
omega = np.array([[(1 - confidence/100) * 0.05]])

posterior_returns = black_litterman_posterior(cov_matrix, pi, P, Q, omega)
weights_bl = mean_variance_optimizer(posterior_returns, cov_matrix)

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
st.subheader("Posterior Returns")
st.dataframe(posterior_returns.to_frame("Posterior Return"))

st.subheader("Portfolio Weights Comparison")

comparison = pd.DataFrame({
    "Mean–Variance": weights_mv,
    "Black–Litterman": weights_bl
})

st.bar_chart(comparison)
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

st.caption(
    "AlphaStack is a decision-support system for portfolio scenario analysis. "
    "Not investment advice."
)
