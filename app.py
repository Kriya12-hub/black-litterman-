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
# ALPHASTACK HEADER + MOVING BANNER (ONLY UI ADDITION)
# -------------------------------------------------
st.markdown("""
<style>
.alphastack-header {
    padding: 8px 0 18px 0;
}

.alphastack-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

.alphastack-sub {
    font-size: 14px;
    opacity: 0.8;
    margin-top: 4px;
}

.marquee {
    overflow: hidden;
    white-space: nowrap;
    margin-top: 10px;
}

.marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 16s linear infinite;
    font-size: 13px;
    opacity: 0.75;
}

@keyframes marquee {
    0% { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
</style>

<div class="alphastack-header">
    <div class="alphastack-title">AlphaStack</div>
    <div class="alphastack-sub">
        Black–Litterman Portfolio Optimizer
    </div>

    <div class="marquee">
        <span>
            Market Equilibrium × Investor Views × Confidence-Weighted Allocation × Scenario Analysis Platform
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
# DATA LOADING (CACHED)
# -------------------------------------------------
@st.cache_data
def load_data(tickers):
    data = yf.download(tickers, start="2019-01-01", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
    else:
        prices = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]

    returns = prices.pct_change().dropna()
    return returns

returns = load_data(tickers)
cov_matrix = returns.cov() * 252

# -------------------------------------------------
# MEAN–VARIANCE (MAX SHARPE)
# -------------------------------------------------
def mean_variance_optimizer(expected_returns, cov_matrix):
    n = len(expected_returns)

    def neg_sharpe(weights):
        port_return = weights @ expected_returns
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -port_return / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    init = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return pd.Series(result.x, index=expected_returns.index)

historical_returns = returns.mean() * 252
weights_mv = mean_variance_optimizer(historical_returns, cov_matrix)

# -------------------------------------------------
# BLACK–LITTERMAN
# -------------------------------------------------
def black_litterman_posterior(cov_matrix, pi, P, Q, omega, tau=0.05):
    cov = cov_matrix.values
    pi = np.asarray(pi).reshape(-1, 1)
    P = np.asarray(P)
    Q = np.asarray(Q)
    omega = np.asarray(omega)

    middle = np.linalg.inv(P @ (tau * cov) @ P.T + omega)
    adjustment = tau * cov @ P.T @ middle @ (Q - P @ pi)
    posterior = pi + adjustment

    return pd.Series(posterior.ravel(), index=cov_matrix.index)

n = len(tickers)
market_weights = pd.Series([1/n] * n, index=tickers)
pi = cov_matrix @ market_weights

P = np.zeros((1, n))
P[0, tickers.index(asset_long)] = 1
P[0, tickers.index(asset_short)] = -1

Q = np.array([[view_return]])

confidence_scaled = confidence / 100
omega = np.array([[(1 - confidence_scaled) * 0.05]])

posterior_returns = black_litterman_posterior(
    cov_matrix,
    pi,
    P,
    Q,
    omega
)

weights_bl = mean_variance_optimizer(posterior_returns, cov_matrix)

# -------------------------------------------------
# OUTPUTS
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

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption(
    "AlphaStack is a decision-support tool for portfolio scenario analysis. "
    "It does not provide investment advice."
)
