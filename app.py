import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AlphaStack",
    layout="wide",
)

# =========================
# CUSTOM CSS (LOGO + MARQUEE)
# =========================
st.markdown("""
<style>
.header-box {
    padding: 1rem 0 0.5rem 0;
}

.logo-title {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 38px;
    font-weight: 800;
}

.logo-sub {
    font-size: 16px;
    color: #555;
    margin-left: 52px;
}

.marquee {
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    box-sizing: border-box;
    margin-top: 10px;
    border-radius: 8px;
    background: #f7f7f7;
    padding: 10px 0;
}

.marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 18s linear infinite;
    font-weight: 600;
    color: #333;
}

@keyframes marquee {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-box">
    <div class="logo-title">📊 AlphaStack</div>
    <div class="logo-sub">Black–Litterman Portfolio Optimizer</div>
    <div class="marquee">
        <span>
        Market Equilibrium × Investor Views × Confidence-Weighted Allocation × Scenario-Based Portfolio Intelligence
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "This application compares traditional Mean–Variance optimization with the Black–Litterman model, "
    "which blends market equilibrium with investor views."
)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]

    raw = yf.download(
        tickers,
        start="2022-01-01",
        end="2024-01-01",
        progress=False,
        auto_adjust=True
    )

    # ✅ Handle both single & multi-index safely
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices.dropna(how="all")
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252

    return returns, cov_matrix


# =========================
# SIDEBAR – INVESTOR VIEW
# =========================
st.sidebar.title("Investor View")

asset_long = st.sidebar.selectbox(
    "Asset expected to outperform",
  tickers

)

asset_short = st.sidebar.selectbox(
    "Asset expected to underperform",
    assets,
    index=2
)

view_return = st.sidebar.slider(
    "Expected Outperformance (%)",
    min_value=0.1,
    max_value=15.0,
    value=5.0
) / 100

confidence = st.sidebar.slider(
    "Confidence Level",
    min_value=1,
    max_value=100,
    value=50
) / 100

# =========================
# BLACK–LITTERMAN CORE
# =========================
tau = 0.05
market_weights = np.ones(n) / n

pi = tau * cov_matrix.values @ market_weights

P = np.zeros((1, n))
P[0, assets.index(asset_long)] = 1
P[0, assets.index(asset_short)] = -1

Q = np.array([view_return])

omega = np.array([[((1 - confidence) + 0.001)]])

middle = np.linalg.inv(P @ (tau * cov_matrix.values) @ P.T + omega)
mu_bl = pi + (tau * cov_matrix.values) @ P.T @ middle @ (Q - P @ pi)

posterior_returns = pd.Series(mu_bl.flatten(), index=assets)

# =========================
# OPTIMIZATION
# =========================
inv_cov = np.linalg.inv(cov_matrix.values)

weights_mv = inv_cov @ returns.mean().values
weights_mv = np.maximum(weights_mv, 0)
weights_mv /= weights_mv.sum()

weights_bl = inv_cov @ posterior_returns.values
weights_bl = np.maximum(weights_bl, 0)
weights_bl /= weights_bl.sum()

weights_df = pd.DataFrame({
    "Mean–Variance": weights_mv,
    "Black–Litterman": weights_bl
}, index=assets)

# =========================
# OUTPUT
# =========================
st.subheader("Posterior Returns")
st.dataframe(posterior_returns.to_frame("Posterior Return"))

st.subheader("Portfolio Weights Comparison")

fig, ax = plt.subplots()
weights_df.plot(kind="bar", ax=ax)
ax.set_ylabel("Weight")
ax.set_xlabel("Asset")
ax.legend()
st.pyplot(fig)

st.dataframe(weights_df)
