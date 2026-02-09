import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AlphaStack",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# TITLE + BRANDING
# --------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:38px;">📊</span>
        <h1 style="margin:0;">AlphaStack</h1>
    </div>
    <p style="margin-top:-5px; color:gray;">
        Black–Litterman Portfolio Optimizer
    </p>
    <marquee behavior="scroll" direction="left">
        Market Equilibrium × Investor Views × Confidence-Weighted Allocation × Scenario-Based Portfolio Intelligence
    </marquee>
    """,
    unsafe_allow_html=True
)

st.write(
    "This application compares traditional Mean–Variance optimization with the Black–Litterman model, "
    "which blends market equilibrium with investor views."
)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    assets = tickers
    tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]
    

    data = yf.download(
        tickers,
        start="2022-01-01",
        end="2024-01-01",
        progress=False,
        group_by="ticker"
    )

    # Safely extract Adjusted Close
    prices = pd.DataFrame({
        ticker: data[ticker]["Adj Close"]
        for ticker in tickers
        if "Adj Close" in data[ticker]
    })

    prices = prices.dropna(how="all")

    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252  # annualized

    return tickers, returns, cov_matrix

# --------------------------------------------------
# SIDEBAR – INVESTOR VIEW
# --------------------------------------------------
st.sidebar.title("Investor View")

asset_long = st.sidebar.selectbox(
    "Asset expected to outperform",
    assets,
    index=0
)

asset_short = st.sidebar.selectbox(
    "Asset expected to underperform",
    assets,
    index=2
)

expected_outperformance = (
    st.sidebar.slider(
        "Expected Outperformance (%)",
        min_value=1.0,
        max_value=15.0,
        value=8.0,
        step=0.5
    ) / 100
)

confidence = st.sidebar.slider(
    "Confidence Level",
    min_value=10,
    max_value=90,
    value=70,
    step=5
) / 100

# --------------------------------------------------
# BLACK–LITTERMAN CALCULATION
# --------------------------------------------------
# Market equilibrium weights (equal-weighted)
market_weights = np.ones(n) / n

# Risk aversion & tau
delta = 2.5
tau = 0.05

# Implied equilibrium returns
pi = delta * cov_matrix.values @ market_weights

# Investor view matrix (1 view)
P = np.zeros((1, n))
P[0, assets.index(asset_long)] = 1
P[0, assets.index(asset_short)] = -1

Q = np.array([[expected_outperformance]])

# Confidence-adjusted uncertainty
Omega = np.array([[((1 - confidence) + 1e-6)]])

# Posterior returns (Black–Litterman)
middle = np.linalg.inv(
    np.linalg.inv(tau * cov_matrix.values) + P.T @ np.linalg.inv(Omega) @ P
)

mu_bl = middle @ (
    np.linalg.inv(tau * cov_matrix.values) @ pi +
    P.T @ np.linalg.inv(Omega) @ Q
)

posterior_returns = pd.Series(mu_bl.flatten(), index=assets)

# --------------------------------------------------
# PORTFOLIO WEIGHTS
# --------------------------------------------------
# Mean-Variance weights (simple normalized positive returns)
mv_raw = np.maximum(returns.mean().values, 0)
mv_weights = mv_raw / mv_raw.sum()

# Black–Litterman weights (normalized positive posterior returns)
bl_raw = np.maximum(posterior_returns.values, 0)
bl_weights = bl_raw / bl_raw.sum()

weights_df = pd.DataFrame({
    "Mean-Variance": mv_weights,
    "Black-Litterman": bl_weights
}, index=assets)

# --------------------------------------------------
# OUTPUT TABLES
# --------------------------------------------------
st.subheader("Posterior Returns")
st.dataframe(posterior_returns.to_frame("Posterior Return"), use_container_width=True)

st.subheader("Portfolio Weights")
st.dataframe(weights_df, use_container_width=True)

# --------------------------------------------------
# CHART (FORCED VISIBILITY)
# --------------------------------------------------
st.subheader("Portfolio Weights Comparison")

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(assets))
width = 0.35

ax.bar(x - width/2, mv_weights, width, label="Mean-Variance")
ax.bar(x + width/2, bl_weights, width, label="Black-Litterman")

ax.set_xticks(x)
ax.set_xticklabels(assets, rotation=45)
ax.set_ylabel("Weight")
ax.set_ylim(0, max(bl_weights.max(), mv_weights.max()) + 0.1)
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray;">
        Made by <b>Kriya Chhajed</b> · AlphaStack
    </p>
    """,
    unsafe_allow_html=True
)
