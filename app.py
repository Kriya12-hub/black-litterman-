import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AlphaStack",
    page_icon="📊",
    layout="wide"
)

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <h1 style="margin-bottom:0;">📊 AlphaStack</h1>
    <p style="margin-top:-6px; color:gray;">
        Black–Litterman Portfolio Optimizer
    </p>
    <marquee behavior="scroll" direction="left">
        Market Equilibrium × Investor Views × Confidence-Weighted Allocation
    </marquee>
    """,
    unsafe_allow_html=True
)

st.write(
    "This application demonstrates how investor views are incorporated "
    "into portfolio construction using the Black–Litterman framework."
)

# ===============================
# DATA LOADING (ROBUST)
# ===============================
@st.cache_data
def load_data():
    tickers = [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "RELIANCE.NS",
        "TCS.NS"
    ]

    data = yf.download(
        tickers,
        start="2022-01-01",
        end="2024-01-01",
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(axis=1, how="any")

    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252

    return returns, cov_matrix


returns, cov_matrix = load_data()

assets = cov_matrix.columns.tolist()
n = len(assets)

if n < 2:
    st.error("Not enough valid assets to build portfolio.")
    st.stop()

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.title("Investor View")

asset_long = st.sidebar.selectbox(
    "Asset expected to outperform",
    assets,
    index=0
)

asset_short = st.sidebar.selectbox(
    "Asset expected to underperform",
    [a for a in assets if a != asset_long],
    index=0
)

expected_outperformance = st.sidebar.slider(
    "Expected Outperformance (%)",
    min_value=1.0,
    max_value=15.0,
    value=8.0,
    step=0.5
) / 100

confidence = st.sidebar.slider(
    "Confidence Level",
    min_value=10,
    max_value=90,
    value=75,
    step=5
) / 100

# ===============================
# BLACK–LITTERMAN CORE
# ===============================
tau = 0.05
delta = 2.5

market_weights = np.ones(n) / n
pi = delta * cov_matrix.values @ market_weights

# Relative view matrix
P = np.zeros((1, n))
P[0, assets.index(asset_long)] = 1
P[0, assets.index(asset_short)] = -1

Q = np.array([[expected_outperformance]])

Omega = np.array([
    [(1 - confidence) * (P @ cov_matrix.values @ P.T)[0, 0] + 1e-6]
])

inv_tau_cov = np.linalg.inv(tau * cov_matrix.values)
middle = np.linalg.inv(inv_tau_cov + P.T @ np.linalg.inv(Omega) @ P)

mu_bl = middle @ (inv_tau_cov @ pi + P.T @ np.linalg.inv(Omega) @ Q)
mu_bl = mu_bl.flatten()
mu_bl = np.asarray(mu_bl).reshape(n)


# ===============================
# 🔥 VIEW-AMPLIFIED RETURNS (KEY FIX)
# ===============================
view_adjustment = np.zeros(n)
view_adjustment[assets.index(asset_long)] += expected_outperformance * confidence
view_adjustment[assets.index(asset_short)] -= expected_outperformance * confidence

adjusted_returns = mu_bl + view_adjustment

posterior_returns = pd.Series(adjusted_returns, index=assets)

# ===============================
# WEIGHT CONVERSION (SOFTMAX)
# ===============================
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

bl_weights = softmax(adjusted_returns)

mv_raw = np.maximum(returns.mean().loc[assets].values, 0)
mv_weights = mv_raw / mv_raw.sum()

weights_df = pd.DataFrame(
    {
        "Mean–Variance": mv_weights,
        "Black–Litterman": bl_weights
    },
    index=assets
)

# ===============================
# OUTPUT TABLES
# ===============================
st.subheader("Adjusted Posterior Returns")
st.dataframe(
    posterior_returns.to_frame("Expected Return"),
    use_container_width=True
)

st.subheader("Portfolio Weights")
st.dataframe(weights_df, use_container_width=True)

# ===============================
# CHART (THIS WILL MOVE)
# ===============================
st.subheader("Portfolio Weights Comparison")

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(n)
width = 0.35

ax.bar(x - width/2, mv_weights, width, label="Mean–Variance")
ax.bar(x + width/2, bl_weights, width, label="Black–Litterman")

ax.set_xticks(x)
ax.set_xticklabels(assets, rotation=45)
ax.set_ylabel("Weight")
ax.set_ylim(0, max(bl_weights.max(), mv_weights.max()) + 0.1)
ax.legend()

st.pyplot(fig)

# ===============================
# FOOTER
# ===============================
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray;">
        Made by <b>Kriya Chhajed</b> · AlphaStack
    </p>
    """,
    unsafe_allow_html=True
)
