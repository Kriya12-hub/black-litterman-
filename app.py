import streamlit as st
import pandas as pd
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AlphaStack",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- GLOBAL CSS --------------------
st.markdown("""
<style>
:root {
  --bg: #0E1117;
  --panel: #111827;
  --card: #161B22;
  --primary: #4DA3FF;
  --accent: #00E5A8;
  --text: #E6EAF0;
  --muted: #9CA3AF;
  --radius: 14px;
}

/* App background */
.stApp {
  background: linear-gradient(120deg, #0E1117, #0B1220);
  color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #4F46E5, #7C3AED);
  color: white;
}
section[data-testid="stSidebar"] * {
  color: white !important;
}

/* Header */
.header {
  background: linear-gradient(90deg, #0E1117, #111827);
  padding: 20px 24px;
  border-radius: var(--radius);
  margin-bottom: 22px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
.logo {
  font-size: 34px;
  font-weight: 800;
  letter-spacing: .3px;
}
.logo span { color: var(--primary); }

/* Marquee */
.marquee {
  overflow: hidden;
  white-space: nowrap;
  margin-top: 8px;
}
.marquee span {
  display: inline-block;
  padding-left: 100%;
  animation: marquee 18s linear infinite;
  color: var(--muted);
  font-size: 14px;
}
@keyframes marquee {
  0% { transform: translateX(0); }
  100% { transform: translateX(-100%); }
}

/* Cards */
.card {
  background: var(--card);
  border-radius: var(--radius);
  padding: 18px;
  box-shadow: 0 12px 28px rgba(0,0,0,.35);
}
.card h4 {
  margin: 0 0 6px 0;
  font-weight: 600;
}
.card .value {
  font-size: 26px;
  font-weight: 800;
}
.card.blue { background: linear-gradient(135deg,#2563EB,#1D4ED8); }
.card.orange { background: linear-gradient(135deg,#F59E0B,#F97316); }
.card.purple { background: linear-gradient(135deg,#7C3AED,#5B21B6); }
.card.green { background: linear-gradient(135deg,#10B981,#059669); }

/* Tables */
[data-testid="stDataFrame"] {
  background: var(--card);
  border-radius: var(--radius);
  overflow: hidden;
}

/* Charts */
[data-testid="stPlotlyChart"], [data-testid="stAltairChart"] {
  background: var(--card);
  border-radius: var(--radius);
  padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## AlphaStack")
st.sidebar.markdown("**Portfolio Intelligence**")

st.sidebar.markdown("### Investor View")
asset_long = st.sidebar.selectbox(
    "Asset expected to outperform",
    ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]
)

asset_short = st.sidebar.selectbox(
    "Asset expected to underperform",
    ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"],
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

st.sidebar.markdown("---")
st.sidebar.markdown("**AlphaStack ©**")

# -------------------- HEADER --------------------
st.markdown("""
<div class="header">
  <div class="logo">Alpha<span>Stack</span></div>
  <div class="marquee">
    <span>
      Black–Litterman Portfolio Optimizer • Market Equilibrium + Investor Views • Confidence-Weighted Allocation • Institutional-Grade Portfolio Logic
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------- METRIC CARDS --------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card purple">
      <h4>Total Assets</h4>
      <div class="value">₹42,00,000</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card blue">
      <h4>Expected Return</h4>
      <div class="value">12.8%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card orange">
      <h4>Portfolio Risk</h4>
      <div class="value">18.4%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card green">
      <h4>Confidence</h4>
      <div class="value">{confidence}%</div>
    </div>
    """.format(confidence=confidence), unsafe_allow_html=True)

st.markdown("")

# -------------------- MAIN CONTENT --------------------
left, right = st.columns([2.2, 1])

# ----- MOCK DATA (replace with your BL outputs) -----
tickers = ["HDFCBANK.NS","ICICIBANK.NS","INFY.NS","RELIANCE.NS","TCS.NS"]
mv = np.array([0.05, 0.35, 0.35, 0.13, 0.12])
bl = np.array([0.18, 0.17, 0.00, 0.61, 0.04])

df = pd.DataFrame({
    "Mean–Variance": mv,
    "Black–Litterman": bl
}, index=tickers)

with left:
    st.subheader("Portfolio Weights Comparison")
    st.bar_chart(df)

with right:
    st.subheader("Allocation Split")
    donut = pd.DataFrame({
        "Allocation": ["Black–Litterman","Mean–Variance"],
        "Weight": [bl.sum(), mv.sum()]
    })
    st.write(donut)

st.markdown("")

# -------------------- TABLE --------------------
st.subheader("Detailed Weights")
st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div style="margin-top:30px; color:#9CA3AF; font-size:13px;">
AlphaStack is a decision-support tool for scenario analysis. It does not provide investment advice.
</div>
""", unsafe_allow_html=True)
