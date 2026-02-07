import pandas as pd
import yfinance as yf

def load_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by="column")

    # Case 1: MultiIndex columns (most common on cloud)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"]
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"]
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in data")

    # Case 2: Single-level columns (local/simple case)
    else:
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in data")

    return prices.dropna()

def compute_returns(price_data):
    return price_data.pct_change().dropna()
