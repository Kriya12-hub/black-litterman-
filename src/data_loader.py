import pandas as pd
import yfinance as yf

def load_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data

def compute_returns(price_data):
    returns = price_data.pct_change().dropna()
    return returns
