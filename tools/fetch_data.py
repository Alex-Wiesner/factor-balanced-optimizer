import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from math import sqrt


def _standardize(factor: pd.Series) -> pd.Series:
    return (factor - factor.mean()) / factor.std()


def _momentum_calc(ticker=str) -> float:
    returns = fetch_returns(
        [ticker],
        datetime.date.today() - datetime.timedelta(365),
        datetime.date.today() - datetime.timedelta(30),
        "1mo",
    )
    momentum = (returns + 1).prod() - 1
    return momentum.iloc[0]


def _vol_calc(ticker: str) -> float:
    returns = fetch_returns(
        [ticker],
        datetime.date.today() - datetime.timedelta(180),
    )
    vol = returns.std().iloc[0] / sqrt(252)
    return float(vol)


def fetch_returns(
    tickers: list[str], start: str, end: str | None = None, interval: str = "1d"
) -> pd.DataFrame:
    returns = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )["Close"]
    returns = returns.pct_change().dropna()
    assert returns.shape[1] == len(tickers)
    return returns


def fetch_factors(tickers: list[str]) -> pd.DataFrame:
    caps, value, momentum, vol = {}, {}, {}, {}

    for ticker in tickers:
        info = yf.Ticker(ticker).info
        caps[f"{ticker}"] = info["marketCap"]
        value[f"{ticker}"] = info["previousClose"] / info["bookValue"]
        momentum[f"{ticker}"] = _momentum_calc(ticker)
        vol[f"{ticker}"] = _vol_calc(ticker)

    size = np.log(pd.Series(caps))
    value = pd.Series(value)
    momentum = pd.Series(momentum)
    vol = pd.Series(vol)

    size_z = _standardize(size)
    value_z = _standardize(value)
    momentum_z = _standardize(momentum)
    vol_z = _standardize(vol)

    factors = pd.concat([size_z, value_z, momentum_z, vol_z], axis=1)
    factors.columns = ["Size", "P/B", "Momentum", "Vol"]

    return factors
