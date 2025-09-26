import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
from math import sqrt
from typing import Optional, Mapping


def _fetch_info_with_backoff(ticker: str, max_retries: int = 5) -> Mapping:
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if info:
                return info
        except Exception:
            pass

        wait_seconds = 2**attempt
        time.sleep(wait_seconds)

    raise RuntimeError(
        f"Unable to fetch fundamental data for ticker '{ticker}' after {
            max_retries
        } attempts."
    )


def _standardize(factor: pd.Series) -> pd.Series:
    return (factor - factor.mean()) / factor.std()


def _momentum_calc(ticker: str) -> float:
    start = datetime.date.today() - datetime.timedelta(365)
    end = datetime.date.today() - datetime.timedelta(30)
    returns = fetch_returns([ticker], start, end, "1mo")
    momentum = (returns + 1).prod() - 1
    return float(momentum.iloc[0])


def _vol_calc(ticker: str) -> float:
    returns = fetch_returns(
        [ticker],
        datetime.date.today() - datetime.timedelta(180),
    )
    vol = returns.std().iloc[0] / sqrt(252)
    return float(vol)


def fetch_returns(
    tickers: list[str],
    start: str | datetime.date,
    end: Optional[str | datetime.date] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if raw.empty:
        raise RuntimeError("Download returned no data for tickers.")

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    returns = raw.pct_change().dropna()

    if returns.shape[1] != len(tickers):
        raise ValueError(
            "Returned column count does not match number of tickers. "
            "Check that all tickers are valid."
        )

    return returns


def fetch_factors(tickers: list[str]) -> pd.DataFrame:
    caps, pb, momentum, vol = {}, {}, {}, {}

    for ticker in tickers:
        info = _fetch_info_with_backoff(ticker)
        caps[ticker] = info.get("marketCap", np.nan)
        price = info.get("previousClose")
        book_val = info.get("bookValue")
        pb[ticker] = price / book_val if book_val and book_val != 0 else np.nan
        momentum[ticker] = _momentum_calc(ticker)
        vol[ticker] = _vol_calc(ticker)

    size_series = np.log(pd.Series(caps))
    pb_series = pd.Series(pb)
    momentum_series = pd.Series(momentum)
    vol_series = pd.Series(vol)

    factors = pd.concat(
        [
            _standardize(size_series),
            _standardize(pb_series),
            _standardize(momentum_series),
            _standardize(vol_series),
        ],
        axis=1,
    )
    factors.columns = ["Size", "P/B", "Momentum", "Vol"]

    return factors
