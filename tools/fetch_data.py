import yfinance as yf
import pandas as pd
from tools.cache import make_key, cached_fetch
import numpy as np
import datetime
import time
from math import sqrt
from typing import Optional, Mapping
import logging

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def _is_valid(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).info.get("regularMarketPrice")
        return info is not None
    except Exception:
        return False


def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []

    for ticker in tickers:
        if _is_valid(ticker):
            valid.append(ticker)
        else:
            invalid.append(ticker)

    if invalid:
        _logger.info(f"Invalid tickers:{', '.join(invalid)}")

    return valid, invalid


def _fetch_info_with_backoff(ticker: str, max_retries: int = 5) -> Mapping:
    key = make_key("info", ticker=ticker.upper(), max_retries=max_retries)

    def _inner_fetch(ticker: str, max_retries: int = 5) -> Mapping:
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

    return cached_fetch(key, _inner_fetch, ticker=ticker, max_retries=max_retries)


def _fetch_price_with_backoff(
    ticker: str,
    start: Optional[str | datetime.date] = None,
    end: Optional[str | datetime.date] = None,
    period: str = "max",
    interval: str = "1d",
    max_retries: int = 5,
) -> Mapping:
    key = make_key(
        "price",
        ticker=ticker.upper(),
        start=str(start) if start else "",
        end=str(end) if end else "",
        period=period,
        interval=interval,
        max_retries=max_retries,
    )

    def _inner_fetch(
        ticker: str,
        start: Optional[str | datetime.date] = None,
        end: Optional[str | datetime.date] = None,
        period: str = "max",
        interval: str = "1d",
        max_retries: int = 5,
    ) -> Mapping:
        for attempt in range(max_retries):
            try:
                history = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                if not history.empty:
                    return history
            except Exception:
                pass

            wait_seconds = 2**attempt
            time.sleep(wait_seconds)

        raise RuntimeError(
            f"Unable to fetch historical data for ticker '{ticker}' after {
                max_retries
            } attempts."
        )

    return cached_fetch(
        key,
        _inner_fetch,
        ticker=ticker,
        start=start,
        end=end,
        period=period,
        interval=interval,
        max_retries=max_retries,
    )


def _standardize(factor: pd.Series) -> pd.Series:
    return (factor - factor.mean()) / factor.std()


def _momentum_calc(ticker: str) -> float:
    returns = fetch_returns([ticker], period="1y", interval="1mo")
    momentum = (returns + 1).prod() - 1
    return float(momentum.iloc[0])


def _vol_calc(ticker: str) -> float:
    returns = fetch_returns([ticker], period="6mo")
    vol = returns.std().iloc[0] / sqrt(252)
    return float(vol)


def fetch_returns(
    tickers: list[str],
    start: Optional[str | datetime.date] = None,
    end: Optional[str | datetime.date] = None,
    period: str = "max",
    interval: str = "1d",
) -> pd.DataFrame:
    raw = pd.DataFrame()

    for ticker in tickers:
        raw[ticker] = _fetch_price_with_backoff(
            ticker, start=start, end=end, period=period, interval=interval
        )["Close"]

    returns = raw.pct_change(fill_method=None).dropna()

    return returns


def fetch_factors(tickers: list[str]) -> pd.DataFrame:
    caps, pb, momentum, vol, roe, gm, liquidity = {}, {}, {}, {}, {}, {}, {}

    for ticker in tickers:
        info = _fetch_info_with_backoff(ticker)
        roe[ticker] = info.get("returnOnEquity", np.nan)
        liquidity[ticker] = info.get("averageVolume", np.nan)
        gm[ticker] = info.get("grossMargins", np.nan)
        caps[ticker] = info.get("marketCap", np.nan)
        price = info.get("previousClose")
        book_val = info.get("bookValue")
        pb[ticker] = price / book_val if book_val and book_val != 0 else np.nan
        momentum[ticker] = _momentum_calc(ticker)
        vol[ticker] = _vol_calc(ticker)

    roe_series = pd.Series(roe)
    liquidity_series = pd.Series(liquidity)
    gm_series = pd.Series(gm)
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
            _standardize(roe_series),
            _standardize(gm_series),
            _standardize(liquidity_series),
        ],
        axis=1,
    )
    factors.fillna(factors.median(), inplace=True)

    factors.columns = [
        "Size",
        "P/B",
        "Momentum",
        "Vol",
        "ROE",
        "GrossMargins",
        "Liquidity",
    ]

    return factors
