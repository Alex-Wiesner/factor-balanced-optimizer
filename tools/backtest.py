import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from tools.fetch_data import validate_tickers, fetch_factors, fetch_returns
from tools.optimizer import solve_weights


def _annualise_factor(series: pd.Series) -> float:
    return series.mean() * 252 / (series.std() * np.sqrt(252))


def _max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    dd = nav / running_max - 1
    return dd.min()


def _turnover(prev_w: pd.Series, cur_w: pd.Series) -> float:
    return np.abs(cur_w - prev_w).sum()


def run_backtest(
    tickers: list[str],
    start: str,
    end: str,
    *,
    rebalance_freq: str = "M",
    tc_bps: float = 5,
    optimizer_kwargs: dict | None = None,
    plot: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    tickers, invalid = validate_tickers(tickers)

    price_df = fetch_returns(tickers, start=start, end=end, period="max", interval="1d")
    price_level = (price_df + 1).cumprod()

    factor_df = fetch_factors(tickers)

    all_days = price_level.index
    rebalance_dates = pd.date_range(start=start, end=end, freq=rebalance_freq)
    rebalance_dates = rebalance_dates.intersection(all_days)

    nav = pd.Series(index=all_days, dtype=float)
    nav.iloc[0] = 1.0
    weight_history = pd.DataFrame(index=rebalance_dates, columns=tickers, dtype=float)
    prev_weights = pd.Series(0.0, index=tickers)

    for i, cur_date in enumerate(tqdm(rebalance_dates, desc="Rebalancing")):
        cur_weights = solve_weights(tickers, **optimizer_kwargs)
        weight_history.loc[cur_date] = cur_weights
        trn_cost = tc_bps / 10000 * _turnover(prev_weights, cur_weights)
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
        else:
            next_date = all_days[-1] + pd.Timedelta(days=1)

        holding_mask = (price_level.index >= cur_date) & (price_level.index < next_date)
        period_prices = price_level.loc[holding_mask]

        daily_ret = (period_prices.pct_change().fillna(0) @ cur_weights) - trn_cost

        if i == 0:
            nav.loc[holding_mask] = (1.0 + daily_ret).cumprod()
        else:
            first_date_in_window = price_level.index[holding_mask][0]
            prev_idx = nav.index.get_loc(first_date_in_window) - 1
            prev_nav = nav.iloc[prev_idx]
            nav.loc[holding_mask] = prev_nav * (1 + daily_ret).cumprod()

        prev_weights = cur_weights.copy()

    nav.ffill(inplace=True)

    daily_ret_series = nav.pct_change().dropna()
    ann_ret = daily_ret_series.mean() * 252
    ann_vol = daily_ret_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    max_dd = _max_drawdown(nav)
    avg_turnover = _turnover(weight_history.shift(1).fillna(0), weight_history).mean()

    print("\n=== Back‑test performance ===")
    print(f"Period                : {start} → {end}")
    print(f"Tickers (valid)       : {', '.join(tickers)}")
    print(f"Rebalance frequency   : {rebalance_freq}")
    print(f"Transaction cost (bps): {tc_bps:.2f}")
    print(f"Cumulative return     : {(nav.iloc[-1] - 1) * 100:.2f}%")
    print(f"Annualised return     : {ann_ret * 100:.2f}%")
    print(f"Annualised volatility : {ann_vol * 100:.2f}%")
    print(f"Sharpe ratio          : {sharpe:.3f}")
    print(f"Max draw‑down         : {max_dd * 100:.2f}%")
    print(f"Average turnover      : {avg_turnover * 100:.2f}%\n")

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)

        axes[0].sharex(axes[1])

        axes[0].plot(nav.index, nav.values, label="Portfolio NAV")
        axes[0].set_ylabel("NAV (starting at 1.0)")
        axes[0].set_title("Back‑test equity curve")
        axes[0].legend()

        rolling_sharpe = daily_ret_series.rolling(window=30).apply(
            _annualise_factor, raw=False
        )
        axes[1].plot(rolling_sharpe.index, rolling_sharpe, color="darkorange")
        axes[1].set_ylabel("30‑day rolling Sharpe")
        axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")

        exposure_matrix = factor_df.T @ weight_history.T
        avg_abs_exposure = exposure_matrix.abs().mean(axis=1)
        fake_dates = pd.date_range(
            start=nav.index[0], periods=len(avg_abs_exposure), freq="D"
        )
        axes[2].bar(fake_dates, avg_abs_exposure.values, color="steelblue")
        axes[2].set_ylabel("Avg. |exposure| (std‑dev units)")
        axes[2].set_title("Average absolute factor exposure across rebalances")
        axes[2].set_xticks(fake_dates)
        axes[2].set_xticklabels(avg_abs_exposure.index, rotation=45)

        plt.tight_layout()
        fig_path = Path.cwd() / "backtest_report.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {fig_path}")

    return nav, weight_history
