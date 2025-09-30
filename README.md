# Factor-Balanced Portfolio Optimizer

## Installation

```bash
git clone https://github.com/Alex-Wiesner/factor-balanced-optimizer
cd factor-balanced-optimizer
pip install -r requirements.txt
```

## Quick Start

```bash
# Solve for a static set of tickers (weights printed to console)
python optimize.py solve "AAPL,MSFT,AMZN,GOOGL,TSLA"

# Run a full back-test (default montly rebalancing)
python optimize.py back-test "AAPL,MSFT,AMZN,GOOGL,TSLA" 2020-01-01 2024-12-31
```

## CLI Reference

```bash
python optimize.py solve TICKERS [OPTIONS]
```

|Option|Description|Default|
|:-----|:----------|:-----:|
|--variance-weight|Weight on the variance-penalty term (higher → lower variance).|0.10|
|--exposure-weight|Weight on the factor-exposure reward term.|0.02|
|--max-factor|Maximum absolute exposure per factor (scalar or comma-separated list).|2.0|
|--leverage-cap|Optional cap on gross exposure|None|
|--solver|CVXPY solver to use (ECOS, OSQP, SCS).|ECOS|
|--output|Write resulting weights to a JSON file instead of stdout.|None|

**Output** - a Pandas Series printed to the console (or saved as JSON) with ticker symbols as the index and portfolio weights as values.

```bash
python optimize.py backtest TICKERS START END [OPTIONS]
```

|Argument|Description|
|:-------|:----------|
|TICKERS|Comma-separated list of tickers (e.g."AAPL,MSFT,AMZN").|
|START|Back-test start date (YYYY-MM-DD).|
|END|Back-test end date (YYYY-MM-DD).|

|Option|Description|Default|
|:-----|:----------|:-----:|
|--freq|Rebalance frequency (D, W, ME, Q).|ME (month-end)|
|--tc|Transaction cost in basis points per trade.|5.0|
|--variance-weight|Weight on the variance-penalty term (higher → lower variance).|0.10|
|--exposure-weight|Weight on the factor-exposure reward term.|0.02|
|--max-factor|Maximum absolute exposure per factor (scalar or comma-separated list).|2.0|
|--leverage-cap|Optional cap on gross exposure|None|
|--solver|CVXPY solver to use (ECOS, OSQP, SCS).|ECOS|
|--no-plot|Skip saving the performance chart.|False|

**Result** - prints a performance summary and writes backtes_report.png (unless --no-plot).

## License

MIT License @ 2025 Alex R. Wiesner
