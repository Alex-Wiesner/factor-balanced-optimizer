from tools.optimizer import solve_weights
from tools.backtest import run_backtest
from tools.fetch_data import validate_tickers
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("tickers")
@click.option(
    "--variance-weight",
    type=click.FloatRange(min=0),
    default=0.10,
    help="Weight on the variance‑penalty term (larger → lower variance).",
)
@click.option(
    "--exposure-weight",
    type=click.FloatRange(min=0),
    default=0.02,
    help="Weight on the factor‑exposure reward term.",
)
@click.option(
    "--max-factor",
    default="2.0",
    help="Maximum absolute exposure per factor.",
)
@click.option(
    "--leverage-cap",
    type=click.FloatRange(min=0),
    default=2.0,
    help="Long/short leverage limit per asset.",
)
@click.option(
    "--gross-cap",
    type=click.FloatRange(min=0),
    default=None,
    help="Optional cap on gross exposure (sum of absolute weights).",
)
@click.option(
    "--solver",
    type=click.Choice(["ECOS", "OSQP", "SCS"], case_sensitive=False),
    default="ECOS",
    help="Primary CVXPY solver to use.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    help="Write resulting weights to a JSON file.",
)
def solve(
    tickers,
    variance_weight,
    exposure_weight,
    max_factor,
    leverage_cap,
    gross_cap,
    solver,
    output,
):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    valid_tickers, invalid_tickers = validate_tickers(ticker_list)
    weights = solve_weights(
        valid_tickers,
        variance_weight,
        exposure_weight,
        max_factor,
        leverage_cap,
        gross_cap,
        solver,
    )

    if output:
        weights.to_json(output, orient="index", indent=2)
    else:
        if invalid_tickers:
            click.echo(f"Removed invalid tickers: {
                       ', '.join(invalid_tickers)}")
        click.echo(weights)


@cli.command()
@click.argument("tickers")
@click.argument("start")
@click.argument("end")
@click.option("--freq", default="ME", help="Rebalance frequency (D,W,ME,Q)")
@click.option("--tc", default=5.0, help="Transaction cost in bps")
@click.option("--variance_weight", default=0.10, type=float)
@click.option("--exposure_weight", default=0.02, type=float)
@click.option("--max_factor", default="2.0", type=str)
@click.option("--leverage_cap", default=2.0, type=float)
@click.option("--gross_cap", default=None, type=float)
@click.option("--solver", default="ECOS")
@click.option("--no-plot", is_flag=True, default=False)
def backtest(
    tickers,
    start,
    end,
    freq,
    tc,
    variance_weight,
    exposure_weight,
    max_factor,
    leverage_cap,
    gross_cap,
    solver,
    no_plot,
):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    valid_tickers, invalid_tickers = validate_tickers(ticker_list)
    if type(max_factor) is str:
        max_factor = [float(t) for t in max_factor.split(",")]
    optimizer_kwargs = {
        "variance_weight": variance_weight,
        "exposure_weight": exposure_weight,
        "max_factor_abs": max_factor,
        "leverage_cap": leverage_cap,
        "gross_exposure_cap": gross_cap,
        "solver": solver,
    }
    run_backtest(
        tickers=ticker_list,
        start=start,
        end=end,
        rebalance_freq=freq,
        tc_bps=tc,
        optimizer_kwargs=optimizer_kwargs,
        plot=not no_plot,
    )


if __name__ == "__main__":
    cli()
