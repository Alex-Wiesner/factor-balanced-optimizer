from tools.optimizer import solve_weights
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
    type=click.FloatRange(min=0),
    default=2.0,
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
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    weights = solve_weights(
        tickers,
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
        click.echo(weights)


if __name__ == "__main__":
    cli()
