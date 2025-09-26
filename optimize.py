from tools.optimizer import solve_weights
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("tickers")
@click.option("--variance_weight", type=float, default=0.3)
@click.option("--exp_weight", type=float, default=0.02)
@click.option("--max_factor", type=float, default=2)
@click.option("--leverage_cap", type=float, default=2)
@click.option("--gross_cap", type=float, default=None)
def solve(tickers, variance_weight, exp_weight, max_factor, leverage_cap, gross_cap):
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    var_w, exp_w, max_factor_abs, leverage_cap, gross_exp_cap = (
        variance_weight,
        exp_weight,
        max_factor,
        leverage_cap,
        gross_cap,
    )
    print(
        solve_weights(
            tickers, var_w, exp_w, max_factor_abs, leverage_cap, gross_exp_cap
        )
    )


if __name__ == "__main__":
    cli()
