from tools.fetch_data import fetch_factors, fetch_returns
import cvxpy as cp
import pandas as pd


def solve_weights(
    tickers: list[str],
    variance_weight: float = 0.3,
    exposure_weight: float = 0.02,
    max_factor_abs: float = 2.0,
    leverage_cap: float = 2.0,
    gross_exposure_cap: float | None = None,
) -> pd.Series:
    Sigma = fetch_returns(tickers, start="2020-01-01").cov().values
    F = fetch_factors(tickers).values
    w = cp.Variable((Sigma.shape[0], 1))

    exposure = F.T @ w
    avrg_exposure = cp.mean(exposure)

    balance_obj = cp.sum_squares(exposure - avrg_exposure)
    variance_obj = variance_weight * cp.quad_form(w, Sigma)
    reward_obj = exposure_weight * cp.norm(exposure, 2)

    obj = cp.Minimize(balance_obj + variance_obj + reward_obj)

    constraints = [cp.sum(w) == 1, w <= leverage_cap, w >= -leverage_cap]
    if gross_exposure_cap is not None:
        constraints.append(cp.norm1(w) <= gross_exposure_cap)
    if max_factor_abs > 0:
        constraints.append(cp.abs(exposure) <= max_factor_abs)

    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver finished with status {prob.status}.")

    weights = w.value.ravel()

    return pd.Series(weights, index=tickers)
