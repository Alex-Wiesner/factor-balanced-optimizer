from tools.fetch_data import fetch_factors, fetch_returns
import cvxpy as cp
import numpy as np
import pandas as pd
import datetime
from sklearn.covariance import LedoitWolf


def solve_weights(
    tickers: list[str],
    variance_weight: float = 0.3,
    exposure_weight: float = 0.02,
    max_factor_abs: float = 2.0,
    leverage_cap: float = 2.0,
    gross_exposure_cap: float | None = None,
    solver: str = "ECOS",
) -> pd.Series:
    if any(v < 0 for v in (variance_weight, exposure_weight)):
        raise ValueError("Weights must be non‑negative.")
    if max_factor_abs < 0:
        raise ValueError("max_factor_abs cannot be negative.")

    ret = fetch_returns(
        tickers, start=datetime.date.today() - datetime.timedelta(365))
    Sigma = LedoitWolf().fit(ret).covariance_
    cond = np.linalg.cond(Sigma)
    if cond > 1e12:
        raise ValueError(
            f"Covariance matrix ill‑conditioned (cond={cond:.2e})")

    F = fetch_factors(tickers).values
    n = len(tickers)
    w = cp.Variable((n, 1))

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
    try:
        prob.solve(solver=getattr(cp, solver))
    except (cp.SolverError, AttributeError):
        # fallback to OSQP, then SCS
        for alt in ("OSQP", "SCS"):
            try:
                prob.solve(solver=getattr(cp, alt))
                break
            except cp.SolverError:
                continue
        else:
            raise RuntimeError("All solvers failed.")

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver finished with status {prob.status}")

    weights = pd.Series(w.value.ravel(), index=tickers)

    return weights
