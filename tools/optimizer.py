from tools.fetch_data import fetch_factors, fetch_returns
import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def solve_weights(
    tickers: list[str],
    variance_weight: float = 0.3,
    exposure_weight: float = 0.02,
    max_factor_abs: list[float] | float = 2.0,
    leverage_cap: float = 2.0,
    gross_exposure_cap: float | None = None,
    solver: str = "ECOS",
) -> pd.Series:
    if any(v < 0 for v in (variance_weight, exposure_weight)):
        raise ValueError("Weights must be non‑negative.")

    ret = fetch_returns(tickers, period="1y")
    Sigma = LedoitWolf().fit(ret).covariance_
    cond = np.linalg.cond(Sigma)
    if cond > 1e12:
        raise ValueError(
            f"Covariance matrix ill‑conditioned (cond={cond:.2e})")

    F = fetch_factors(tickers).values
    n = F.shape
    w = cp.Variable((n[0], 1))

    max_vec = np.asarray(max_factor_abs, dtype=float)
    if max_vec.size == 1:
        max_vec = np.full(n[1], float(max_factor_abs[0]))
    elif max_vec.size != n[1]:
        raise ValueError(
            "max_factor_abs vector length must equal number of factors.")

    exposure = F.T @ w
    avrg_exposure = cp.mean(exposure)

    balance_obj = cp.sum_squares(exposure - avrg_exposure)
    variance_obj = variance_weight * cp.quad_form(w, Sigma)
    reward_obj = exposure_weight * cp.norm(exposure, 2)

    obj = cp.Minimize(balance_obj + variance_obj + reward_obj)

    constraints = [cp.sum(w) == 1, w <= leverage_cap, w >= -leverage_cap]

    if gross_exposure_cap is not None:
        constraints.append(cp.norm1(w) <= gross_exposure_cap)

    constraints.append(cp.abs(exposure) <= max_vec)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=getattr(cp, solver))
    except (cp.SolverError, AttributeError) as exc:
        raise RuntimeError(f"Solver error:{exc}")

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver finished with status {prob.status}")

    weights = pd.Series(w.value.ravel(), index=tickers)

    return weights
