"""Risk model (Ledoit-Wolf covariance) and the long-only maximum-Sharpe optimizer with name and group caps.

(Equal/cap weight, inverse volatility, mean-variance, risk parity, HRP, CVaR, robust MV and Black-Litterman were
also tried during the policy search, which selected maximum Sharpe with Ledoit-Wolf covariance.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

ANN = 252


def cov_lw(R: pd.DataFrame) -> pd.DataFrame:
    """Annualised Ledoit-Wolf covariance of a (days x assets) daily-return window."""
    X = (R - R.mean()).fillna(0.0)
    S = LedoitWolf().fit(X.values).covariance_
    S = 0.5 * (S + S.T) * ANN + np.eye(len(X.columns)) * 1e-6
    return pd.DataFrame(S, index=R.columns, columns=R.columns)


def _constraints(w, cap, groups):
    cons = [cp.sum(w) == 1, w >= 0, w <= cap]
    for idx, gcap in (groups or []):
        if len(idx):
            cons.append(cp.sum(w[idx]) <= gcap)
    return cons


def _solve(prob):
    for s in ("CLARABEL", "OSQP", "SCS"):
        try:
            prob.solve(solver=s, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return True
        except Exception:
            continue
    return False


def min_variance(names, S, cap=0.1, groups=None):
    w = cp.Variable(len(names))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S.loc[names, names].values))), _constraints(w, cap, groups))
    if not _solve(prob):
        return pd.Series(1.0 / len(names), index=names)
    x = np.clip(w.value, 0, None)
    return pd.Series(x / x.sum(), index=names)


def max_sharpe(names, S, mu, cap=0.1, groups=None):
    """Tangency portfolio, homogeneous form: min y'Sy s.t. mu'y = 1, sum(y) = k, 0 <= y <= cap*k (+ group caps)."""
    m = mu.reindex(names).fillna(0).values
    if (m <= 0).all():
        return min_variance(names, S, cap, groups)
    y = cp.Variable(len(names))
    k = cp.Variable()
    cons = [m @ y == 1, y >= 0, k >= 0, cp.sum(y) == k, y <= cap * k]
    for idx, gcap in (groups or []):
        if len(idx):
            cons.append(cp.sum(y[idx]) <= gcap * k)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(S.loc[names, names].values))), cons)
    if not _solve(prob) or k.value is None or k.value <= 1e-9:
        return min_variance(names, S, cap, groups)
    w = np.clip(y.value / k.value, 0, None)
    return pd.Series(w / w.sum(), index=names)


def integer_lots(weights: pd.Series, exposure: float, price: pd.Series, capital: float, lot: int, buy_fee: float):
    """Whole 100-share lots closest to the target rupiah amounts without exceeding the cash (MILP, HiGHS)."""
    names = list(weights.index)
    p = price.reindex(names).values
    n = cp.Variable(len(names), integer=True)
    dev = cp.Variable(len(names))
    val = cp.multiply(n, p * lot)
    tgt = weights.values * exposure * capital
    cons = [n >= 0, dev >= val - tgt, dev >= tgt - val, cp.sum(val) * (1 + buy_fee) <= exposure * capital]
    cp.Problem(cp.Minimize(cp.sum(dev)), cons).solve(solver="HIGHS")
    nv = np.round(n.value).astype(int)
    book = pd.DataFrame({"lots": nv, "shares": nv * lot, "price": p, "value_rp": nv * p * lot,
                         "target_weight": weights.values * exposure}, index=names)
    book["actual_weight"] = book["value_rp"] / capital
    return book[book["lots"] > 0]
