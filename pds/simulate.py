"""Monte Carlo stress test of a decision: filtered historical simulation (FHS) plus an MSCI-shock jump.

* GARCH(1,1)-t standardised residuals of every stock are block-bootstrapped jointly (mean block 10 days) and rescaled
  to each stock's current conditional volatility, so the scenarios carry today's volatility regime.
* Residuals are demeaned and every stock gets an explicit drift mu = rf + 6% x beta, so scenarios do not inherit the
  past drift of the survivors.
* With probability p an MSCI classification shock hits on a random day: "classic" (day-0 -1% to -4%, other countries)
  or "Indonesia-2026" (day-0 -8% to -15%, then a further -5% to -20% over 40 days), scaled per name by multipliers
  (1.15x / 1.5x for flagged low-float names, 0.95x / 0.7x for high-float blue chips).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from arch import arch_model


def stationary_block_indices(n_hist: int, n_paths: int, H: int, mean_block: int, rng, pool=None):
    pool = np.arange(n_hist) if pool is None else np.asarray(pool)
    m = len(pool)
    idx = np.empty((n_paths, H), dtype=np.int64)
    cur = rng.integers(m, size=n_paths)
    for h in range(H):
        idx[:, h] = pool[cur]
        jump = rng.random(n_paths) < 1.0 / mean_block
        cur = np.where(jump, rng.integers(m, size=n_paths), (cur + 1) % m)
    return idx


def garch_standardise(R: pd.DataFrame, min_obs=250):
    """Per-asset GARCH(1,1)-t fit; returns standardised residuals and the next-day conditional vol."""
    Z = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    sig_next = {}
    for c in R.columns:
        x = R[c].dropna() * 100
        if len(x) < min_obs:
            s = x.std()
            Z.loc[x.index, c] = (x - x.mean()) / s
            sig_next[c] = x.tail(63).std() / 100
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(x, vol="GARCH", p=1, q=1, dist="t", mean="Constant").fit(disp="off", show_warning=False)
            Z.loc[x.index, c] = res.std_resid
            f = res.forecast(horizon=1, reindex=False)
            sig_next[c] = float(np.sqrt(f.variance.values[-1, 0])) / 100
        except Exception:
            Z.loc[x.index, c] = (x - x.mean()) / x.std()
            sig_next[c] = x.tail(63).std() / 100
    return Z, pd.Series(sig_next)


def simulate_paths(R_hist: pd.DataFrame, H: int, n_paths: int, family: str, seed=0, mean_block=10,
                   jkse_vol=None, Z=None, sig_next=None, mu_daily=None, jump=None, risk_names=None):
    """Return array (n_paths, H, n_assets) of simple daily returns."""
    rng = np.random.default_rng(seed)
    X = R_hist.fillna(0.0).values
    n = len(X)
    if family in ("fhs", "msci_jump"):
        Zv = Z.reindex(R_hist.index).fillna(0.0).values
        idx = stationary_block_indices(n, n_paths, H, mean_block, rng)
        sims = Zv[idx] * sig_next.reindex(R_hist.columns).values[None, None, :]
        if mu_daily is not None:
            sims = sims + mu_daily.reindex(R_hist.columns).fillna(0).values[None, None, :]
        if family == "msci_jump" and jump is not None:
            sims = add_msci_jump(sims, jump, R_hist.columns, risk_names, rng)
        return np.clip(sims, -0.35, 0.35)
    raise ValueError(family)


def add_msci_jump(sims, jump: dict, cols, risk_names, rng):
    """With prob p an MSCI classification shock hits on a uniformly drawn day.
    Type 'classic' (prob 1-q): market day-0 shock ~ U(-4%,-1%), no drift (other-country analogs).
    Type 'severe'  (prob q):   day-0 ~ U(-15%,-8%) then extra drift over 40 days totalling U(-20%,-5%)
                               (Indonesia 28-Jan-2026 analog). Risky names get an extra day-0 hit."""
    P, H, N = sims.shape
    hit = rng.random(P) < jump["p"]
    severe = rng.random(P) < jump["q_severe"]
    day = rng.integers(0, H, size=P)
    # per-name multipliers: day-0 (28-Jan-2026 cross-section) and follow-through drift
    # (governance-shock analogs + Jan-Aug 2026: low-float conglomerates ~1.5x, high-float blue chips ~0.7x)
    m0 = np.asarray(jump.get("m0", np.ones(N)), dtype=float)
    md = np.asarray(jump.get("md", np.ones(N)), dtype=float)
    for i in np.where(hit)[0]:
        d = day[i]
        if severe[i]:
            s0 = rng.uniform(-0.15, -0.08)
            drift_total = rng.uniform(-0.20, -0.05)
            L = min(40, H - d - 1)
            if L > 0:
                sims[i, d + 1:d + 1 + L, :] += (np.log1p(drift_total) / 40) * md[None, :]
        else:
            s0 = rng.uniform(-0.04, -0.01)
        sims[i, d, :] = (1 + sims[i, d, :]) * (1 + s0 * m0) - 1
    return sims


def portfolio_terminal(sims: np.ndarray, w: np.ndarray, e: float, rf_daily: float, cost=0.0):
    """Buy-and-hold terminal simple return of e*w (risky) + (1-e) cash over the horizon."""
    growth = np.prod(1 + sims, axis=1)                       # (P, N)
    risky = growth @ w                                        # w sums to 1
    H = sims.shape[1]
    return e * risky + (1 - e) * (1 + rf_daily) ** H - 1 - cost


def path_drawdown(sims: np.ndarray, w: np.ndarray, e: float, rf_daily: float):
    cum = np.cumprod(1 + sims, axis=1) @ w                    # (P, H) value of risky sleeve
    H = sims.shape[1]
    cash = (1 + rf_daily) ** np.arange(1, H + 1)
    v = e * cum + (1 - e) * cash[None, :]
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(v), 1)), v], axis=1), axis=1)[:, 1:]
    return (v / peak - 1).min(axis=1)


def risk_table(term: np.ndarray, dd: np.ndarray | None = None, gamma=5.0):
    x = np.clip(1 + term, 1e-6, None)
    ce = np.mean(x ** (1 - gamma)) ** (1 / (1 - gamma)) - 1
    q5 = np.quantile(term, 0.05)
    out = {"mean": term.mean(), "median": np.median(term), "p_loss": (term < 0).mean(), "p_loss_10": (term < -0.10).mean(),
           "VaR95": -q5, "CVaR95": -term[term <= q5].mean(), "CE_g5": ce, "best5": np.quantile(term, 0.95)}
    if dd is not None:
        out["median_maxDD"] = np.median(dd); out["p_DD_gt_15"] = (dd < -0.15).mean()
    return out
