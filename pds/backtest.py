"""Walk-forward monthly backtest of the policy with IDX frictions and the cash/equity overlay.

Per month-end t: information <= close(t) -> target weights -> trade at the close of e(t) (next session) -> hold to
e(t+1). Costs: buy 0.15% / sell 0.25% plus half the relative tick as spread. Names locked at the auto-rejection
limits on the execution day cannot be bought (upper limit) or sold (lower limit).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from . import portfolio as PF
from .features import STRUCT_FEATS


class Context:
    """Data shared by every rebalance: panel, market table, daily returns, signal, mask, exposure, sectors."""

    def __init__(self, panel, mkt, r_daily, signal, mask, exposure, sectors):
        self.panel = panel[panel["in_top50"]]
        self.mkt = mkt
        self.r = r_daily
        self.signal = signal
        self.mask = mask
        self.exposure = exposure
        self.sectors = sectors.to_dict()
        self.dates = self.panel.index.get_level_values("date").unique().sort_values()
        z = self.panel[STRUCT_FEATS].groupby(level="date").rank(pct=True)
        self.struct_score = z.mean(axis=1)             # high = illiquid relative to its size ("paper cap")

    def universe(self, t):
        return self.panel.xs(t, level="date")

    def cov(self, t, names, lookback):
        R = self.r.loc[:t].tail(lookback)
        R = R[[c for c in self.universe(t).index if c in R.columns]]
        R = R.loc[:, R.notna().sum() >= min(60, int(lookback * 0.5))]
        S = PF.cov_lw(R)
        ok = [n for n in names if n in S.index]
        return S.loc[ok, ok]


def signal_to_mu(sig: pd.Series, vol: pd.Series, kappa: float) -> pd.Series:
    """Grinold-Kahn: mu_i = kappa * sigma_i * z_i (kappa plays the role of an information coefficient)."""
    z = (sig - sig.mean()) / (sig.std() + 1e-12)
    return kappa * vol.reindex(sig.index).fillna(vol.median()) * z


def target_weights(ctx: Context, t, cfg: dict):
    """Risky-sleeve weights (sum 1) and equity exposure e for date t."""
    u = ctx.universe(t)
    u = u[~u.index.isin(set(cfg.get("exclude") or []))]                 # official HSC / free-float screen
    if t in ctx.mask.index.get_level_values("date"):                    # investability + fundamentals
        allowed = ctx.mask.xs(t, level="date")
        u = u[u.index.isin(allowed[allowed.astype(bool)].index)]
    names = list(u.index)
    if not names:
        return pd.Series(dtype=float), 0.0
    sig = ctx.signal.xs(t, level="date").reindex(names) if t in ctx.signal.index.get_level_values("date") else None
    st = ctx.struct_score.xs(t, level="date").reindex(names)
    if cfg.get("K") and sig is not None:                                # keep the K best-ranked names
        names = list(sig.reindex(names).dropna().sort_values(ascending=False).index[:cfg["K"]])
    S = ctx.cov(t, names, cfg["lookback"])
    names = list(S.index)
    groups = []
    hi = st.reindex(names) >= st.quantile(2 / 3)                          # paper-cap tercile <= struct_cap
    groups.append((np.where(hi.values)[0], cfg["struct_cap"]))
    sec = pd.Series(names).map(ctx.sectors).fillna("Other").values
    for s in np.unique(sec):                                            # sector cap
        idx = np.where(sec == s)[0]
        if len(idx) * cfg["cap"] > cfg["sector_cap"]:
            groups.append((idx, cfg["sector_cap"]))
    cap = cfg["cap"] if len(names) * cfg["cap"] >= 1.0 else 1.0 / max(len(names), 1)
    vol = pd.Series(np.sqrt(np.diag(S)), index=names)
    mu = signal_to_mu(sig.reindex(names).fillna(sig.median()), vol, cfg["kappa"]) if sig is not None else None
    if mu is None:
        w = PF.min_variance(names, S, cap, groups)
    else:
        w = PF.max_sharpe(names, S, mu, cap, groups)
    w = w[w > 1e-4]
    w = w / w.sum()
    e = float(np.clip(ctx.exposure.get(t, 1.0), 0.0, 1.0))
    return w, e


def limit_bands(price, date):
    """IDX auto-rejection limits (upper, lower) as fractions on a given date.
    Upper: 35% (<= Rp200) / 25% (Rp200-5,000) / 20% (> Rp5,000). Lower: symmetric, except 7% from 13-Mar-2020 to
    3-Sep-2023, 15% from 8-Apr-2025 and 7% from 19-Oct-2026."""
    p = np.asarray(price, dtype=float)
    up = np.where(p <= 200, 0.35, np.where(p <= 5000, 0.25, 0.20))
    d = pd.Timestamp(date)
    if pd.Timestamp("2020-03-13") <= d < pd.Timestamp("2023-09-04") or d >= pd.Timestamp("2026-10-19"):
        dn = np.full_like(up, 0.07)
    elif d >= pd.Timestamp("2025-04-08"):
        dn = np.full_like(up, 0.15)
    else:
        dn = up.copy()
    return up, dn


def apply_limit_locks(tgt: pd.Series, cur: pd.Series, ret_exec: pd.Series, price: pd.Series, exec_date):
    """Upper-limit names cannot be bought and lower-limit names cannot be sold on the execution day."""
    allx = tgt.index.union(cur.index)
    t = tgt.reindex(allx).fillna(0.0)
    c = cur.reindex(allx).fillna(0.0)
    r = ret_exec.reindex(allx).fillna(0.0).values
    up, dn = limit_bands(price.reindex(allx).fillna(1000).values, exec_date)
    lock_up = r >= up - 0.005
    lock_dn = r <= -(dn - 0.005)
    t = pd.Series(np.where(lock_up & (t.values > c.values), c.values, t.values), index=allx)
    t = pd.Series(np.where(lock_dn & (t.values < c.values), c.values, t.values), index=allx)
    if t.sum() > 1.0 + 1e-9:
        buys = (t > c) & ~pd.Series(lock_dn, index=allx)
        excess = t.sum() - 1.0
        room = (t[buys] - c[buys]).sum()
        if room > 0:
            t[buys] = t[buys] - (t[buys] - c[buys]) * min(1.0, excess / room)
    return t[t > 1e-6]


def run_backtest(ctx: Context, cfg: dict, start, end):
    """Monthly net returns of the policy (and of the cash it holds), after costs."""
    dates = ctx.dates[(ctx.dates >= pd.Timestamp(start)) & (ctx.dates <= pd.Timestamp(end))]
    w_drift = pd.Series(dtype=float)
    rows = []
    for t in dates:
        u = ctx.universe(t)
        w, e = target_weights(ctx, t, cfg)
        ex = u["exec_date"].iloc[0]
        rex = ctx.r.loc[ex] if ex in ctx.r.index else pd.Series(dtype=float)
        tgt = apply_limit_locks(w * e, w_drift, rex, u["price"], ex)
        allx = tgt.index.union(w_drift.index)
        d = tgt.reindex(allx).fillna(0) - w_drift.reindex(allx).fillna(0)
        hs = u["rel_tick"].reindex(allx).fillna(0.01) / 2
        cost = float((d.clip(lower=0) * (C.BUY_FEE + hs)).sum() + ((-d).clip(lower=0) * (C.SELL_FEE + hs)).sum())
        fr = u["fwd_ret"].reindex(tgt.index).fillna(0.0)
        rf = float(ctx.mkt.loc[t, "rf_fwd"])
        gross = float((tgt * fr).sum() + (1.0 - tgt.sum()) * rf)
        net = (1 - cost) * (1 + gross) - 1
        w_drift = tgt * (1 + fr) / (1 + gross)
        rows.append({"date": t, "net": net, "cost": cost, "turnover": float(d.abs().sum()), "exposure": float(tgt.sum()),
                     "n_names": int((tgt > 1e-4).sum()), "rf": rf, "ihsg": float(ctx.mkt.loc[t, "mkt_fwd"])})
    return pd.DataFrame(rows).set_index("date")


def ce_return(r: pd.Series, gamma=C.GAMMA) -> float:
    """Annualised certainty-equivalent return of monthly returns under CRRA(gamma)."""
    x = (1 + r.dropna()).clip(lower=1e-6)
    return float(np.mean(x ** (1 - gamma)) ** (12 / (1 - gamma)) - 1)


def metrics(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    ex = r - rf.reindex(r.index)
    eq = (1 + r).cumprod()
    return {"CAGR": float(eq.iloc[-1] ** (12 / len(r)) - 1), "vol": float(r.std() * np.sqrt(12)),
            "Sharpe": float(ex.mean() / ex.std() * np.sqrt(12)), "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "CE_g5": ce_return(r, 5.0), "worst_month": float(r.min()), "months": len(r)}
