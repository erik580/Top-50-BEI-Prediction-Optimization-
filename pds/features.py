"""Point-in-time features (stock and market level) and the monthly decision panel.

Every feature at month-end t uses only data dated <= t. Forward returns run from the execution day
e(t) (next session after t: decide after the close, trade next day) to e(t+1).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

ANN = 252
STRUCT_FEATS = ["days_to_trade_cap", "n_limit_up_6m", "zero_vol_3m"]   # "paper market cap" proxies


def _roll(df, w, fn, minp=None):
    return getattr(df.rolling(w, min_periods=minp or max(5, int(w * 0.6))), fn)()


def stock_daily_features(close, adj, vol, r, mkt_r, mcap):
    """Daily matrices (dates x tickers) of stock characteristics."""
    F = {}
    lp = np.log(adj.ffill(limit=5))
    for n, w in {"mom_1m": 21, "mom_12m": 252}.items():
        F[n] = lp - lp.shift(w)
    F["mom_12_1"] = F["mom_12m"] - F["mom_1m"]
    F["vol_3m"] = _roll(r, 63, "std") * np.sqrt(ANN)
    hi = adj.ffill(limit=5).rolling(252, min_periods=60).max()
    F["dist_52w_high"] = adj.ffill(limit=5) / hi - 1                      # the V2 ranking signal
    m = mkt_r.reindex(r.index)
    beta = r.rolling(252, min_periods=120).cov(m).div(m.rolling(252, min_periods=120).var(), axis=0)
    F["beta_12m"] = beta
    adv = (close * vol).where(vol > 0).rolling(63, min_periods=20).mean()
    F["zero_vol_3m"] = (vol == 0).astype(float).where(close.notna()).rolling(63, min_periods=20).mean()
    F["days_to_trade_cap"] = np.log(mcap.reindex(columns=close.columns) / adv)
    up = (r > 0.15).astype(float).where(r.notna())
    F["n_limit_up_6m"] = up.rolling(126, min_periods=40).sum()
    F["rel_tick"] = pd.DataFrame(C.tick_size(close.values), index=close.index, columns=close.columns) / close
    return F


def market_daily_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Market-level state used by the reports and the exposure decision."""
    m = macro.ffill(limit=5)
    jr = m["JKSE"].pct_change(fill_method=None)
    out = pd.DataFrame(index=m.index)
    out["JKSE_r3m"] = np.log(m["JKSE"]) - np.log(m["JKSE"]).shift(63)
    out["JKSE_vol1m"] = jr.rolling(21).std() * np.sqrt(ANN)
    out["JKSE_vol3m"] = jr.rolling(63).std() * np.sqrt(ANN)
    out["JKSE_dd_12m"] = m["JKSE"] / m["JKSE"].rolling(252, min_periods=60).max() - 1
    return out


def point_in_time_universe(mcap, close, vol, n=C.UNIVERSE_SIZE):
    """Top-n by market cap among stocks that traded in the last 10 sessions and have >= 120 days of history."""
    hist = close.notna().cumsum()
    active = (vol.rolling(10, min_periods=1).sum() > 0) & close.notna()
    elig = (hist >= C.MIN_HISTORY_DAYS) & active
    rank = mcap.reindex(columns=close.columns).where(elig).rank(axis=1, ascending=False)
    return rank <= n


def monthly_panel(SF, MF, top, mcap, close, adj, jkse, keep, rf_daily, start="2011-01-01", end="2026-09-30"):
    """Long panel (date, ticker) of features and forward returns at every month-end, plus the market table."""
    me = pd.DatetimeIndex([d for d in pd.Series(close.index, index=close.index).groupby(close.index.to_period("M")).last()])
    me = me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]
    idx = close.index
    exec_dates = idx[np.minimum(idx.get_indexer(me) + 1, len(idx) - 1)]
    lvl = adj.ffill(limit=5)
    rows = []
    for k, t in enumerate(me):
        e0 = exec_dates[k]
        e1 = exec_dates[k + 1] if k + 1 < len(me) else idx[-1]
        d = pd.DataFrame({name: v.loc[t] for name, v in SF.items()})
        d["in_top50"] = top.loc[t]
        d["mcap"] = mcap.reindex(columns=close.columns).loc[t]
        d["price"] = close.loc[t]
        d["fwd_ret"] = lvl.loc[e1] / lvl.loc[e0] - 1
        d["exec_date"] = e0
        d["date"] = t
        rows.append(d.reset_index().rename(columns={"index": "ticker", "Ticker": "ticker"}))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["in_top50"] | panel["ticker"].isin(keep)]
    panel = panel.set_index(["date", "ticker"]).sort_index()
    mkt = MF.reindex(me)
    mkt.index.name = "date"
    lvl_rf = (1 + rf_daily).cumprod()
    nxt = [exec_dates[k + 1] if k + 1 < len(me) else idx[-1] for k in range(len(me))]
    mkt["rf_fwd"] = [lvl_rf.loc[b] / lvl_rf.loc[a] - 1 for a, b in zip(exec_dates, nxt)]
    jk = jkse.reindex(idx).ffill(limit=5)
    mkt["mkt_fwd"] = [jk.loc[b] / jk.loc[a] - 1 for a, b in zip(exec_dates, nxt)]
    mkt["exec_date"] = exec_dates
    return panel, mkt


def cross_section(SF, top, mcap, close, t, exec_date):
    """Feature row of the point-in-time top-50 on a non-month-end date (used for the live decision)."""
    d = pd.DataFrame({name: v.loc[t] for name, v in SF.items()})
    d["in_top50"] = top.loc[t]
    d["mcap"] = mcap.reindex(columns=close.columns).loc[t]
    d["price"] = close.loc[t]
    d["fwd_ret"] = np.nan
    d["exec_date"] = exec_date
    d["date"] = t
    d = d.reset_index().rename(columns={"index": "ticker", "Ticker": "ticker"}).set_index(["date", "ticker"])
    return d[d["in_top50"]]


def xs_rank(panel: pd.DataFrame, col: str) -> pd.Series:
    """Cross-sectional percentile rank in [-0.5, 0.5] within the top-50 of each date."""
    p = panel[panel["in_top50"]]
    return p[col].groupby(level="date").rank(pct=True).sub(0.5).fillna(0.0)
