"""Public data inputs: IDX prices, macro and global markets, shares, ownership, fundamentals, BI-Rate."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

# isolated bad Yahoo prints that revert the next day
BAD_DAYS = [("ANTM", "2010-01-18"), ("ANTM", "2010-01-26"), ("ANTM", "2012-09-24")]


def load_prices() -> dict[str, pd.DataFrame]:
    """Daily close (split-adjusted), adjusted close (total return) and volume of 500 IDX stocks."""
    px = pd.read_parquet(C.DATA / "idx_prices_daily.parquet")
    out = {k: px[k].sort_index() for k in ("close", "adj", "volume")}
    for t, d in BAD_DAYS:
        if t in out["close"] and pd.Timestamp(d) in out["close"].index:
            out["close"].loc[pd.Timestamp(d), t] = np.nan
            out["adj"].loc[pd.Timestamp(d), t] = np.nan
    return out


def daily_returns(adj: pd.DataFrame) -> pd.DataFrame:
    """Total-return daily returns; short gaps forward-filled; moves beyond any IDX limit treated as data errors."""
    r = adj.ffill(limit=5).pct_change(fill_method=None)
    r = r.where(adj.notna() | adj.ffill(limit=5).notna())
    r[(r > 0.75) | (r < -0.6)] = np.nan
    return r


def load_macro() -> pd.DataFrame:
    """IHSG, LQ45, USD/IDR, EIDO, EEM, VIX, US 10Y, DXY, Brent, gold, copper, coal, regional indices (+ EIDO volume)."""
    return pd.read_csv(C.DATA / "macro_daily.csv", index_col=0, parse_dates=True)


def load_intl() -> pd.DataFrame:
    """17 national equity indices, their FX rates and global risk series, 1995-2026."""
    return pd.read_parquet(C.DATA / "intl_indices_daily.parquet")


def load_shares() -> pd.Series:
    return pd.read_csv(C.DATA / "shares_outstanding.csv").set_index("ticker")["shares"]


def load_sectors() -> pd.Series:
    return pd.read_csv(C.DATA / "sectors.csv").set_index("ticker")["sector"]


def load_ownership() -> pd.DataFrame:
    return pd.read_csv(C.DATA / "ownership_freefloat.csv").set_index("ticker")


def market_cap(close: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """Point-in-time market cap = shares outstanding x close."""
    cols = close.columns.intersection(shares.index)
    return close[cols] * shares[cols]


def month_ends(index: pd.DatetimeIndex, start=None, end=None) -> pd.DatetimeIndex:
    """Last trading day of every calendar month."""
    s = pd.Series(index, index=index)
    me = pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)
    if start is not None:
        me = me[me >= pd.Timestamp(start)]
    if end is not None:
        me = me[me <= pd.Timestamp(end)]
    return me


def risk_free_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily cash return: BI-Rate minus 1 percentage point (money-market fund proxy), act/365."""
    b = pd.read_csv(C.DATA / "bi_rate_history.csv", parse_dates=["date"])
    b = b.dropna(subset=["date", "rate"]).sort_values("date").drop_duplicates("date", keep="last").set_index("date")["rate"].astype(float)
    rate = b.reindex(b.index.union(index)).ffill().reindex(index).bfill()
    ann = (rate - 1.0).clip(lower=0.0) / 100.0
    dt = pd.Series(index, index=index).diff().dt.days.fillna(1).clip(upper=5)
    return (1 + ann) ** (dt / 365.0) - 1
