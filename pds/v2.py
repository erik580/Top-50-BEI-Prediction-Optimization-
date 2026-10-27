"""V2 first-principles screens: investability -> fundamentals (micro) -> macro sensitivity (second order).

First principles for a retail investor
  1. Avoid permanent loss before seeking return: only buy what trades at a real price (no stale or floor-pinned
     quotes, enough liquidity), from companies that make money and are not over-levered.
  2. Long-run return = earnings yield + growth + re-rating, so profitability and valuation matter.
  3. Volatility costs a utility-maximising investor money: prefer lower risk at equal expected return.
Second order: macro shocks (rupiah, oil, coal, gold, copper, global rates, foreign flows) reach each stock through
  its measured sensitivity to them.
Third order: forced flows (MSCI passive outflows, the free-float rule) -> official screen + Monte Carlo jump multipliers.
Fundamentals become usable only after a publication lag (75 days interim, 95 days audited annual).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

MIN_PRICE_FLOOR = 50            # IDX minimum price on the regular board until 28-Sep-2026
MIN_ADV = 5e9                   # median daily value traded (Rp) over 63 sessions
MAX_STALE = 0.20                # max share of zero-return sessions over 63 sessions
MAX_DEBT_TO_EQUITY = 3.0        # non-financial companies
MACRO_FACTORS = ["USDIDR", "BRENT", "COAL_ROTT", "GOLD", "COPPER", "US10Y", "EIDO_GAP"]


# ----------------------------------------------------------------------------- investability
def investability(close: pd.DataFrame, vol: pd.DataFrame, r: pd.DataFrame, dates) -> pd.DataFrame:
    """Boolean mask (date, ticker): tradable at a real price with enough liquidity."""
    val = (close * vol)
    adv = val.rolling(63, min_periods=40).median()
    stale = (r == 0).astype(float).where(r.notna()).rolling(63, min_periods=40).mean()
    floor = close <= MIN_PRICE_FLOOR
    ok = (adv >= MIN_ADV) & (stale < MAX_STALE) & ~floor & close.notna()
    m = ok.reindex(dates).stack()
    m.index.names = ["date", "ticker"]
    feats = pd.DataFrame({"adv_med_63": adv.reindex(dates).stack(), "stale_63": stale.reindex(dates).stack(),
                          "at_floor": floor.reindex(dates).stack()})
    feats.index.names = ["date", "ticker"]
    return m.rename("investable"), feats


# ----------------------------------------------------------------------------- fundamentals
def _lag(pe: pd.Timestamp) -> pd.Timestamp:
    return pe + pd.Timedelta(days=95 if pe.month == 12 else 75)     # audited annual vs interim reports


def fundamentals_panel(dates, mcap: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    f = pd.read_parquet(C.DATA / "fundamentals_quarterly.parquet")
    f = f[f["period_end"].notna()].copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    keep = ["net_income_common", "net_income", "revenue", "common_equity", "equity_incl_minority", "total_assets", "total_debt", "operating_cf"]
    f = f[f["item"].isin(keep)]
    cur = f.groupby("ticker")["currency"].agg(lambda s: s[s.isin(["IDR", "USD"])].mode().iloc[0] if s.isin(["IDR", "USD"]).any() else "IDR")
    q = f.pivot_table(index=["ticker", "period_end"], columns="item", values="value", aggfunc="first").sort_index()
    q["ni"] = q.get("net_income_common", pd.Series(dtype=float)).fillna(q.get("net_income"))
    q["eq"] = q.get("common_equity", pd.Series(dtype=float)).fillna(q.get("equity_incl_minority"))
    rows = []
    for t, g in q.groupby(level="ticker"):
        g = g.droplevel(0).sort_index()
        g = g[~g.index.duplicated()]
        avail = pd.Series([_lag(p) for p in g.index], index=g.index)
        ni4 = g["ni"].rolling(4, min_periods=4).sum()
        ni4_prev = ni4.shift(4)
        rev4 = g.get("revenue", pd.Series(np.nan, index=g.index)).rolling(4, min_periods=4).sum()
        eq_avg = g["eq"].rolling(4, min_periods=2).mean()
        prof8 = (g["ni"] > 0).astype(float).rolling(8, min_periods=4).mean()
        ta = g.get("total_assets", pd.Series(np.nan, index=g.index))
        debt = g.get("total_debt", pd.Series(np.nan, index=g.index))
        ocf4 = g.get("operating_cf", pd.Series(np.nan, index=g.index)).rolling(4, min_periods=4).sum()
        feat = pd.DataFrame({"avail": avail, "ttm_ni": ni4, "roe": ni4 / eq_avg.where(eq_avg > 0),
                             "roa": ni4 / ta.where(ta > 0), "ni_growth": (ni4 - ni4_prev) / g["eq"].abs().where(g["eq"].abs() > 0),
                             "profit_frac_8q": prof8, "equity": g["eq"], "debt_to_equity": debt / g["eq"].where(g["eq"] > 0),
                             "asset_growth": ta / ta.shift(4) - 1, "accruals": (ni4 - ocf4) / ta.where(ta > 0), "rev_ttm": rev4})
        feat["ticker"] = t
        rows.append(feat)
    F_ = pd.concat(rows).reset_index(drop=True).sort_values("avail")
    out = []
    for d in dates:
        latest = F_[F_["avail"] <= d].groupby("ticker").tail(1).set_index("ticker")
        latest = latest[latest["avail"] >= d - pd.Timedelta(days=200)]          # stale filings are not used
        if latest.empty:
            continue
        x = latest.drop(columns=["avail"]).copy()
        m = mcap.loc[d].reindex(x.index) if d in mcap.index else pd.Series(np.nan, index=x.index)
        conv = np.where(cur.reindex(x.index).fillna("IDR").values == "USD", float(fx.loc[:d].iloc[-1]), 1.0)
        x["ep"] = x["ttm_ni"] * conv / m
        x["bp"] = x["equity"] * conv / m
        x["date"] = d
        out.append(x.reset_index())
    P = pd.concat(out).set_index(["date", "ticker"]).sort_index()
    return P


# ----------------------------------------------------------------------------- macro, second order
def macro_factor_returns(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.ffill(limit=5)
    w = m.resample("W-FRI").last()
    fr = pd.DataFrame({"MKT": np.log(w["JKSE"]).diff(), "USDIDR": np.log(w["USDIDR"]).diff(), "BRENT": np.log(w["BRENT"]).diff(),
                       "COAL_ROTT": np.log(w["COAL_ROTT"]).diff(), "GOLD": np.log(w["GOLD"]).diff(), "COPPER": np.log(w["COPPER"]).diff(),
                       "US10Y": w["US10Y"].diff() / 10.0,
                       "EIDO_GAP": np.log(w["EIDO"]).diff() - (np.log(w["JKSE"]).diff() - np.log(w["USDIDR"]).diff())})
    return fr


def macro_exposures(adj: pd.DataFrame, macro: pd.DataFrame, d, names, window=156, trend_weeks=13, ridge=1.0):
    """Second-order check at date d (data <= d only).
    beta: ridge betas of each stock's weekly market-relative return on standardised macro factor returns (3 years);
    move: standardised macro move of the last quarter; tilt = beta . move (positive = macro tailwind)."""
    fr = macro_factor_returns(macro).dropna(how="all")
    wk = np.log(adj.ffill(limit=5).resample("W-FRI").last()).diff()
    rel = wk.sub(fr["MKT"], axis=0)
    F_ = fr.loc[:d].tail(window)[MACRO_FACTORS].fillna(0.0)
    mu, sd = F_.mean(), F_.std() + 1e-12
    Z = ((F_ - mu) / sd).values
    trend = fr.loc[:d, MACRO_FACTORS].tail(trend_weeks).sum()
    move = (trend / (sd * np.sqrt(trend_weeks))).clip(-3, 3)
    betas = {}
    for n in [n for n in names if n in rel.columns]:
        y = rel.loc[F_.index, n].values
        ok = np.isfinite(y)
        if ok.sum() < 60:
            continue
        A = Z[ok].T @ Z[ok] + ridge * ok.sum() * np.eye(Z.shape[1])
        betas[n] = np.linalg.solve(A, Z[ok].T @ (y[ok] - y[ok].mean()))
    beta = pd.DataFrame(betas, index=MACRO_FACTORS).T
    return beta, move, (beta @ move.values).rename("macro_tilt")


# ----------------------------------------------------------------------------- micro screen
def fundamental_screen(f: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """True = profitable (TTM), positive equity, debt/equity <= 3 (non-financials). Names without a filing pass."""
    tick = f.index.get_level_values("ticker") if isinstance(f.index, pd.MultiIndex) else f.index
    lev = f["debt_to_equity"].where(~pd.Index(tick).map(sectors).isin(["Financials"]))
    ok = (f["ttm_ni"] > 0) & (f["equity"] > 0) & ~(lev > MAX_DEBT_TO_EQUITY)
    return ok | f["roe"].isna()
