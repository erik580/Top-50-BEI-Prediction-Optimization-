"""Cash/equity exposure learned from a pooled cross-country history (17 equity markets since 1995).

Indonesia alone has few crashes; pooling markets gives hundreds of stress months (Asian crisis 1997-98, 2008,
euro crisis, China 2015, Turkey/Argentina 2018, COVID 2020, ...). Returns are standardised by each market's own
volatility, so markets are comparable. A logistic model estimates the probability that next month is a crash
(standardised return below -1); historical scenarios are re-weighted by that probability and rescaled to today's
Indonesian volatility, and the prescriptive step picks the equity share that maximises expected CRRA utility.
(Random-forest, nearest-neighbour and LightGBM versions of this step were also tried; the pooled logit was kept.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

COUNTRIES = {"IDN": ("^JKSE", "IDR=X"), "KOR": ("^KS11", "KRW=X"), "HKG": ("^HSI", None), "IND": ("^NSEI", "INR=X"),
             "SGP": ("^STI", "SGD=X"), "MYS": ("^KLSE", "MYR=X"), "TWN": ("^TWII", "TWD=X"), "BRA": ("^BVSP", "BRL=X"),
             "MEX": ("^MXX", "MXN=X"), "TUR": ("XU100.IS", "TRY=X"), "ARG": ("^MERV", "ARS=X"), "GRC": ("GD.AT", "EUR=X"),
             "ZAF": ("^J203.JO", "ZAR=X"), "CHN": ("000001.SS", "CNY=X"), "JPN": ("^N225", "JPY=X"), "USA": ("^GSPC", None),
             "DEU": ("^GDAXI", "EUR=X")}
FEATS = ["r1", "r3", "r6", "r12", "vol1", "vol3", "vol_ratio", "dd12", "min1d", "skew3", "above_ma", "fx_r1", "fx_r3",
         "fx_vol1", "g_vix", "g_vix_chg", "g_us10y_chg", "g_dxy_r1", "g_spx_r1", "g_spx_dd12", "rel_r1", "rel_r3", "em_breadth"]


def _country_feats(px: pd.Series, fx: pd.Series | None):
    p = px.dropna()
    p = p[p > 0]
    r = p.pct_change(fill_method=None).clip(-0.5, 0.5)
    me = p.resample("ME").last().index
    d = pd.DataFrame(index=me)
    lvl = p.resample("ME").last()
    for n, k in {"r1": 1, "r3": 3, "r6": 6, "r12": 12}.items():
        d[n] = np.log(lvl / lvl.shift(k))
    d["vol1"] = r.rolling(21, min_periods=15).std().resample("ME").last() * np.sqrt(252)
    d["vol3"] = r.rolling(63, min_periods=40).std().resample("ME").last() * np.sqrt(252)
    d["vol_ratio"] = d.vol1 / d.vol3
    d["dd12"] = (p / p.rolling(252, min_periods=120).max() - 1).resample("ME").last()
    d["min1d"] = r.rolling(21, min_periods=15).min().resample("ME").last()
    d["skew3"] = r.rolling(63, min_periods=40).skew().resample("ME").last()
    d["above_ma"] = (p > p.rolling(200, min_periods=120).mean()).astype(float).resample("ME").last()
    if fx is not None and fx.notna().sum() > 500:
        f = fx.dropna(); f = f[f > 0]
        fl = f.resample("ME").last()
        d["fx_r1"] = np.log(fl / fl.shift(1)).reindex(me)
        d["fx_r3"] = np.log(fl / fl.shift(3)).reindex(me)
        d["fx_vol1"] = (f.pct_change(fill_method=None).rolling(21, min_periods=15).std() * np.sqrt(252)).resample("ME").last().reindex(me)
    else:
        d["fx_r1"] = d["fx_r3"] = d["fx_vol1"] = 0.0
    d["y"] = lvl.shift(-1) / lvl - 1                      # next-month local-currency return
    d["y_z"] = d["y"] / (d["vol3"] * np.sqrt(1 / 12))     # vol-standardised
    return d


def build_country_panel(px: pd.DataFrame) -> pd.DataFrame:
    """Monthly (date, country) panel of vol-standardised features and next-month returns."""
    g = pd.DataFrame(index=px.resample("ME").last().index)
    vix, tnx, dxy, spx = px["^VIX"].dropna(), px["^TNX"].dropna(), px["DX-Y.NYB"].dropna(), px["^GSPC"].dropna()
    g["g_vix"] = vix.resample("ME").last()
    g["g_vix_chg"] = g["g_vix"].diff()
    g["g_us10y_chg"] = tnx.resample("ME").last().diff(3)
    g["g_dxy_r1"] = np.log(dxy.resample("ME").last()).diff()
    g["g_spx_r1"] = np.log(spx.resample("ME").last()).diff()
    g["g_spx_dd12"] = (spx / spx.rolling(252, min_periods=120).max() - 1).resample("ME").last()
    rows = []
    for c, (ix, fx) in COUNTRIES.items():
        if ix not in px or px[ix].notna().sum() < 1000:
            continue
        d = _country_feats(px[ix], px[fx] if fx and fx in px else None)
        d["country"] = c
        rows.append(d)
    P = pd.concat(rows)
    P.index.name = "date"
    P = P.reset_index().set_index(["date", "country"]).sort_index()
    gg = g.reindex(P.index.get_level_values("date"))
    for k in gg.columns:
        P[k] = gg[k].values
    # cross-country context: breadth of EM markets with positive 1m return, and relative momentum
    em = [c for c in COUNTRIES if c not in ("USA", "JPN", "DEU")]
    r1 = P["r1"].unstack("country")
    P["em_breadth"] = (r1[[c for c in em if c in r1]] > 0).mean(axis=1).reindex(P.index.get_level_values("date")).values
    P["rel_r1"] = P["r1"] - P["g_spx_r1"]
    r3 = P["r3"].unstack("country")
    P["rel_r3"] = (P["r3"].values - r3.mean(axis=1).reindex(P.index.get_level_values("date")).values)
    return P


def crra_best_exposure(scen: np.ndarray, w: np.ndarray, rf: float, gamma: float, grid=np.linspace(0, 1, 21)):
    """argmax_e sum_j w_j U((1-e) rf + e r_j) with CRRA utility."""
    best, bu = 1.0, -np.inf
    for e in grid:
        x = np.clip(1 + (1 - e) * rf + e * scen, 1e-4, None)
        u = np.sum(w * (x ** (1 - gamma) - 1) / (1 - gamma))
        if u > bu:
            bu, best = u, e
    return best


def _fit_logit(Xt, yt):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    model = LogisticRegression(C=0.1, max_iter=2000).fit((Xt - mu) / sd, (yt < -1.0).astype(int))
    return model, mu, sd


def scenario_weights(p_crash: float, base: np.ndarray) -> np.ndarray:
    """Re-weight historical standardised outcomes so that crash months carry probability p_crash."""
    crash = base < -1.0
    w = np.where(crash, p_crash / max(crash.mean(), 1e-9), (1 - p_crash) / max(1 - crash.mean(), 1e-9))
    return w / w.sum()


def walk_forward_exposure(P: pd.DataFrame, target="IDN", start="2012-01-31", gamma=5.0, refit=12, rf_annual=0.045):
    """Monthly equity share in [0, 1] (and the crash probability behind it) using only information available at t."""
    X = P[FEATS].fillna(0.0)
    y = P["y_z"]
    dates = P.index.get_level_values("date").unique().sort_values()
    tdates = dates[dates >= pd.Timestamp(start)]
    out, fit = {}, None
    for i, t in enumerate(tdates):
        if (t, target) not in P.index:
            continue
        tr = (P.index.get_level_values("date") < t) & y.notna().values          # label of t-1 is known at t
        if fit is None or i % refit == 0:
            fit = (*_fit_logit(X[tr].values, y[tr].values), y[tr].values)
        model, mu, sd, base = fit
        vol_now = float(P.loc[(t, target), "vol3"])
        if not np.isfinite(vol_now):
            continue
        pc = model.predict_proba((X.loc[(t, target)].values[None, :] - mu) / sd)[0, 1]
        scen = base * vol_now * np.sqrt(1 / 12)
        out[t] = {"exposure": crra_best_exposure(scen, scenario_weights(pc, base), rf_annual / 12, gamma),
                  "p_crash": pc, "vol3": vol_now, "base_rate": float((base < -1.0).mean())}
    return pd.DataFrame(out).T


def crash_model_at(P: pd.DataFrame, query, train_end, target="IDN"):
    """Fit on labels known by train_end; return crash probability, scenario base and weights for the query row."""
    X = P[FEATS].fillna(0.0)
    d = P.index.get_level_values("date")
    tr = (d <= pd.Timestamp(train_end)) & P["y_z"].notna().values
    mu, sd = X[tr].mean(), X[tr].std() + 1e-9
    Z = (X - mu) / sd
    model = LogisticRegression(C=0.1, max_iter=2000).fit(Z[tr].values, (P.loc[tr, "y_z"] < -1.0).astype(int).values)
    p = float(model.predict_proba(Z.loc[[query]].values)[0, 1])
    base = P.loc[tr, "y_z"].values
    dist = np.sqrt(((Z[tr] - Z.loc[query]) ** 2).sum(axis=1))
    return {"p_crash": p, "base": base, "weights": scenario_weights(p, base), "base_rate": float((base < -1.0).mean()),
            "vol3": float(P.loc[query, "vol3"]), "distance": dist}
