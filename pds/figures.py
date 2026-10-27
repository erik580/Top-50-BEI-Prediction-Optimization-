"""Figures of the V2 notebook (matplotlib, colour-blind-safe palette from viz.py)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import viz as V
from . import config as C


def ihsg_events(jkse: pd.Series, start="2024-01-01", end=C.EVAL_END):
    jk = jkse.dropna().loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(jk.index, jk.values, color=V.SERIES[0], lw=1.6)
    dec = pd.Timestamp(C.DECISION_DATE)
    ax.axvspan(dec, jk.index[-1], color=V.SERIES[1], alpha=0.08, lw=0)
    ax.axvline(dec, color=V.SERIES[1], lw=1.2, ls="--")
    ax.set_ylim(4300, jk.max() * 1.06)
    ax.text(dec, jk.max() * 1.05, "decision date 31-Aug-2026  ", color=V.INK2, fontsize=8.5, va="top", ha="right")
    picks = [("2025-04-08", "US tariffs -7.9%", (0, -30)), ("2026-01-20", "peak 9,135", (-62, 0)),
             ("2026-01-28", "MSCI freeze -7.4%", (70, 8)), ("2026-05-13", "MSCI deletes BREN, AMMN, TPIA...", (-95, -26)),
             ("2026-06-08", "rupiah > 18,000", (-70, -14)), ("2026-10-16", "MSCI downgrade consultation -11.6%", (-110, -30))]
    box = dict(boxstyle="round,pad=0.2", fc=V.SURF, ec="none", alpha=0.9)
    for d, lab, off in picks:
        d = pd.Timestamp(d)
        if d not in jk.index:
            continue
        y = jk.loc[d]
        ax.annotate(lab, (d, y), xytext=off, textcoords="offset points", ha="center", va="center", fontsize=8, color=V.INK2,
                    bbox=box, arrowprops=dict(arrowstyle="-", color=V.MUTED, lw=0.7, shrinkA=0, shrinkB=3))
        ax.plot([d], [y], "o", color=V.SURF, mec=V.SERIES[0], mew=1.5, ms=5, zorder=5)
    ax.set_title("IHSG (Jakarta Composite), Jan-2024 to 16-Oct-2026")
    ax.set_ylabel("index level")
    fig.tight_layout()
    return fig


def headline_vs_float(t50: pd.DataFrame, n=15):
    d = t50.sort_values("mcap_T", ascending=False).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["mcap_T"], height=0.38, color=V.SERIES[0], label="headline market cap")
    ax.barh(y - 0.2, d["eff_float_cap_T"], height=0.38, color=V.SERIES[1], label="effective free-float cap")
    ax.set_yticks(y, [f"{t}{' *' if h else ''}" for t, h in zip(d.index, d["hsc"])])
    ax.set_xlabel("Rp trillion, 31-Aug-2026")
    ax.set_title(f"Top-{n} by headline market cap vs the part that actually floats  (* = HSC list)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def timing_panel(jkse: pd.Series, e: pd.Series, start="2015-01-01", end=C.DECISION_DATE):
    jk = jkse.dropna().loc[start:end]
    e = e.loc[start:end]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    a1.plot(jk.index, jk.values, color=V.SERIES[0], lw=1.4)
    a1.set_title("IHSG level (top) and the learned equity exposure (bottom, rest in cash)")
    a1.set_ylabel("IHSG")
    a2.step(e.index, e.values, where="post", color=V.SERIES[1], lw=1.6)
    a2.fill_between(e.index, 0, e.values, step="post", color=V.SERIES[1], alpha=0.12, lw=0)
    a2.set_ylim(0, 1.02); V.pct_axis(a2, "y")
    a2.set_ylabel("equity exposure")
    a2.axvline(pd.Timestamp(C.DECISION_DATE), color=V.MUTED, lw=1, ls="--")
    fig.tight_layout()
    return fig


def forecast_fan(jk: pd.Series, fc: dict, t0: str, t1: str):
    """IHSG history + locked forecast quantile bands for the horizon (straight-line interpolation in time)."""
    jk = jk.dropna().loc[:t0].iloc[-130:]
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(jk.index, jk.values, color=V.SERIES[0], lw=1.6, label="IHSG (actual)")
    x = [t0, t1]; now = fc["IHSG_now"]
    for lo, hi, a in [("q5", "q95", 0.12), ("q25", "q75", 0.22)]:
        ax.fill_between(x, [now, fc[f"IHSG_level_{lo}"]], [now, fc[f"IHSG_level_{hi}"]], color=V.SERIES[1], alpha=a, lw=0,
                        label=f"forecast {lo[1:]}-{hi[1:]}% band")
    ax.plot(x, [now, fc["IHSG_level_q50"]], color=V.SERIES[1], lw=1.8, ls="--", label="forecast median")
    for q in ("q5", "q50", "q95"):
        ax.annotate(f"{fc[f'IHSG_level_{q}']:,.0f}", (t1, fc[f"IHSG_level_{q}"]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=V.INK2)
    ax.axvline(t0, color=V.MUTED, lw=1, ls=":")
    ax.set_title(f"IHSG forecast locked on {t0.date()} for {t1.date()} (cross-country conditional distribution)")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return fig


def holdout_paths(paths: pd.DataFrame, keep: list[str]):
    fig, ax = plt.subplots(figsize=(11, 5))
    ends = []
    for i, k in enumerate(keep):
        if k not in paths:
            continue
        s = paths[k].dropna() - 1
        ax.plot(s.index, s.values, color=V.SERIES[i], lw=2.2 if i == 0 else 1.6, label=k)
        ends.append([s.values[-1], s.index[-1]])
    # end labels, nudged apart so close values do not overlap
    ends.sort(key=lambda x: x[0])
    lo, hi = ax.get_ylim(); gap = 0.035 * (hi - lo); pos = []
    for v, _ in ends:
        pos.append(max(v, pos[-1] + gap) if pos else v)
    for (v, x), y in zip(ends, pos):
        ax.annotate(f"{v:+.1%}", (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=V.INK2)
    ax.axhline(0, color=V.AXIS, lw=1)
    ax.axvline(pd.Timestamp("2026-10-16"), color=V.MUTED, lw=1, ls=":")
    ax.text(pd.Timestamp("2026-10-16"), ax.get_ylim()[1], "16-Oct MSCI\nconsultation ", ha="right", va="top", fontsize=8, color=V.INK2)
    V.pct_axis(ax, "y")
    ax.set_title("Reality check: cumulative return from 1-Sep-2026 (execution) to 16-Oct-2026")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return fig


def equity_curves(curves: pd.DataFrame, title: str):
    """Growth of Rp1 for the V2 policy and its benchmarks (log scale)."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, k in enumerate(curves.columns):
        s = curves[k].dropna()
        ax.plot(s.index, s.values, color=V.SERIES[i], lw=2.2 if i == 0 else 1.5, label=k)
        V.label_end(ax, s.index[-1], s.values[-1], f"{s.values[-1]:.2f}")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylabel("value of Rp1 (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig
