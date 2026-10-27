"""Settings of the V2 decision pipeline: paths, timeline, IDX trading rules and the investor."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
LOCKED = ROOT / "locked"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- timeline
DECISION_DATE = "2026-08-31"       # last trading day of the assignment month (August 2026)
EVAL_END = "2026-10-16"            # last complete close before the live decision
LIVE_DATE = "2026-10-19"           # live decision (close), executed at the 20-Oct close
LIVE_END = "2026-11-30"            # live forecast horizon
BACKTEST_START = "2015-01-31"
BACKTEST_END = "2026-06-30"        # last rebalance whose holding month is complete before the decision
SELECTION_END = "2023-12-29"       # selection window 2015-2023
TEST_START = "2024-01-31"          # test window 2024-Jun 2026

# ----------------------------------------------------------------------------- IDX market rules
LOT = 100                          # shares per lot
BUY_FEE = 0.0015                   # retail broker buy fee incl. levies
SELL_FEE = 0.0025                  # sell fee incl. 0.1% final tax
UNIVERSE_SIZE = 50
MIN_HISTORY_DAYS = 120

# ----------------------------------------------------------------------------- investor
CAPITAL = 100_000_000              # hypothetical retail investor, Rp100 million
GAMMA = 5.0                        # CRRA risk aversion (moderate)


def tick_size(price):
    """IDX tick size (Rp) by price band."""
    p = np.asarray(price, dtype=float)
    return np.select([p < 200, p < 500, p < 2000, p < 5000], [1, 2, 5, 10], default=25)
