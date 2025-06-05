#!/usr/bin/env python
"""
run_ewma_xyz.py  – nightly EWMA mean-shift detector + plot for one symbol

usage examples
--------------
# default JPM, λ = 0.2, L = 3, warm-up 180 d, plot last 250 d
python run_ewma_xyz.py

# run MSFT with a tighter control limit and 400-day plot tail
python run_ewma_xyz.py --symbol MSFT --lambda 0.1 --L 2.7 --plot-window 400
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import psycopg2, argparse, logging, os, pathlib, pickle
from dotenv import load_dotenv
from typing import Tuple

# ────────────────────────────────────────────── CLI ───────────────────────────────────────────── #

load_dotenv()
cli = argparse.ArgumentParser()
cli.add_argument("--symbol",       default="JPM")
cli.add_argument("--lookback",     type=int,   default=365, help="days of history on 1st run")
cli.add_argument("--lambda_",      dest="lam", type=float, default=0.2, help="EWMA smoothing λ")
cli.add_argument("--L",            type=float, default=3.0, help="control-limit multiple")
cli.add_argument("--warm",         type=int,   default=180, help="warm-up window (days)")
cli.add_argument("--refractory",   type=int,   default=5,   help="days to skip after detection")
cli.add_argument("--plot-window",  type=int,   default=250, help="tail to plot")
cli.add_argument("--dry-run",      action="store_true", help="run without persisting state")
args = cli.parse_args()

# ────────────────────────────────────── DB + paths + logging ─────────────────────────────────── #

DB_KWARGS = dict(
    host     = os.getenv("DB_HOST"),
    port     = int(os.getenv("DB_PORT")),
    dbname   = os.getenv("DB_NAME"),
    user     = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
)

SCRIPT_DIR          = pathlib.Path(__file__).resolve().parent
SYMBOL              = args.symbol.upper()
PRICE_TABLE         = "stock_data.daily_prices"
STATE_PATH          = SCRIPT_DIR / f"ewma_state_{SYMBOL}.pkl"
LOGFILE             = SCRIPT_DIR / f"ewma_nightly_{SYMBOL}.log"

LAM                 = args.lam            # EWMA λ
L_MULT              = args.L             # control limit multiple (≈3 ↔ 99.7 % in-control)
WARM_WINDOW_DAYS    = args.warm
REFRACTORY_DAYS     = args.refractory
PLOT_WINDOW_DAYS    = args.plot_window

logging.basicConfig(
    filename=LOGFILE,
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(f"ewma_{SYMBOL.lower()}")

# ────────────────────────────────────── EWMA detector class ──────────────────────────────────── #

class EWMADetector:
    """
    One-sided or two-sided EWMA mean-shift detector (two-sided by default).
    """
    def __init__(self, mu0, sigma0, lam=0.2, L=3.0, *, two_sided=True):
        self.lam      = lam
        self.L        = L
        self.two_sided= two_sided
        self.set_baseline(mu0, sigma0)

    def reset(self):
        self.z        = self.mu0      # start EWMA at baseline mean
        self.t        = -1            # sample counter for debugging

    # ───────── baseline (μ, σ) update ─────────
    def set_baseline(self, mu0: float, sigma0: float):
        self.mu0      = mu0
        self.sigma0   = max(sigma0, 1e-8)
        # pre-compute std(EWMA) factor for speed ▶ s = σ·√( λ/(2–λ) )
        self.s_e      = self.sigma0 * np.sqrt(self.lam / (2.0 - self.lam))
        self.reset()

    # ───────── streaming step ─────────
    def step(self, x: float):
        self.t += 1
        self.z  = self.lam * x + (1.0 - self.lam) * self.z
        # control limits
        up  = self.z - self.mu0 >  self.L * self.s_e
        dn  = self.mu0 - self.z >  self.L * self.s_e if self.two_sided else False
        if up or dn:
            dev = (self.z - self.mu0) / self.s_e   # signed deviation
            self.reset()
            return True, np.sign(dev), float(dev)
        return False, None, (self.z - self.mu0)/self.s_e

# ────────────────────────────────────── DB helpers ───────────────────────────────────────────── #

def db_connection():
    return psycopg2.connect(**DB_KWARGS)

def load_window(days) -> pd.Series:
    sql = f"""
        WITH sid AS (SELECT id FROM stock_data.symbols WHERE symbol = %s)
        SELECT p.date, p.close
        FROM {PRICE_TABLE} p, sid
        WHERE p.symbol_id = sid.id
          AND p.date >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY p.date;
    """
    with db_connection() as conn:
        return (pd.read_sql(sql, conn, params=(SYMBOL,),
                            parse_dates=["date"]).set_index("date")["close"])

def fetch_since(last_dt) -> pd.Series:
    sql = f"""
        WITH sid AS (SELECT id FROM stock_data.symbols WHERE symbol = %s)
        SELECT p.date, p.close
        FROM {PRICE_TABLE} p, sid
        WHERE p.symbol_id = sid.id
          AND p.date > %s
        ORDER BY p.date;
    """
    with db_connection() as conn:
        return (pd.read_sql(sql, conn, params=(SYMBOL, last_dt),
                            parse_dates=["date"]).set_index("date")["close"])

# ────────────────────────────────────── state I/O ───────────────────────────────────────────── #

def save_state(det, last_dt, hist):
    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"detector": det, "last": last_dt, "hist": hist}, f, -1)
    tmp.replace(STATE_PATH)
    logger.info("State saved – last=%s", last_dt.date())

def load_state() -> Tuple[EWMADetector, pd.Timestamp, pd.Series]:
    if STATE_PATH.exists():
        with STATE_PATH.open("rb") as f:
            p = pickle.load(f)
        logger.info("State loaded – last=%s", p["last"].date())
        return p["detector"], p["last"], p["hist"]

    # first run → warm-up
    warm = load_window(WARM_WINDOW_DAYS)
    det  = EWMADetector(warm.mean(), warm.std(ddof=1),
                        lam=LAM, L=L_MULT, two_sided=True)
    for v in warm.values:          # prime the EWMA
        det.step(float(v))
    last = warm.index.max()
    save_state(det, last, warm)
    return det, last, warm

# ───────────────────────────────────────── plot ─────────────────────────────────────────────── #

def make_plot(price, ewma_z, cps):
    tail_p = price.iloc[-PLOT_WINDOW_DAYS:]
    tail_z = pd.Series(ewma_z, index=price.index).iloc[-PLOT_WINDOW_DAYS:]

    fig, (axp, axz) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    tail_p.plot(ax=axp, lw=1.2)
    axp.set_ylabel("price")

    tail_z.plot(ax=axz, lw=1.0)
    axz.axhline( L_MULT, color="r", ls="--", lw=.8)
    axz.axhline(-L_MULT, color="r", ls="--", lw=.8)
    axz.set_ylabel("EWMA-z")
    for cp in cps:
        if cp in tail_p.index:
            axp.axvline(cp, color="red", lw=1, alpha=.65)
            axz.axvline(cp, color="red", lw=1, alpha=.65)

    fig.suptitle(f"{SYMBOL} – last {PLOT_WINDOW_DAYS} days (EWMA)")
    out_file = SCRIPT_DIR / f"ewma_plot_{SYMBOL}.png"
    fig.tight_layout(); fig.savefig(out_file, dpi=150); plt.close(fig)
    logger.info("Plot saved → %s", out_file.relative_to(SCRIPT_DIR))

# ───────────────────────────────────────── main ────────────────────────────────────────────── #

def main():
    if args.dry_run:
        logger.warning("DRY-RUN – state will NOT be persisted!")

    det, last_dt, price_hist = load_state()
    new_rows = fetch_since(last_dt)

    # For plotting we replay a fresh detector to build full z-scores
    det_full = EWMADetector(det.mu0, det.sigma0, lam=LAM, L=L_MULT)
    ewma_all, cp_dates = [], []
    for ts, px in price_hist.items():
        is_cp, _, stat = det_full.step(float(px))
        ewma_all.append(stat)
        if is_cp:
            cp_dates.append(ts)

    cooldown = 0
    if not new_rows.empty:
        for ts, px in new_rows.items():
            is_cp, direction, stat = det.step(float(px))
            ewma_all.append(stat)
            price_hist.loc[ts] = px

            if cooldown == 0 and is_cp:
                cp_dates.append(ts)
                logger.info("EWMA shift %s at %s  z=%.2f  refractory=%dd",
                            "↑" if direction>0 else "↓",
                            ts.date(), stat, REFRACTORY_DAYS)
                cooldown = REFRACTORY_DAYS

                # ── baseline re-estimate on trailing window ──
                win_start = ts - pd.Timedelta(days=WARM_WINDOW_DAYS)
                win = price_hist.loc[win_start:ts]
                if len(win) < 30:                                   # fallback
                    win = price_hist.iloc[-30:]
                det.set_baseline(float(win.mean()), float(win.std(ddof=1)))
                logger.info("Baseline reset → μ=%.3f  σ=%.3f  n=%d",
                            det.mu0, det.sigma0, len(win))

            if cooldown > 0:
                cooldown -= 1
        if not args.dry_run:
            save_state(det, price_hist.index.max(), price_hist)
    else:
        logger.info("No new data – plotting existing history")

    if cp_dates:
        logger.info("Changepoints up to %s: %s",
                    price_hist.index.max().date(),
                    ", ".join(d.date().isoformat() for d in cp_dates))

    make_plot(price_hist, ewma_all, cp_dates)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("EWMA nightly run failed: %s", exc)
        raise
