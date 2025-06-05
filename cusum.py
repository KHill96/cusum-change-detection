import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from dotenv import load_dotenv
import argparse, os, pathlib, logging, pickle
from typing import Tuple

load_dotenv()

cli = argparse.ArgumentParser()
cli.add_argument("--symbol",     default="AAPL")
cli.add_argument("--lookback",   type=int, default=365, help="days of history")
cli.add_argument("--k",          type=float, default=0.5, help="drift k")
cli.add_argument("--h",          type=float, default=5.0, help="threshold h")
cli.add_argument('--refractory', type=int, default=5, help="days to skip after detecting a changepoint" )
cli.add_argument("--warm", type=int, default=180, help="warm-up days")
cli.add_argument("--plot-window", type=int, default=250, help="tail in plot")
cli.add_argument("--dry-run", action="store_true", help="run without persisting state")
cli.add_argument("--force", action="store_true",
                 help="rebuild state ignoring cache")
cli.add_argument("--min-run", type=int, default=10,
                 help="minimum bars between change-points")
cli.add_argument("--sigma-cap", type=float, default=2.5,
                 help="cap σ at this multiple of initial σ")
args = cli.parse_args()


DB_KWARGS = dict(
    host     = os.getenv("DB_HOST"),
    port     = int(os.getenv("DB_PORT")),
    dbname   = os.getenv("DB_NAME"),
    user     = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


SYMBOL             = args.symbol.upper()
PRICE_TABLE        = "stock_data.daily_prices"
STATE_PATH         = SCRIPT_DIR / f"state_{SYMBOL}.pkl"
LOGFILE            = SCRIPT_DIR / f"nightly_{SYMBOL}.log"

WARM_WINDOW_DAYS   = args.warm
PLOT_WINDOW_DAYS   = args.plot_window
K_DRIFT            = args.k
H_THRESHOLD        = args.h
REFRACTORY_DAYS    = args.refractory

logging.basicConfig(
    filename=LOGFILE,
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(f"cusum_{SYMBOL.lower()}")

class CUSUMDetector:
    def __init__(self, mu0, sigma0, k=0.5, h=5.0, *, two_sided=True):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.k = k
        self.h = h
        self.two_sided = two_sided
        self.reset()
    
    def reset(self):
        self.gp = self.gn = 0.0
        self.t = -1
        
    def set_baseline(self, mu0: float, sigma0: float):
        """
        Update the in-control mean and stdev after a detected shift, then
        clear the cumulative scores so that monitoring restarts from zero.
        """
        self.mu0    = mu0
        # avoid division-by-zero in very quiet markets
        self.sigma0 = max(sigma0, 1e-8)
        self.reset()
    
    def step(self, x):
        self.t += 1
        z = (x - self.mu0) / self.sigma0
        self.gp = max(0.0, self.gp + z - self.k)
        up = self.gp > self.h
        down = False
        if self.two_sided:
            self.gn = max(0.0, self.gn - z - self.k)
            down = self.gn > self.h
        if up or down:
            d = 1 if up else -1
            stat = self.gp if up else self.gn
            self.reset()
            return True, d, stat
        return False, None, max(self.gp, self.gn)
    
def db_connection():
    return psycopg2.connect(**DB_KWARGS)


def load_window(days) -> pd.Series:
    sql = f"""
        WITH sid AS (SELECT id FROM stock_data.symbols WHERE symbol=%s)
        SELECT p.date, p.close
        FROM {PRICE_TABLE} p, sid
        WHERE p.symbol_id = sid.id
            AND p.date >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY p.date;
    """
    
    with db_connection() as conn:
        return (pd.read_sql(sql, conn, params=(SYMBOL, ), parse_dates=["date"]).set_index("date")["close"])
    
def fetch_since(last_dt) -> pd.Series:
    sql = f"""
        WITH sid AS (SELECT id FROM stock_data.symbols WHERE symbol=%s)
        SELECT p.date, p.close
        FROM {PRICE_TABLE} p, sid
        WHERE p.symbol_id = sid.id
            AND p.date > %s
        ORDER BY p.date;
    """
    
    with db_connection() as conn:
        return (pd.read_sql(sql, conn, params=(SYMBOL, last_dt), parse_dates=["date"]).set_index("date")["close"])
    

def save_state(det, last_dt, hist):
    with STATE_PATH.with_suffix(".tmp").open("wb") as f:
        pickle.dump({"detector": det, "last": last_dt, "hist": hist}, f, -1)
    STATE_PATH.with_suffix(".tmp").replace(STATE_PATH)
    logger.info("State saved - last-date=%s", last_dt.date())
    

def load_state() -> Tuple[CUSUMDetector, pd.Timestamp, pd.Series]:
    if STATE_PATH.exists():
        with STATE_PATH.open("rb") as f:
            p = pickle.load(f)
        if len(p["hist"]) >= args.lookback:          # cached long enough
            logger.info("State loaded – last=%s", p["last"].date())
            return p["detector"], p["last"], p["hist"]
        logger.info("Existing state too short – rebuilding ...")

    # fresh build
    full_hist = load_window(args.lookback)           # ← use lookback!
    warm      = full_hist.iloc[:WARM_WINDOW_DAYS]
    det       = CUSUMDetector(warm.mean(), warm.std(ddof=1),
                              k=K_DRIFT, h=H_THRESHOLD)
    for v in warm.values:
        det.step(float(v))
    last = full_hist.index.max()
    save_state(det, last, full_hist)
    return det, last, full_hist


def make_plot(price, cusum_vals, cps):
    tail_p = price.iloc[-PLOT_WINDOW_DAYS:]
    tail_c = pd.Series(cusum_vals, index=price.index).iloc[-PLOT_WINDOW_DAYS:]

    fig, (axp, axc) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    tail_p.plot(ax=axp, lw=1)
    axp.set_ylabel("price")
    tail_c.plot(ax=axc, lw=1)
    axc.axhline(H_THRESHOLD,  color="r", ls="--", lw=.8)
    axc.axhline(-H_THRESHOLD, color="r", ls="--", lw=.8)
    axc.set_ylabel("CUSUM")
    for cp in cps:
        if cp in tail_p.index:
            axp.axvline(cp, color="red", lw=1, alpha=.6)
            axc.axvline(cp, color="red", lw=1, alpha=.6)
    fig.suptitle(f"{SYMBOL} - last {PLOT_WINDOW_DAYS} days (CUSUM)")
    out_file = SCRIPT_DIR / f"plot_{SYMBOL}_3_why.png"
    fig.tight_layout(); fig.savefig(out_file, dpi=150); plt.close(fig)
    logger.info("Plot saved → %s", out_file.relative_to(SCRIPT_DIR))
    
    
def main():
    if args.dry_run:
        logger.warning("Running in DRY MODE - changes will not be saved.")
    det, last_dt, price_hist = load_state()
    new_rows = fetch_since(last_dt)
    
    det_full = CUSUMDetector(det.mu0, det.sigma0,
                         k=K_DRIFT, h=H_THRESHOLD, two_sided=True)
    cusum_all, cp_dates = [], []
    for idx, (ts, px) in enumerate(price_hist.items()):
        is_cp, _, stat = det_full.step(float(px))
        cusum_all.append(stat)
        if is_cp:
            cp_dates.append(ts)

            # SAME trailing-window logic you use in streaming part
            win_start = ts - pd.Timedelta(days=WARM_WINDOW_DAYS)
            win = price_hist.loc[win_start:ts]
            if len(win) < 30:
                win = price_hist.iloc[max(0, idx-29):idx+1]

            det_full.set_baseline(float(win.mean()), float(win.std(ddof=1)))

            logger.info("Replay reset @%s → μ=%.3f  σ=%.3f  n=%d",
                        ts.date(), det_full.mu0, det_full.sigma0, len(win))
            
        cooldown = 0
        
    if not new_rows.empty:
        for ts, px in new_rows.items():
            is_cp, direction, stat = det.step(float(px))
            cusum_all.append(stat)
            price_hist.loc[ts] = px

            if cooldown == 0 and is_cp:
                cp_dates.append(ts)
                logger.info("CUSUM shift %s at %s  stat=%.2f  refractory=%dd",
                            "↑" if direction>0 else "↓",
                            ts.date(), stat, REFRACTORY_DAYS)
                cooldown = REFRACTORY_DAYS
                win_start = ts - pd.Timedelta(days=WARM_WINDOW_DAYS)
                baseline_window = price_hist.loc[win_start:ts]
                # fall back to last 30 obs if the normal warm window is too short
                if len(baseline_window) < 30:
                    baseline_window = price_hist.iloc[-30:]

                new_mu = float(baseline_window.mean())
                new_sd = float(baseline_window.std(ddof=1))

                det.set_baseline(new_mu, new_sd)
                logger.info("Baseline reset → μ=%.3f  σ=%.3f  (n=%d)",
                            new_mu, new_sd, len(baseline_window))
            if cooldown > 0:
                cooldown -= 1
        save_state(det, price_hist.index.max(), price_hist)
    else:
        logger.info("No new data - plotting existing history")
    
    if cp_dates:
        logger.info("Changepoints up to %s: %s",
                    price_hist.index.max().date(),
                    ", ".join(d.date().isoformat() for d in cp_dates))

    make_plot(price_hist, cusum_all, cp_dates)

    if not new_rows.empty and not args.dry_run:
        save_state(det, price_hist.index.max(), price_hist)
    elif args.dry_run:
        logger.info("Dry run - no state was saved.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Nightly CUSUM run failed: %s", exc)
        raise

