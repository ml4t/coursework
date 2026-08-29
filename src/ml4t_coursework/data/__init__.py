"""The course dataset: what it is, how to fetch it once, and how to know your copy is sound.

The prices are licensed and we may not redistribute them, so what ships here is everything except
the prices: the symbol list, the window, the fetch, and a one-way fingerprint. `data/DATA.md` in
the course repository is the student-facing explanation of why, and no unit re-explains it.
"""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

import numpy as np
import pandas as pd

from .. import project

FILENAME = "etf_close.parquet"
START, END = "2006-01-01", "2026-01-01"

SYMBOLS = """
ACWI ACWX AGG BIL BND BNDX DBA DBC DIA DVY EEM EFA EMB EWA EWC EWG EWH EWI EWJ EWL
EWN EWP EWQ EWT EWU EWW EWY EWZ EZA FXB FXE FXI FXY GLD GOVT GSG HYG IAU IBB IEF
IEFA IEMG IJR INDA ITA ITB IVE IVW IWM IYR JNK KRE LQD MCHI MDY MTUM MUB OIH PPLT
QQQ QUAL RSP SCHD SDY SHY SLV SMH SOXX SPY THD TIP TLT UNG USMV USO UUP VCSH VEA
VGK VIG VLUE VNQ VTI VTV VUG VWO XBI XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY
XME XRT
""".split()


def fingerprint() -> pd.DataFrame:
    """Per-symbol session counts, dates, volatility and average move.

    Deliberately one-way: enough to tell you your download is sound, not enough to rebuild prices
    we are not allowed to hand you.
    """
    with resources.as_file(resources.files(__package__) / "etf_close_fingerprint.csv") as path:
        return pd.read_csv(path, index_col="symbol")


def path() -> Path:
    return project.data_dir() / FILENAME


def load(target: Path | None = None) -> pd.DataFrame:
    """The panel every notebook reads. Never downloads."""
    target = path() if target is None else Path(target)
    if not target.is_file():
        raise FileNotFoundError(
            f"The price panel is not in your project folder yet.\n"
            f"  Expected: {target}\n"
            f"  Run the setup notebook once. It fetches the file and checks it, and no other "
            f"notebook in the course downloads anything."
        )
    frame = pd.read_parquet(target)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame.sort_index().astype("float64")


def download(force: bool = False, quiet: bool = False, out: Path | str | None = None) -> Path:
    """Fetch the panel, once, and check it against the fingerprint.

    Writes into the student's project folder by default. `out` names a file instead, which is
    what a checkout run outside Colab uses: there is no Drive and no project folder, only a
    repository with a `data/` directory in it.
    """
    target = path() if out is None else Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not force:
        if not quiet:
            frame = load(target)
            print(f"already here: {target}\n  {frame.shape[0]:,} sessions x {frame.shape[1]} "
                  f"symbols, {frame.index[0].date()} to {frame.index[-1].date()}")
        return target
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "The one-time download needs yfinance:\n"
            "  !pip install -q yfinance\n"
            "Nothing else in the course does, which is why it is not installed with the helper."
        ) from exc

    prices = yf.download(SYMBOLS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    prices = prices.sort_index().sort_index(axis=1).astype("float64")
    prices.index.name = "date"
    prices.to_parquet(target, compression="zstd")
    if not quiet:
        print(f"{target}\n  {prices.shape[0]:,} sessions x {prices.shape[1]} symbols, "
              f"{prices.index[0].date()} to {prices.index[-1].date()}")
        print(check(prices))
    return target


def check(prices: pd.DataFrame | None = None) -> str:
    """Compare a copy against the fingerprint and say, in words, whether it is sound.

    A relative volatility deviation over 1%, or a session count off by more than
    five, is a delisting or a bad bar rather than an adjustment revision.
    """
    prices = load() if prices is None else prices
    reference = fingerprint()
    returns = prices.pct_change(fill_method=None)
    mine = pd.DataFrame({
        "sessions": prices.notna().sum(),
        "annualized_vol": returns.std() * np.sqrt(252),
        "mean_abs_return": returns.abs().mean(),
    })

    missing = sorted(set(reference.index) - set(mine.index))
    shared = reference.index.intersection(mine.index)
    vol_off = (mine.loc[shared, "annualized_vol"] / reference.loc[shared, "annualized_vol"] - 1).abs()
    move_off = (mine.loc[shared, "mean_abs_return"] / reference.loc[shared, "mean_abs_return"] - 1).abs()
    session_gap = (mine.loc[shared, "sessions"] - reference.loc[shared, "sessions"]).abs()

    lines = [f"  checked {len(shared)} of {len(reference)} symbols against the course's copy",
             f"    session count      largest difference {int(session_gap.max())}",
             f"    annualized vol     largest deviation  {vol_off.max():.2%}",
             f"    average daily move largest deviation  {move_off.max():.2%}"]
    suspect = sorted(set(vol_off[vol_off > 0.01].index) | set(session_gap[session_gap > 5].index))
    if missing:
        lines.append(f"    missing entirely   {', '.join(missing)}")
    if suspect or missing:
        lines.append("  Worth a look: " + (", ".join(suspect) or "none") + ". A symbol that "
                     "drifts this far is usually a delisting, a reused ticker or a bad bar, not a "
                     "revision.")
    else:
        lines.append("  Sound. Your results will match the course's to three or four decimals.")
    return "\n".join(lines)
