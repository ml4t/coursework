"""A market is a spec, not a pipeline.

There is one pipeline - the twenty components wired by `runner.run` - and one spec per market.
A spec is the configuration plus the adapter that puts the market's stored file where
`data_panel` can read it. Nine markets is one implementation and nine of these.

The field names mirror `case_studies/<market>/config/setup.yaml` in the companion-code repo, so a
student who moves from a course notebook to a case-study notebook meets fields they have already
set. The two implementations stay independent: the book repo has to be able to change without the
course changing, and a shared file would make that impossible.

`download_mb` and `runtime_minutes` are here because they are what actually gates a student, and
nothing else in a config shows that `us_equities_panel` is roughly thirty times `etfs`. They are
measured on a run, not estimated; a spec that has not been run says so with a zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

MARKETS: dict[str, "MarketSpec"] = {}

BARS_PER_YEAR = {"daily": 252, "8h": 3 * 365, "hourly": 24 * 365, "weekly": 52}


@dataclass(frozen=True)
class MarketSpec:
    """One market's configuration and its data adapter."""

    market: str
    title: str
    acquire: Callable[[], Path]
    bar: str = "daily"
    #: Bars between a decision and the price it transacts at.
    execution_delay: int = 1
    #: Bars held out between the last fitted row and the first reported one. It has to be at
    #: least the labeler's horizon, or a training label overlaps a reported date. It is the
    #: runner's safety margin rather than a second copy of the horizon: how often to trade is
    #: `backtest_config`'s decision and is read from there.
    purge: int = 21
    #: Names in the universe, for documentation and for subsetting a large panel.
    assets: tuple[str, ...] = ()
    #: What `costs.class` says in the case study: `material`, `dominant`, `negligible`.
    cost_class: str = "material"
    download_mb: float = 0.0
    runtime_minutes: float = 0.0
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.bar not in BARS_PER_YEAR:
            raise ValueError(
                f"{self.market}: bar must be one of {', '.join(BARS_PER_YEAR)}, "
                f"got {self.bar!r}.\n"
                f"  The bar decides how a Sharpe is annualized, so it cannot be free text."
            )

    @property
    def bars_per_year(self) -> int:
        return BARS_PER_YEAR[self.bar]

    def describe(self) -> str:
        cost = f"costs {self.cost_class}"
        size = (f"{self.download_mb:,.0f} MB" if self.download_mb >= 10
                else f"{self.download_mb:.1f} MB" if self.download_mb
                else "download not yet measured")
        runtime = ("runtime not yet measured" if not self.runtime_minutes
                   else f"{self.runtime_minutes * 60:.0f} s end to end"
                   if self.runtime_minutes < 1
                   else f"{self.runtime_minutes:,.0f} min end to end")
        lines = [f"{self.market} - {self.title}",
                 f"  {len(self.assets) or 'an unlisted number of'} assets on {self.bar} bars, "
                 f"{cost}",
                 f"  decision: transact {self.execution_delay} bar later, "
                 f"{self.purge} bars purged between fitting and reporting",
                 f"  cost to run: {size}, {runtime}"]
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


def register_market(spec: MarketSpec, replace: bool = False) -> MarketSpec:
    existing = MARKETS.get(spec.market)
    if existing is not None and existing != spec and not replace:
        raise ValueError(
            f"A different spec is already registered as {spec.market!r}.\n"
            f"  Pass replace=True if you meant to change it."
        )
    MARKETS[spec.market] = spec
    return spec


def market(key: str) -> MarketSpec:
    if key not in MARKETS:
        known = ", ".join(sorted(MARKETS)) or "none yet"
        raise KeyError(
            f"There is no market registered as {key!r}.\n  Registered: {known}."
        )
    return MARKETS[key]


def market_catalog() -> str:
    """Every registered market, with what it costs to run one.

    Not called `markets`: this module is `ml4t_coursework.markets`, and a function re-exported
    under that name from the package would shadow the module for anyone importing it.
    """
    if not MARKETS:
        return "no markets registered"
    return "\n".join(market(k).describe() for k in sorted(MARKETS))


def _etf_assets() -> tuple[str, ...]:
    from . import data

    return tuple(data.SYMBOLS)


def _etf_file() -> Path:
    """Fetch the ETF closes if they are not already stored, and return the file."""
    from . import data

    stored = data.path()
    return stored if stored.is_file() else data.download(quiet=True)


register_market(MarketSpec(
    market="etfs",
    title="US-listed ETFs, adjusted daily closes",
    acquire=_etf_file,
    bar="daily",
    execution_delay=1,
    purge=21,
    assets=_etf_assets(),
    cost_class="material",
    # Measured 2026-09-06 on the workstation: a 2.2 MB parquet, then a baseline run and a
    # pipeline run over 5,031 sessions in about ten seconds each.
    download_mb=2.2,
    runtime_minutes=0.4,
    note=("The course's own market: a wide close file, an exchange calendar, and a universe that "
          "changes as funds list and close."),
    tags=("tier-a",),
))
