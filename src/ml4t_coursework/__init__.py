"""The plumbing every ML4T course notebook runs on.

    !pip install -q ml4t-coursework
    import ml4t_coursework as mlc
    mlc.use("foundations")

    from ml4t_coursework import save_component, load_component, append_result, report, run

The package holds the student's project folder, the components they write and the contracts those
are checked against, their results log and their submission report. It is named for what it does
rather than for a course, because more than one course installs it.
"""

__version__ = "0.2.0"

from .components import catalog, load_component, save_component, source_of, status
from .contracts import add_source
from .contracts import get as contract
from .course import Course, active_course, register_course, use
from .markets import MarketSpec, market, market_catalog, register_market
from .project import describe, home, setup
from .report import report
from .results import append_result, latest, results
from .runner import run

__all__ = [
    "save_component", "load_component", "append_result", "report",
    "setup", "home", "describe", "status", "catalog", "contract", "results", "latest",
    "source_of", "use", "Course", "register_course", "active_course", "add_source",
    "run", "MarketSpec", "market", "market_catalog", "register_market",
    "__version__",
]
