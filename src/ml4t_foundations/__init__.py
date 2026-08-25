"""The plumbing every ML4T: Foundations notebook runs on.

    !pip install -q ml4t-foundations
    from ml4t_foundations import save_component, load_component, append_result, report
"""

__version__ = "0.1.0"

from .components import catalog, load_component, save_component, source_of, status
from .contracts import get as contract
from .project import describe, home, setup
from .report import report
from .results import append_result, latest, results

__all__ = [
    "save_component", "load_component", "append_result", "report",
    "setup", "home", "describe", "status", "catalog", "contract", "results", "latest",
    "source_of", "__version__",
]
