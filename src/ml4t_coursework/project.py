"""The one project folder, and how a notebook finds it.

Where it is comes from the course the session selected, so two courses on one Drive keep separate
folders and separate components.

It is a Google Drive folder, not a repository: git is not a prerequisite of this course. The setup
notebook creates it once and every later notebook mounts it. Nothing in the course depends on what
is in it - data re-fetches if absent, the results file is created if missing, and a component the
student does not have falls back to the shipped reference - so losing the folder costs minutes.
"""

from __future__ import annotations

import os
from pathlib import Path

from .course import active_course

DRIVE_MOUNT = Path("/content/drive")


def on_colab() -> bool:
    return "google.colab" in os.sys.modules or Path("/content").is_dir()


def mount_drive(quiet: bool = True) -> bool:
    """Mount Google Drive when running on Colab. A no-op anywhere else.

    `google.colab` being importable does not mean a Drive can be mounted. The
    published Colab runtime image has the package but no backend to mount
    through, so `drive.mount` raises there; a headless or automated session can
    also reach a mount it cannot complete. Returning False sends `home` to the
    environment override or the home directory, which is the fallback it already
    implements. Raising instead would fail before that fallback is ever consulted.
    """
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return False
    if (DRIVE_MOUNT / "MyDrive").is_dir():
        return True
    try:
        drive.mount(str(DRIVE_MOUNT))
    except Exception:
        return False
    return (DRIVE_MOUNT / "MyDrive").is_dir()


def home(create: bool = True) -> Path:
    """The project folder, resolved in the order: environment override, Drive, home directory."""
    course = active_course()
    override = os.environ.get(course.env_var)
    if override:
        path = Path(override).expanduser()
    elif (DRIVE_MOUNT / "MyDrive").is_dir():
        path = DRIVE_MOUNT / "MyDrive" / course.folder
    else:
        path = Path.home() / course.folder
    if create:
        for sub in ("components", "results", "data"):
            (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def components_dir(create: bool = True) -> Path:
    return home(create) / "components"


def results_file(create: bool = True) -> Path:
    return home(create) / "results" / "runs.csv"


def data_dir(create: bool = True) -> Path:
    return home(create) / "data"


def setup(quiet: bool = False) -> Path:
    """Called once, by the setup notebook. Mounts Drive if there is Drive to mount."""
    mounted = mount_drive()
    path = home(create=True)
    if not quiet:
        where = "your Google Drive" if mounted else "this machine"
        print(f"Project folder ready on {where}:\n  {path}")
        print("  components/  one file per component you write")
        print("  results/     one row per pipeline run")
        print("  data/        the price panel, fetched once")
    return path


def describe() -> str:
    path = home(create=False)
    if not path.is_dir():
        return f"No project folder yet at {path}. Run the setup notebook once."
    saved = sorted(p.stem for p in (path / "components").glob("*.meta.json"))
    rows = 0
    if results_file(create=False).is_file():
        rows = max(sum(1 for _ in results_file(create=False).open()) - 1, 0)
    return (f"{path}\n  components saved: {len(saved)}"
            f"{' (' + ', '.join(saved) + ')' if saved else ''}\n  results rows: {rows}")
