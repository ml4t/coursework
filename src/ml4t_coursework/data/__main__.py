"""Fetch the course price panel from a shell, for a checkout that has no Colab around it.

    uv run --with "ml4t-coursework[data]" python -m ml4t_coursework.data --out data/etf_close.parquet

Inside a notebook, call `ml4t_coursework.data.download()` instead: it writes into the project
folder on the student's Drive and needs no path.
"""

from __future__ import annotations

import argparse
import pathlib

from . import download


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ml4t_coursework.data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="where to write the panel. Defaults to the project folder, which is "
                             "what a notebook wants and a checkout does not have.")
    parser.add_argument("--force", action="store_true",
                        help="download again even if the file is already there.")
    args = parser.parse_args()
    download(force=args.force, out=args.out)


if __name__ == "__main__":
    main()
