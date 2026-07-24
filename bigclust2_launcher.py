import logging
import argparse
from importlib.metadata import PackageNotFoundError, version


def _installed_version() -> str:
    try:
        return version("bigclust2")
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bigclust2",
        description="A Python package for inspecting and visualising large clusterings",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    parser.add_argument(
        "--from",
        dest="dataset",
        metavar="DATASET",
        default=None,
        help="Path or URL of a dataset to load on startup",
    )
    parser.add_argument(
        "--filters",
        dest="filters",
        metavar="EXPR",
        default=None,
        help=(
            "Filter expression applied to the dataset on startup "
            "(e.g. 'dataset == \"hemibrain\"'). Requires --from."
        ),
    )
    parser.add_argument(
        "--embedding",
        dest="embedding",
        metavar="MODE",
        default=None,
        help=(
            "Load-time embedding mode, e.g. 'calculate from distances' or "
            "'calculate from features'. Requires --from."
        ),
    )
    args = parser.parse_args()

    if args.version:
        print(f"{parser.prog} {_installed_version()}")
        return

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("bigclust2").setLevel(logging.DEBUG)

    from bigclust2.gui import main as run_gui

    run_gui(dataset=args.dataset, filters=args.filters, embedding=args.embedding)
