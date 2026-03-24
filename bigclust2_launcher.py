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
    args = parser.parse_args()

    if args.version:
        print(f"{parser.prog} {_installed_version()}")
        return

    from bigclust2.gui import main as run_gui

    run_gui()
