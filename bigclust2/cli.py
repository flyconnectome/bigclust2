import argparse

from .gui import main
from .__version__ import __version__


def app():
    parser = argparse.ArgumentParser(
        prog="bigclust2", description="A Python package for inspecing and visualising large clusterings"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.parse_args()
    main()
