import os
import sys

# Suppress octarine's warning about event loop potentially not running
os.environ['OCTARINE_CHECK_LOOP'] = '0'

# Qt-free helpers for authoring and maintaining projects on disk. Importing these
# must not drag in the GUI stack so that headless data-prep scripts - and
# `bigclust2 update` - work on machines without a display or GPU.
from .project_builder import ProjectBuilder
from .project_update import ProjectMetaUpdate, update_project_meta

# Some feedback for the user that stuff is happening, especially when running from the command line
__context__ = "cli" if 'bigclust2' in sys.argv[0] else "gui"


def __getattr__(name):
    """Load the GUI the first time `main` is looked up (PEP 562).

    Importing it costs seconds (Qt, pygfx, wgpu), which is why it is deferred
    rather than imported alongside the Qt-free helpers above.
    """
    if name != "main":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if __context__ == "cli":
        print("Loading bigclust2... ", end="", flush=True)
    from .gui import main

    if __context__ == "cli":
        print("done.", flush=True)

    globals()["main"] = main  # subsequent lookups skip this hook
    return main


__all__ = ["main", "ProjectBuilder", "ProjectMetaUpdate", "update_project_meta"]
