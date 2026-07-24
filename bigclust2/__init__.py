import os
import sys

# Suppress octarine's warning about event loop potentially not running
os.environ['OCTARINE_CHECK_LOOP'] = '0'

# Qt-free project authoring helper. Imported before the GUI so headless
# data-prep scripts (`from bigclust2 import ProjectBuilder`) work regardless of
# the GUI stack.
from .project_builder import ProjectBuilder

# Some feedback for the user that stuff is happening, especially when running from the command line
__context__ = "cli" if 'bigclust2' in sys.argv[0] else "gui"
if __context__ == "cli":
    print("Loading bigclust2... ", end="", flush=True)

from .gui import main

if __context__ == "cli":
    print("done.", flush=True)

__all__ = ["main", "ProjectBuilder"]