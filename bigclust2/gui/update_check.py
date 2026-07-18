"""Background check against PyPI for a newer bigclust2 release.

This module must stay importable headless: only QtCore, never widgets or the
wgpu-based GUI stack.
"""

import requests

from packaging.version import Version, InvalidVersion
from PySide6 import QtCore

PYPI_JSON_URL = "https://pypi.org/pypi/bigclust2/json"


def fetch_latest_version(timeout=5):
    """Return the latest release version string on PyPI. Raises on any failure."""
    resp = requests.get(PYPI_JSON_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["info"]["version"]


def is_outdated(current, latest):
    """Return True if ``current`` is strictly older than ``latest``.

    Invalid version strings compare as up to date (never nag on garbage).
    """
    try:
        return Version(current) < Version(latest)
    except InvalidVersion:
        return False


class UpdateCheckSignals(QtCore.QObject):
    """Signal holder: QRunnable is not a QObject, so signals live here."""

    result = QtCore.Signal(str)  # latest version string from PyPI
    error = QtCore.Signal(str)  # human-readable failure message


class UpdateCheckRunnable(QtCore.QRunnable):
    """Fetch the latest PyPI version off the GUI thread."""

    def __init__(self):
        super().__init__()
        self.signals = UpdateCheckSignals()

    def run(self):
        """Fetch from PyPI and emit either ``result`` or ``error``."""
        try:
            latest = fetch_latest_version()
        except Exception as e:  # network, HTTP, JSON shape - all non-fatal
            self.signals.error.emit(str(e))
        else:
            self.signals.result.emit(latest)
