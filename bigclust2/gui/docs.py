"""Links into the documentation site.

In-app help deep-links into the docs rather than restating them: a shortcut list
or a data-format description that lives in two places drifts, and the copy nobody
is looking at is the one that goes stale.

Kept in its own module so both `core` and the dialogs it imports (`loaders`, …)
can link out without an import cycle.
"""

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

__all__ = ["DOCS_URL", "docs_url", "open_docs", "docs_link"]

DOCS_URL = "https://flyconnectome.github.io/bigclust2/"


def docs_url(page: str = "") -> str:
    """Absolute URL of a documentation page.

    Parameters
    ----------
    page :  str
            Page path relative to the site root, without the `.md` suffix and
            with a trailing slash - e.g. `"reference/shortcuts/"`. Empty for the
            landing page.

    Returns
    -------
    str

    """
    return DOCS_URL + page.lstrip("/")


def open_docs(page: str = "") -> None:
    """Open a documentation page in the user's browser."""
    QDesktopServices.openUrl(QUrl(docs_url(page)))


def docs_link(page: str = "", text: str = "Documentation") -> str:
    """An HTML anchor to a documentation page.

    For embedding in `QLabel`s and rich-text message boxes. Note the widget also
    needs `setOpenExternalLinks(True)` (or a `linkActivated` connection) or the
    anchor renders as a link and does nothing when clicked.
    """
    return f'<a href="{docs_url(page)}">{text}</a>'
