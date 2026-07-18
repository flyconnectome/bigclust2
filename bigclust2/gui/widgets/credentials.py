"""Storing, prompting for and managing tokens for authenticated services.

The Qt-free specs and auth-error classification live in
``bigclust2.credentials``; this file is only the Qt layer.

neuPrint and FlyTable read their tokens from environment variables, so stored
tokens are persisted in ``QSettings`` and injected into ``os.environ``. Clio
has its own token file, so we hand pasted tokens to ``clio.set_token()``.

Token values are never logged.
"""

from __future__ import annotations

import logging
import os
import shutil

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ...credentials import SERVICE_SPECS, CredentialSpec


logger = logging.getLogger(__name__)

SETTINGS_ORG = "bigclust2"
SETTINGS_APP = "Credentials"

#: Guards `apply_saved_credentials` so repeated calls are cheap no-ops.
_CREDENTIALS_APPLIED = False


def _settings():
    return QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)


def _settings_key(service_key: str) -> str:
    return f"credentials/{service_key}/token"


def _clio_module():
    """Import clio lazily; returns None when the optional dep is missing."""
    try:
        import clio

        return clio
    except ImportError:
        return None


def _clio_token_file() -> Path | None:
    clio = _clio_module()
    if clio is None:
        return None
    from clio.client import CLIO_TOKEN_FILE

    return Path(CLIO_TOKEN_FILE).expanduser()


class CredentialsManager:
    """Reads/writes stored tokens and reports per-service status."""

    def __init__(self, settings=None):
        self._settings = settings if settings is not None else _settings()

    # -- persistence ---------------------------------------------------

    def stored_token(self, service_key: str) -> str:
        value = self._settings.value(_settings_key(service_key), "")
        return str(value).strip() if value else ""

    def set_credentials(self, service_key: str, token: str) -> None:
        """Persist a token and make it effective immediately."""
        token = (token or "").strip()
        if not token:
            raise ValueError("Token must not be empty.")

        spec = SERVICE_SPECS[service_key]
        if spec.storage == "clio_file":
            clio = _clio_module()
            if clio is None:
                raise RuntimeError(
                    "clio-py is not installed. Install it with the 'clio' extra."
                )
            # set_token accepts a raw token or the full JSON document.
            clio.set_token(token)
        else:
            self._settings.setValue(_settings_key(service_key), token)
            self._settings.sync()
            for var in spec.env_vars:
                os.environ[var] = token
        logger.info("Stored %s credentials.", spec.display_name)

    def clear_credentials(self, service_key: str) -> None:
        spec = SERVICE_SPECS[service_key]
        if spec.storage == "clio_file":
            path = _clio_token_file()
            if path is not None and path.is_file():
                path.unlink()
        else:
            self._settings.remove(_settings_key(service_key))
            self._settings.sync()
            for var in spec.env_vars:
                os.environ.pop(var, None)
        logger.info("Cleared %s credentials.", spec.display_name)

    # -- status --------------------------------------------------------

    def has_credentials(self, service_key: str) -> bool:
        spec = SERVICE_SPECS[service_key]
        if spec.storage == "clio_file":
            path = _clio_token_file()
            if path is not None and path.is_file():
                return True
            # gcloud can mint a token on the fly.
            return bool(shutil.which("gcloud"))
        if self.stored_token(service_key):
            return True
        return any(os.environ.get(var) for var in spec.env_vars)

    def credential_status(self, service_key: str) -> str:
        """Human-readable status for the management dialog."""
        spec = SERVICE_SPECS[service_key]
        if spec.storage == "clio_file":
            if _clio_module() is None:
                return "clio-py not installed"
            path = _clio_token_file()
            if path is not None and path.is_file():
                return "Set (token file)"
            if shutil.which("gcloud"):
                return "Not set (gcloud fallback available)"
            return "Not set"

        if self.stored_token(service_key):
            return "Set"
        if any(os.environ.get(var) for var in spec.env_vars):
            return "Set (from environment)"
        return "Not set"

    # -- prompting -----------------------------------------------------

    def ensure_credentials(self, service_key, parent=None) -> bool:
        """Return True if credentials are available, prompting if they are not.

        Must be called from the UI thread.
        """
        if not service_key or service_key not in SERVICE_SPECS:
            return True
        if self.has_credentials(service_key):
            return True
        return self.prompt_for_credentials(service_key, parent=parent)

    def prompt_for_credentials(self, service_key, parent=None, invalid=False) -> bool:
        """Show the token dialog and store the result. Returns True on success."""
        spec = SERVICE_SPECS[service_key]
        dialog = TokenPromptDialog(spec, parent=parent, invalid=invalid)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return False
        token = dialog.token()
        if not token:
            return False
        try:
            self.set_credentials(service_key, token)
        except Exception as exc:  # never include the token in the message
            QtWidgets.QMessageBox.critical(
                parent,
                "Could not save credentials",
                f"Failed to store {spec.display_name} credentials: {exc}",
            )
            return False
        return True


def apply_saved_credentials(settings=None, force=False) -> None:
    """Inject stored tokens into the environment. Idempotent and side-effect free.

    Called at start-up before any backend is used. Services without a stored
    token are left alone, so environments that already export the variables
    (e.g. developer machines) keep working unchanged.
    """
    global _CREDENTIALS_APPLIED
    if _CREDENTIALS_APPLIED and not force:
        return

    manager = CredentialsManager(settings=settings)
    for spec in SERVICE_SPECS.values():
        if spec.storage != "qsettings_env":
            continue
        token = manager.stored_token(spec.key)
        if not token:
            continue
        for var in spec.env_vars:
            os.environ[var] = token
    _CREDENTIALS_APPLIED = True


class TokenPromptDialog(QtWidgets.QDialog):
    """Ask the user to paste a token for one service."""

    def __init__(self, spec: CredentialSpec, parent=None, invalid=False):
        super().__init__(parent)
        self._spec = spec
        self.setWindowTitle(f"{spec.display_name} credentials")
        self.setModal(True)
        self.resize(460, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if invalid:
            headline = (
                f"Your {spec.display_name} token was rejected (invalid or "
                "expired). Please provide a new one."
            )
        else:
            headline = f"No {spec.display_name} token found."
        headline_label = QtWidgets.QLabel(headline)
        headline_label.setWordWrap(True)
        layout.addWidget(headline_label)

        instructions = QtWidgets.QLabel(spec.instructions)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        link = QtWidgets.QLabel(
            f'<a href="{spec.token_page_url}">Open the {spec.display_name} '
            "token page</a>"
        )
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        link.setWordWrap(True)
        layout.addWidget(link)

        row = QtWidgets.QHBoxLayout()
        self.token_edit = QtWidgets.QLineEdit()
        self.token_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.token_edit.setPlaceholderText("Paste your token here")
        row.addWidget(self.token_edit, 1)

        self.reveal_button = QtWidgets.QToolButton()
        self.reveal_button.setText("Show")
        self.reveal_button.setCheckable(True)
        self.reveal_button.setToolTip("Show/hide the token")
        self.reveal_button.toggled.connect(self._on_reveal_toggled)
        row.addWidget(self.reveal_button)
        layout.addLayout(row)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._ok_button = self.buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self._ok_button.setEnabled(False)
        self.token_edit.textChanged.connect(self._on_text_changed)

    def _on_reveal_toggled(self, checked):
        self.token_edit.setEchoMode(
            QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password
        )
        self.reveal_button.setText("Hide" if checked else "Show")

    def _on_text_changed(self, text):
        self._ok_button.setEnabled(bool(text.strip()))

    def token(self) -> str:
        return self.token_edit.text().strip()


class CredentialsDialog(QtWidgets.QDialog):
    """Review, update and clear the stored tokens for all services."""

    def __init__(self, manager=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Credentials")
        self.resize(560, 0)

        self._manager = manager if manager is not None else CredentialsManager()
        self._status_labels = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QtWidgets.QLabel(
            "Tokens for the services bigclust2 can read annotations from. "
            "They are stored locally and are never shown in logs."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        for row, spec in enumerate(SERVICE_SPECS.values()):
            name_label = QtWidgets.QLabel(f"<b>{spec.display_name}</b>")
            grid.addWidget(name_label, row, 0)

            status = QtWidgets.QLabel("")
            self._status_labels[spec.key] = status
            grid.addWidget(status, row, 1)

            update = QtWidgets.QPushButton("Update…")
            update.clicked.connect(
                lambda _checked=False, key=spec.key: self._on_update(key)
            )
            grid.addWidget(update, row, 2)

            clear = QtWidgets.QPushButton("Clear")
            clear.clicked.connect(
                lambda _checked=False, key=spec.key: self._on_clear(key)
            )
            grid.addWidget(clear, row, 3)

        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.refresh_statuses()

    def refresh_statuses(self):
        for key, label in self._status_labels.items():
            label.setText(self._manager.credential_status(key))

    def _on_update(self, service_key):
        if self._manager.prompt_for_credentials(service_key, parent=self):
            self.refresh_statuses()

    def _on_clear(self, service_key):
        spec = SERVICE_SPECS[service_key]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear credentials",
            f"Remove the stored {spec.display_name} token?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            self._manager.clear_credentials(service_key)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not clear credentials",
                f"Failed to clear {spec.display_name} credentials: {exc}",
            )
        self.refresh_statuses()
