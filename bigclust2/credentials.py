"""Credential specs and auth-error classification for external services.

This module is deliberately free of Qt and of any service-client imports so
that it can be used from the (Qt-free) annotation backends. The GUI layer
lives in :mod:`bigclust2.gui.widgets.credentials`.

Token values must never be logged or included in exception messages.
"""

from __future__ import annotations

from dataclasses import dataclass


#: Where users obtain their tokens. Kept as constants so they are easy to update.
NEUPRINT_TOKEN_URL = "https://neuprint.janelia.org/account"
FLYTABLE_TOKEN_URL = "https://flytable.mrc-lmb.cam.ac.uk"
# Clio provides these as module constants; see `_clio_settings_url`.
CLIO_TOKEN_URL_FALLBACK = "https://clio.janelia.org/settings"


def _clio_settings_url() -> str:
    """Clio's token page, preferring the constant shipped with clio-py."""
    try:
        from clio.client import CLIO_SETTINGS_URL

        return CLIO_SETTINGS_URL
    except Exception:
        return CLIO_TOKEN_URL_FALLBACK


@dataclass(frozen=True)
class CredentialSpec:
    """Describes how one service is authenticated and where to get a token."""

    key: str
    display_name: str
    #: Environment variables the client library reads (empty for file-based auth).
    env_vars: tuple[str, ...]
    #: "qsettings_env" -> store in QSettings and inject into os.environ.
    #: "clio_file" -> hand to clio.set_token(), which writes ~/clio_token.json.
    storage: str
    token_page_url: str
    instructions: str


SERVICE_SPECS: dict[str, CredentialSpec] = {
    "neuprint": CredentialSpec(
        key="neuprint",
        display_name="neuPrint",
        env_vars=("NEUPRINT_APPLICATION_CREDENTIALS",),
        storage="qsettings_env",
        token_page_url=NEUPRINT_TOKEN_URL,
        instructions=(
            "Log in to neuPrint, open your account page and copy the "
            '"Auth Token", then paste it below.'
        ),
    ),
    "flytable": CredentialSpec(
        key="flytable",
        display_name="FlyTable",
        env_vars=("SEATABLE_TOKEN",),
        storage="qsettings_env",
        token_page_url=FLYTABLE_TOKEN_URL,
        instructions=(
            "Log in to FlyTable, then copy your API (auth) token from your "
            "account settings and paste it below."
        ),
    ),
    "clio": CredentialSpec(
        key="clio",
        display_name="Clio",
        env_vars=(),
        storage="clio_file",
        token_page_url=_clio_settings_url(),
        instructions=(
            "Log in to Clio, open the settings page and copy your "
            '"ClioStore Token", then paste it below. You can paste either the '
            "token itself or the full JSON document."
        ),
    ),
}


#: Backend name (``AnnotationBackend.BACKEND_NAME``) -> service key.
BACKEND_SERVICE_KEYS: dict[str, str] = {
    "neuPrint": "neuprint",
    "Clio": "clio",
    "FlyTable": "flytable",
    "FlyWire @ FlyTable": "flytable",
    "Hemibrain @ FlyTable": "flytable",
}


def service_key_for_backend(backend_name: str) -> str | None:
    """Return the service key for a backend name, or None if it needs no auth."""
    return BACKEND_SERVICE_KEYS.get(backend_name)


class MissingCredentialsError(RuntimeError):
    """Raised when a service rejects us for lack of (valid) credentials.

    Carries enough context for the GUI to prompt for the right token. The
    message never contains the token itself.
    """

    def __init__(self, service_key: str, *, invalid: bool = False):
        spec = SERVICE_SPECS.get(service_key)
        self.service_key = service_key
        self.display_name = spec.display_name if spec else service_key
        self.invalid = invalid

        if invalid:
            msg = (
                f"{self.display_name} rejected the current credentials "
                "(invalid or expired token)."
            )
        else:
            msg = f"No {self.display_name} credentials found."
        super().__init__(f"{msg} Use Window > Credentials... to set your token.")


def _http_status(exc: BaseException):
    """HTTP status code of a requests-style exception, if any."""
    return getattr(getattr(exc, "response", None), "status_code", None)


def _seatable_status(exc: BaseException):
    """Status code from ``ConnectionError(status_code, text)`` (seatable_api)."""
    # Guard on int args so genuine socket errors are not misread as auth
    # failures; requests raises its own ConnectionError subclass for those.
    if type(exc) is ConnectionError and exc.args and isinstance(exc.args[0], int):
        return exc.args[0]
    return None


def _contains(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


#: seatable_api's auth exceptions, matched by name so we need not import it.
_SEATABLE_AUTH_ERRORS = ("AuthExpiredError", "BaseUnauthError")


def _is_seatable_auth_error(exc: BaseException) -> bool:
    """True if exc is (or derives from) one of seatable's auth exceptions."""
    return any(cls.__name__ in _SEATABLE_AUTH_ERRORS for cls in type(exc).__mro__)


def classify_auth_error(service_key: str, exc: BaseException) -> str | None:
    """Classify an exception raised while connecting to a service.

    Returns ``"missing"`` (no credentials at all), ``"invalid"`` (credentials
    present but rejected) or ``None`` if this is not an auth problem.
    """
    msg = str(exc)

    if _http_status(exc) in (401, 403):
        return "invalid"

    if service_key == "neuprint":
        if isinstance(exc, RuntimeError):
            if _contains(msg, "No token provided"):
                return "missing"
            if _contains(msg, "Did not understand token"):
                return "invalid"
        return None

    if service_key == "clio":
        if _contains(msg, "No Clio token"):
            return "missing"
        if _contains(msg, "identity token", "long-lived Clio token"):
            # gcloud is installed but could not produce a usable token.
            return "missing"
        if _contains(msg, "Clio token not valid", "Did not understand token"):
            return "invalid"
        if _contains(msg, "Clio token is empty"):
            return "missing"
        return None

    if service_key == "flytable":
        if isinstance(exc, ValueError) and _contains(
            msg, "SEATABLE_TOKEN", "SEATABLE_SERVER"
        ):
            return "missing"
        if _is_seatable_auth_error(exc):
            return "invalid"
        if _seatable_status(exc) in (401, 403):
            return "invalid"
        return None

    return None


def raise_for_auth_error(service_key: str, exc: BaseException) -> None:
    """Re-raise ``exc`` as a MissingCredentialsError if it is an auth failure.

    Returns quietly when the exception is unrelated to authentication, so the
    caller can re-raise the original.
    """
    reason = classify_auth_error(service_key, exc)
    if reason is not None:
        raise MissingCredentialsError(
            service_key, invalid=(reason == "invalid")
        ) from exc
