# Credentials

You need tokens for exactly two things: [refreshing meta data from a live
backend](../how-to/update-meta-data.md) and [pushing annotations
back](../how-to/push-annotations.md). Opening a project, reclustering, exploring
and exporting all work with no credentials at all.

## The three services

| Service | Environment variable | Where the token lives | Get one from |
|---|---|---|---|
| **neuPrint** | `NEUPRINT_APPLICATION_CREDENTIALS` | Qt settings + the environment | <https://neuprint.janelia.org/account> |
| **FlyTable** (SeaTable) | `SEATABLE_TOKEN` | Qt settings + the environment | your FlyTable server's account page |
| **Clio** | — | `~/clio_token.json`, via `clio-py` | <https://clio.janelia.org/settings> |

Clio is the odd one out: it has no environment variable, because `clio-py`
manages its own token file. BigClust hands the token to `clio.set_token()` and
lets it write the file.

## Three ways to supply them

=== "In the app"

    **Window → Credentials…** (on macOS this lands under **BigClust →
    Preferences**, because it carries the standard preferences menu role).

    You get one row per service showing its current status, with **Update…** and
    **Clear** buttons. Tokens entered here are stored in Qt's settings and
    injected into the process environment immediately — no restart needed.

    This is the right choice for day-to-day use.

=== "In your shell"

    ```bash
    export NEUPRINT_APPLICATION_CREDENTIALS="eyJhbGciOi..."
    export SEATABLE_TOKEN="..."
    uvx bigclust2@latest
    ```

    BigClust reads these on start-up and never overwrites them. The Credentials
    dialog will show `Set (from environment)` for anything it found this way.

    This is the right choice on a shared machine, or when you already export
    these for other tools.

=== "When asked"

    If a backend needs a token you have not supplied, BigClust asks for it at the
    moment it is needed, with a link to the page that issues it. Paste it in and
    the operation continues.

    The prompt distinguishes *missing* from *rejected*: `No neuPrint token found`
    means you never supplied one, `Your neuPrint token was rejected (invalid or
    expired)` means the one you supplied no longer works.

Saved tokens are applied once at start-up, before any backend is constructed. The
step is idempotent and only fills in variables that have a *stored* token, so a
variable you exported yourself is never clobbered by a stale saved value.

!!! note "Clio can fall back to `gcloud`"

    If `gcloud` is on your `PATH` and authenticated, Clio can use it instead of a
    stored token. The Credentials dialog says `Not set (gcloud fallback
    available)` when it detects this — that is not an error, and you may not need
    to do anything.

## Where tokens are stored

Tokens entered through the dialog go into `QSettings` under the organisation
`bigclust2` and application `Credentials`. In practice that means:

| Platform | Location |
|---|---|
| macOS | `~/Library/Preferences/com.bigclust2.Credentials.plist` |
| Linux | `~/.config/bigclust2/Credentials.conf` |
| Windows | registry, under `HKEY_CURRENT_USER\Software\bigclust2\Credentials` |

This is **not** encrypted storage. It is the same place the rest of your Qt
application settings live, protected by nothing more than your user account's
file permissions. If that is not acceptable for your threat model, export the
environment variables from a shell that reads them out of a real secret manager
and leave the dialog empty.

Token values are never written to the log, at any log level, including
`--debug`.

## Clearing a token

**Window → Credentials… → Clear** removes the stored token and asks first
(`Remove the stored neuPrint token?`).

Clearing does *not* unset an environment variable you exported yourself — the
dialog cannot reach into your shell. If the status still says `Set (from
environment)` after clearing, that is where it is coming from.

Next: [how-to guides](../how-to/index.md), or back to [your first
project](first-project.md).
