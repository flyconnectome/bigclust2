# Command line

```
bigclust2 [--version] [--debug] [--from DATASET] [--filters EXPR] [--embedding MODE]
```

BigClust is a GUI application; the command line exists to launch it, optionally
with a project already open.

!!! tip "Let the app write the command for you"

    With a project open, **Help → Command** shows the exact `bigclust2 --from …
    --filters …` line that re-opens the current view, with copy buttons for both
    the `uvx` and installed forms. Copy it to bookmark or share a view.

## `--from DATASET`

Load a project on start-up, skipping the Open Project dialog.

```bash
bigclust2 --from /path/to/my_clustering
bigclust2 --from https://example.org/clusterings/hemibrain_v1.2
```

Takes a local path or an `http://` / `https://` URL. `gs://` and `s3://` are
**not** supported — see [loading a remote
dataset](../how-to/load-a-remote-dataset.md#what-remote-means).

If the path resolves to a directory holding several projects, you get a **Select
Project** picker before the main window appears. If it cannot be loaded you get a
**Load Error** dialog.

!!! note "The flag is `--from`, not `--dataset`"

    A project directory can hold several datasets, so `--from` names the *source*
    rather than one dataset within it.

## `--filters EXPR`

Apply a filter expression on start-up — the same expression the **Open Project**
dialog's filter field accepts. Requires `--from`.

```bash
bigclust2 --from /path/to/my_clustering --filters 'dataset == "hemibrain"'
```

Quote the expression so your shell passes it through intact. See [filtering on
load](../how-to/load-a-remote-dataset.md) for the expression syntax.

## `--embedding MODE`

Recompute the embedding on load instead of using the precomputed one. Requires
`--from`. Accepts `calculate from distances` or `calculate from features`.

```bash
bigclust2 --from /path/to/my_clustering --embedding 'calculate from distances'
```

Omit it (the default) to use the project's precomputed embedding.

## `--debug`

Turn on debug logging, for BigClust and for the root logger.

```bash
bigclust2 --debug
```

Use this when [reporting a problem](troubleshooting.md#reporting-a-problem).
Combine it with **Help → Debug → Tracebacks** to get full stack traces rather
than one-line messages.

Credentials are never logged, at any level.

## `--version`

```bash
bigclust2 --version
```

```
bigclust2 0.1.1
```

Prints and exits before any GUI or GPU code runs, which makes it the fastest way
to confirm an install resolved.

## Running without installing

```bash
uvx bigclust2@latest --from /path/to/my_clustering
```

Flags pass straight through `uvx`. See [installation](../get-started/installation.md).

## Running from a clone

```bash
uv run run.py --debug
```

!!! warning "`run.py` only understands `--debug`"

    `run.py` is the development entry point and does not use the argument parser —
    it looks for `--debug` in `sys.argv` and ignores everything else. `--from`,
    `--filters`, `--embedding` and `--version` do nothing there.

    Use the real entry point if you need them:

    ```bash
    uv run bigclust2 --from /path/to/my_clustering
    ```
