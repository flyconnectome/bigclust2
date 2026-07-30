# Command line

```
bigclust2 [--version] [--debug] [--from DATASET] [--filters EXPR] [--embedding MODE]
bigclust2 update PROJECT [--auto-map] [--dataset NAME] [--dry-run] [--no-save-sources]
```

BigClust is a GUI application; the command line exists to launch it, optionally
with a project already open. The exception is [`update`](#update-project), which
refreshes a project's meta data on disk without opening a window.

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
bigclust2 0.2.0
```

Prints and exits before any GUI or GPU code runs, which makes it the fastest way
to confirm an install resolved.

## `update PROJECT`

Refresh a project's meta data from the [sources it
declares](data-format.md#meta-sources) and save it. This is the headless
equivalent of **Meta Data Explorer → Update → From Remote…** followed by
**Export → Project**, except that it writes back into the project you point it
at rather than to a copy.

```bash
bigclust2 update /path/to/my_clustering
```

```
  [1/2] hemibrain…
  [2/2] flywire…
  hemibrain: 1,204 values changed in 806 rows (label, soma_side); 3 ids not in the source
  flywire: 812 values changed in 812 rows (label)
Updated 2,016 values in 1,618 rows across 2 datasets. Wrote meta.parquet and info.
```

!!! note "Values, not neurons"

    A *value* is one row/column intersection, so a neuron whose `label` and
    `soma_side` both moved counts twice. That is why the value count can exceed
    the number of neurons in a dataset — the row count next to it is the number
    of neurons actually touched.

    A value only counts as changed if it differs from what is already stored
    *after* being converted to the column's type. A source reporting `"111"`
    for a column holding the integer `111` changes nothing and is not counted.

    A **column your project maps but does not have yet** is created and filled
    for every matched neuron at once, which can be tens of thousands of values
    on the first run and nothing thereafter. That share is reported separately:

    ```
    Updated 118,934 values in 112,868 rows across 3 datasets. 112,831 of those
    populated the new column soma_side.
    ```

    So the edits to data you already had are the difference — about 6,100 here.

The rows never move: same count, same order, same index — only cell values
change, plus any mapped column that did not exist yet. That matters because the
`embeddings`, `features` and KNN files are matched to `meta` [by row
position](data-format.md), so a refresh that reordered rows would silently
corrupt the project.

Writes are atomic — a temp file renamed over the target, so an interrupted run
cannot leave half a table behind. `info` is always written: its `last_updated`
stamps are set to today for the meta block and for every source that succeeded,
which is what records that the snapshot was checked. The meta table itself is
only rewritten when a value actually changed, so a nightly run over a project
nobody has re-annotated costs nothing but the read.

The **sources are only ever read from**. Writing to a backend is [pushing
annotations](../how-to/push-annotations.md), which this command cannot do.

!!! warning "A project whose meta table is a URL cannot be updated"

    There is nothing local to write back to, and the command says so instead of
    pulling anything. The same goes for a directory that holds [several
    projects](data-format.md#multiple-projects) — point at one of them.

### `--auto-map`

Map project columns onto source columns by name for every column the project
does not already map — `soma_side` onto `somaSide`, and so on. Mappings already
stored in `info` always win, so this only ever fills gaps.

```bash
bigclust2 update /path/to/my_clustering --auto-map
```

What it matched is stored back into `info`, so the next refresh does not have to
guess again. Pass `--no-save-sources` to keep the mapping for this run only.

Without this flag, a source that maps no columns updates nothing, and the run
tells you so.

### `--dataset NAME`

Only refresh this dataset. Repeat for several. Sources for datasets you do not
name are left alone — both their rows and their `info` entries.

```bash
bigclust2 update /path/to/my_clustering --dataset hemibrain --dataset flywire
```

### `--dry-run`

Pull and merge, but write nothing. You get the same per-dataset report a real run
would print, which is the cheapest way to find out that a mapping is wrong before
it lands in your project.

```bash
bigclust2 update /path/to/my_clustering --auto-map --dry-run
```

### Credentials

Backends that need a token read it from the environment
(`NEUPRINT_APPLICATION_CREDENTIALS`, `SEATABLE_TOKEN`) — or from the tokens you
stored once in the app under **Window → Credentials**, which this command picks
up too. See [credentials](../get-started/credentials.md).

Nothing is ever prompted for: a source whose credentials are missing fails, is
reported, and the other datasets still update.

### Exit codes

`0` if every source updated; `1` if any of them failed (including a failure to
inspect a source for `--auto-map`), or if the project could not be updated at
all. A partial failure still writes the datasets that worked.

### From Python

The same thing, if you would rather script it:

```python
import bigclust2

result = bigclust2.update_project_meta(
    "/path/to/my_clustering",
    auto_map_columns=True,
    datasets=["hemibrain"],   # optional; default is every declared source
    dry_run=False,
)
print(result.summary())
print(result.meta.head())     # the refreshed table
```

Credentials work slightly differently here: the function reads tokens from the
environment only. Picking up the ones stored in the app is something `bigclust2
update` does for you, since that is the command standing in for the app.

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
