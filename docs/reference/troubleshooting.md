# Troubleshooting

## Installing and starting

| Symptom | Cause and fix |
|---|---|
| `uvx` fails with `realpath: command not found` | macOS older than 13.x. Upgrade the OS — this is a `uv` bootstrap issue. |
| `uvx bigclust2@latest` is slow every time | `@latest` re-resolves whenever a new release exists. [Pin a version](../get-started/installation.md#why-latest). |
| The window never appears, no error | Usually a graphics-stack problem. Run with `--debug` and look for wgpu or WebGPU messages. BigClust needs a real GPU and a display — it will not run headless. |
| `--from` is ignored | You are running `run.py`, which only understands `--debug`. Use `uv run bigclust2 --from …`. See [the CLI reference](cli.md#running-from-a-clone). |

## Opening a project

| Message | Cause and fix |
|---|---|
| `Path does not exist` | Local path is wrong. Relative paths resolve against your current working directory. |
| `No info file found in directory` | The directory has no `info` at its root. Check you are not pointing one level too deep — `…/my_clustering/meta.parquet` will not work. |
| `Malformed info file. The 'info' file must contain valid JSON.` | The `info` file is not valid JSON. Most often this is a copied example containing `#` comments — [JSON has none](data-format.md#the-info-file). Validate with `python -m json.tool < info`. |
| An HTTP error on a URL | Check `curl -I <url>/info` first. Also confirm the server sets CORS headers if you are serving from a different origin. |
| A `gs://` or `s3://` URL fails with a schema error | Not supported. [Serve over HTTPS or sync locally](../how-to/load-a-remote-dataset.md#what-remote-means). |
| A `KeyError` on `id` after loading | Your `meta` table is missing a required column. There is no upfront validation — see [the meta columns](data-format.md#columns). |

### Loads fine, but filtering fails

```
Distance file must have the 'id', 'index' or '__index_level_0__' column as the
first or last column for correct filtering
```

Your distance (or feature) matrix has its ID column in the middle. This only
surfaces when a [load-time filter](../how-to/filter-a-dataset.md) is applied,
because that is the only time the matrix has to be subset by ID.

Fix it on the data side — write the DataFrame with the IDs as the index:

```python
dist.to_parquet("distances.parquet")   # index preserved as __index_level_0__
```

See [the id-column trap](data-format.md#distances).

## Using the app

| Symptom | Cause and fix |
|---|---|
| Selecting is slow | Every selected neuron is a mesh to fetch and render. Untick **View → Synchronize Viewer**. See [the cost of a big selection](../concepts/selection.md#the-cost-of-a-big-selection). |
| MDS / PaCMAP / k-means / agglomerative are greyed out | Your project supplies a [KNN graph](data-format.md#knn-graph) rather than a full distance matrix. Those methods need all pairwise distances. |
| The Embeddings tab has no data source | The project ships only `embeddings` — there is nothing to recompute *from*. See [the data model](../concepts/data-model.md#why-this-decides-what-the-app-can-do). |
| The Distance Heatmap is disabled | Same reason — KNN-graph projects cannot show a full heatmap. |
| Feature Subset shows no checkboxes | Your `features` file does not use MultiIndex columns. See [features](data-format.md#features). |
| Ward linkage raises an error | Ward is only defined for Euclidean distances. Use `average` with a precomputed matrix. [Details](../how-to/recluster.md#agglomerative). |
| Everything is noise (`-1`) after clustering | Either `Min cluster size` is too large, or you are hitting the [singleton rule](../how-to/recluster.md#singletons-become-noise). |
| A pasted ID list is rejected | The bare-list form is [digits-only](../how-to/filter-a-dataset.md#filter-syntax). Use `id in ("abc", "def")`. |
| Project Details says `has features: False` but features work | It only inspects the top level of `info`. [Known limitation](widgets.md#other-dialogs). |
| The plot is sluggish on battery | Set **Settings → Render trigger** to `Reactive`. |
| Neurons show as skeletons, not meshes | The source has no meshes, or those segments were never meshed — BigClust [falls back to skeletons](data-format.md#meshes-vs-skeletons). Run with `--debug` to see which fetch failed. |
| A `skeletons://` source loads nothing | Files must be named after their segment ID (`12345.swc`). Anything else is [ignored](data-format.md#local-skeletons). |
| `Skeleton source must point to an existing directory` | The path after `skeletons://` is wrong. Relative paths resolve against the *project* directory, and only for local projects. |
| "Open in Neuroglancer" drops some neurons | Layers whose only source is local (`skeletons://`, `memory`) cannot be shared — Neuroglancer has no way to reach your filesystem. |

## Credentials and backends

| Symptom | Cause and fix |
|---|---|
| `No <service> token found` | You have not supplied one. [Set it](../get-started/credentials.md). |
| `Your <service> token was rejected (invalid or expired)` | The token exists but no longer works. Get a fresh one and use **Update…**. |
| Credentials dialog says `Set (from environment)` after **Clear** | The value comes from a variable you exported in your shell. The dialog cannot unset that. |
| **Backend validation failed** | The configuration or the credentials are wrong for that backend. The message names which. |
| neuPrint is not offered as a push target | It is [read-only](backends.md#neuprint). |
| Writing `type` in Clio also changed `instance` | [`auto_fix_instances`](backends.md#clio) is on. |
| A push half-succeeded | Check **Window → Show Annotation Log** before retrying — the batches that succeeded are already written. |

## Data that disappeared

| Symptom | Explanation |
|---|---|
| Refreshed meta data is gone after restart | Refreshes live in memory. **Export → Project** to persist. See [persisting a refresh](../how-to/update-meta-data.md#persisting-a-refresh). |
| A clustering is gone after restart | Session state. Cluster tab → **Export**, and load it back with the folder button. |
| A recomputed embedding is gone after restart | Session state, and it is not included in a project snapshot. Note your parameters and the `Random seed:` to reproduce it. |

None of these are bugs — BigClust never modifies your project directory in place.
[Why](../concepts/data-sources.md#live-data-stays-live).

## Reporting a problem

Issues go to <https://github.com/flyconnectome/bigclust2/issues>. **Help → Report a
Problem** opens the same page.

Before you file, please:

1. Turn on **Help → Debug → Tracebacks**.
2. Restart with `--debug`:
   ```bash
   uvx bigclust2@latest --debug
   ```
3. Reproduce the problem and copy the console output.

Include the output of `bigclust2 --version`, your operating system, and — if the
problem is with a specific project — the `info` file, which is normally small
enough to paste. **Project Details** shows it if you do not have the file to hand.

Credentials are never written to the log, so the output is safe to paste as-is.
