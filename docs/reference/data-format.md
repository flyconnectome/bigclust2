# Data format

A BigClust project is a directory containing an `info` file plus the data files it
names. This page is the complete specification; if you would rather read a worked
example first, [create a local dataset](../how-to/create-a-local-dataset.md)
builds one end to end.

```
/my_clustering/
    info                <- JSON: what is here and how to read it
    meta.parquet        <- per-neuron metadata                   (required)
    embeddings.parquet  <- 2D coordinates for the scatter plot   (optional)
    distances.parquet   <- pairwise distances                    (optional)
    features.parquet    <- high-dimensional features             (optional)
```

Data files may be **Parquet** (recommended), **Apache Arrow Feather**, **CSV** or
**JSON**. Parquet is preferred throughout: compact, self-describing, columnar,
and it round-trips dtypes and MultiIndex columns properly.

!!! danger "Everything is aligned by row position"

    `meta`, `embeddings`, `features` and the KNN graph are matched by **row
    order**, not by joining on `id`. Every file must list observations in the same
    order as `meta`.

    Re-sorting one file independently does not raise an error. It silently
    associates the wrong coordinates and features with every neuron. If you take
    one thing from this page, take this.

## The `info` file

JSON, no extension. Only `meta` is validated as required — but a project with no
`embeddings` has no coordinates to draw, so in practice you want both.

```json
{
    "name": "my_clustering",
    "dataset": "hemibrain_v1.2",
    "description": "My clustering dataset",
    "version": "0.1",
    "date_created": "2024-06-01",
    "meta": {
        "file": "meta.parquet",
        "color": "color",
        "last_updated": "2024-06-01"
    },
    "embeddings": {
        "file": "embeddings.parquet"
    },
    "distances": {
        "type": "connectivity",
        "file": "distances.parquet",
        "metric": "cosine"
    },
    "features": {
        "type": "connectivity",
        "file": "connections.parquet"
    }
}
```

!!! warning "JSON has no comments"

    Earlier versions of this document annotated the example above with `#`
    comments. Those examples are **not valid JSON** and a copy-pasted one fails to
    load with `Malformed info file. The 'info' file must contain valid JSON.`

    Everything on this page is valid as written. Field meanings are in the tables,
    not in comments.

### Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `name` | no | Project name. Used in window titles and default export filenames. |
| `dataset` | no | A label for the underlying dataset, for your own reference. |
| `description` | no | Free text, shown in **Project Details**. |
| `version` | no | Your version string. |
| `date_created` | no | Free-form date. |
| `meta` | **yes** | The per-observation table. See [below](#meta). |
| `embeddings` | no[^req] | 2D coordinates, or a list of them. See [below](#embeddings). |
| `distances` | no | Pairwise distances or a KNN graph. See [below](#distances). |
| `features` | no | High-dimensional features. See [below](#features). |
| `neuroglancer` | no | 3D viewer configuration. See [below](#neuroglancer). |

[^req]: Only `meta` is actually enforced by the loader. A project without
    `embeddings` loads without error but has nothing to plot, so treat it as
    required in practice.

## `meta`

The per-observation table — one row per neuron.

```json
"meta": {
    "file": "meta.parquet",
    "color": "color",
    "last_updated": "2024-06-01"
}
```

| Field | Meaning |
|---|---|
| `file` | Path to the table, relative to the project directory, or a URL. |
| `color` | How to colour points. See [below](#colour). |
| `last_updated` | When the snapshot was taken. Drives the staleness banner. |
| `sources` | Live backends to refresh from. See [meta sources](#meta-sources). |

### Columns

| Column | Required | Meaning |
|---|---|---|
| `id` | **yes** | Unique identifier per observation. |
| `label` | **yes** | The labels drawn on the scatter plot. |
| `dataset` | **yes** | Which dataset the row belongs to. Drives point markers, Neuroglancer source selection, and how annotations are routed. |
| `color` | no | Per-row colour. Must be named in `info` to be used. |
| `source` | no | Where to load morphology from, per row. See [`neuroglancer`](#neuroglancer) for the accepted source schemes. |
| `x` / `y` | no | Coordinates, if you are not shipping a separate `embeddings` file. |

Any additional columns are loaded and become available for colouring, labelling,
filtering, hover info and scope filters. Ship everything you have — extra columns
cost almost nothing and each one is a way to interrogate the embedding.

!!! note "The required columns are required by convention, not by validation"

    Nothing checks for `id`, `label` and `dataset` at load time. A table missing
    one of them loads and then fails later, with a `KeyError` rather than a
    helpful message. Check your table before shipping it.

### Colour

`meta.color` accepts three forms:

| Form | Example | Meaning |
|---|---|---|
| A column name | `"color": "color"` | Use that column's per-row values |
| A dataset map | `"color": {"HbL": [0, 0, 255], "HbR": [255, 0, 0]}` | One colour per dataset |
| A single colour | `"color": "#1f77b4"` | Everything the same colour |

Anything [`cmap`](https://github.com/tlambert03/cmap) understands is a valid
colour — hex strings, named colours, `[r, g, b]` lists. The default is white.

### Meta sources

Optional. Declares where each dataset's annotations can be **read** from, so the
snapshot can be refreshed in-app. Never written to.

```json
"meta": {
    "file": "meta.parquet",
    "sources": {
        "hemibrain": {
            "backend": "neuPrint",
            "config": { "dataset": "hemibrain:v1.2.1" },
            "columns": { "label": "type", "soma_side": "somaSide" },
            "last_updated": "2026-06-14"
        }
    }
}
```

| Field | Meaning |
|---|---|
| `backend` | A [registered backend](backends.md) name — `neuPrint`, `Clio`, `FlyTable`, `FlyWire @ FlyTable`, `Hemibrain @ FlyTable`, `CSV`. |
| `config` | That backend's configuration fields. Omitted fields take their defaults. |
| `columns` | Maps **your** meta column → the **source's** column name. A mapped column that does not exist yet is created on refresh. |
| `last_updated` | Set automatically on each successful refresh. |

`id` and `dataset` are never mapped or updated, and neither is any column whose
name starts with `_`.

Several datasets sharing one setup can use a **comma-separated key**, or an
explicit `datasets` list in the value:

```json
"sources": {
    "brain_left,brain_right": {
        "backend": "neuPrint",
        "config": { "dataset": "..." },
        "columns": { "label": "type" }
    }
}
```

Each listed dataset is still matched by its own IDs. Saving from the dialog
regroups identical setups back into one comma-keyed entry.

Refreshed values live in the session. Use **Export → Project** to persist them.
See [refreshing meta data](../how-to/update-meta-data.md).

## `embeddings`

The 2D coordinates the scatter plot draws.

```json
"embeddings": { "file": "embeddings.parquet" }
```

or, if the coordinates are columns of the `meta` table:

```json
"embeddings": { "columns": ["x_coord", "y_coord"] }
```

!!! warning "Exactly two columns"

    An embeddings file must have **exactly two** columns. Three or more raises
    `Embeddings must have exactly 2 dimensions`. If your file carries an ID column
    alongside the coordinates, drop it or write it as the index.

`float32` is plenty for 2D scatter coordinates.

### Multiple embeddings

Make `embeddings` a **list**. Each entry may carry its own `features` and/or
`distances`, which override the top-level ones for that entry.

```json
"embeddings": [
    {
        "name": "connectivity (UMAP)",
        "file": "embeddings_conn.parquet",
        "features": { "file": "connections.parquet", "type": "connectivity" },
        "distances": { "file": "distances.parquet", "metric": "cosine" }
    },
    {
        "name": "morphology (NBLAST)",
        "file": "embeddings_morph.parquet",
        "distances": { "file": "nblast.parquet" }
    },
    {
        "name": "raw layout",
        "columns": ["x", "y"]
    }
]
```

Entries without a `name` get `embedding 1`, `embedding 2`, … The first entry is
active on load.

Features and distances must always belong to an embedding. A project that
declares `features` or `distances` with **no** `embeddings` at all has them
ignored, with a warning in the log.

In the app, switch with the dropdown at the top of the **Embeddings** tab or press
++space++ to cycle. See [the data model](../concepts/data-model.md#multiple-embeddings)
for why this is useful.

## `distances`

A square, symmetric matrix of pairwise distances.

```json
"distances": {
    "type": "connectivity",
    "file": "distances.parquet",
    "metric": "cosine"
}
```

| Field | Meaning |
|---|---|
| `file` | Path or URL. |
| `type` | Free-form label describing what the distances are of. Shown in the UI. |
| `metric` | The metric used. Informational. |

Shape must be `(n_observations, n_observations)`, with index and column names
being the IDs, in the same order as `meta`.

**Store values as `float32`.** A distance matrix grows with the square of your
observation count and is normally the largest file in a project; `float32` halves
it with no practical loss.

Store the full matrix even if it is symmetric — BigClust expects square, not
triangular.

!!! warning "The ID column must be first or last"

    When a [load-time filter](../how-to/filter-a-dataset.md) is applied, BigClust
    subsets the matrix by row and column, and to do that it must locate the ID
    column. It looks only at the **first and last** columns, and only for the
    names `id`, `index` or `__index_level_0__`.

    A matrix with its ID column in the middle loads fine unfiltered and fails the
    moment a filter is set. Writing a DataFrame whose index holds the IDs with
    `df.to_parquet(...)` produces the right layout — pyarrow preserves the index
    as `__index_level_0__`.

    The same requirement applies to `features` when filtered.

### KNN graph

For large projects, supply a k-nearest-neighbours graph instead of a full matrix.
It is dramatically smaller and loads much faster.

```json
"distances": {
    "type": "knn",
    "file": "knn.parquet",
    "metric": "cosine"
}
```

`type` may be prefixed with the source the neighbours came from — `"nblast:knn"` —
for your own bookkeeping; it is still recognised as a KNN graph.

The file is a **wide** table, one row per observation, in the same order as
`meta`. For `k` neighbours:

| Columns | Contents |
|---|---|
| `nn_idx_1` … `nn_idx_k` | **0-based row indices** into `meta` — the neighbour's *position*, not its `id`. Nearest first. Store as `int32`. |
| `nn_dist_1` … `nn_dist_k` | Distance to the corresponding neighbour. Store as `float32`. |

`k` is inferred from the number of `nn_idx_*` columns. Row indices rather than IDs
keeps the graph unambiguous even when a project contains duplicate IDs. Neighbours
that fall outside a filtered subset are dropped automatically; `metric` is
informational.

**What a KNN graph costs you.** A red note appears in the affected tabs:

| Feature | With a KNN graph |
|---|---|
| Distance heatmap | Disabled |
| Recompute embedding | **UMAP** and **t-SNE** only. MDS and PaCMAP, the feature metric, rebalancing and PCA are disabled. `n_neighbors` and `perplexity` are capped to what `k` supports. |
| Clustering | **HDBSCAN** and **Spectral** only. Agglomerative, k-means and **Find k** need the full matrix or feature vectors. |
| Neighborhood fidelity, KNN edges | Computed from the graph, capped at `k` |

See [the data model](../concepts/data-model.md#why-this-decides-what-the-app-can-do).

## `features`

The high-dimensional description of each observation.

```json
"features": {
    "type": "connectivity",
    "file": "connections.parquet"
}
```

Shape is `(n_observations, n_features)`, with the index matching the `meta` `id`
order.

**Use MultiIndex columns** to group features — upstream versus downstream
connections, isomorphic versus dimorphic partners. Each top level becomes a
checkbox under **Feature Subset** in the Embeddings tab and a filter in the
Feature Comparison widget, which is how you ask "lay these neurons out by their
inputs only". Build the DataFrame with a `pandas.MultiIndex` before writing;
pyarrow preserves it. Read the file back and check `df.columns` is still a
MultiIndex.

Feature matrices are usually wide and sparse. Store counts as the smallest dtype
that fits.

## `neuroglancer`

Configures the 3D viewer.

```json
"neuroglancer": {
    "source": {
        "HbR": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
        "HbL": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation"
    },
    "neuropil_mesh": "https://github.com/navis-org/navis-flybrains/raw/refs/heads/main/flybrains/meshes/JRCFIB2018Fraw.ply",
    "color": {
        "HbR": [255, 0, 0],
        "HbL": [0, 0, 255]
    }
}
```

| Field | Accepts |
|---|---|
| `source` | A single source string, a `meta` column name, or a `{dataset: source}` map |
| `neuropil_mesh` | A local path, a URL, or `id@source` for a mesh from a Neuroglancer source |
| `color` | A single colour, a `meta` column name, or a `{dataset: colour}` map |
| `transforms` | On-the-fly mesh transforms. See below. |

Note this is where `gs://` and `s3://` are fine — they are handed to
`cloud-volume`, not to BigClust's own loader. The [HTTP-only
restriction](../how-to/load-a-remote-dataset.md#what-remote-means) applies to the
*project directory*, not to Neuroglancer sources.

### Source schemes

| Scheme | Serves | Notes |
|---|---|---|
| `precomputed://` | meshes, skeletons | Handed to `cloud-volume`. Wraps `gs://`, `s3://`, `https://`, `file://` |
| `graphene://` | meshes | Handed to `cloud-volume` |
| `dvid://` | meshes | Falls back to a remote meshing service for unmeshed bodies |
| `skeletons://` | skeletons | A **local** directory of SWC files. See below |
| `memory` | whatever you put there | For neurons handed to the viewer programmatically |

### Meshes vs. skeletons

Neurons render as meshes where meshes are available and as skeletons
(`navis.TreeNeuron`) otherwise. BigClust always asks a source for a mesh first
and only falls back to its skeletons if the source has none, or if that
particular segment has not been meshed. A skeleton is drawn as its neurites plus
a soma marker, and behaves like a mesh everywhere else — colouring, selection,
caching and transforms all apply unchanged.

### Local skeletons

`skeletons://` serves skeletons from a directory of SWC files on this machine:

```json
"neuroglancer": {
    "source": "skeletons://skeletons/"
}
```

Each file must be named after the segment ID it contains — `12345.swc` serves the
neuron with `id` 12345. Both `.swc` and `.swc.gz` are read; files whose name is
not a plain number are ignored, as they cannot be addressed by ID. The directory
is indexed once, on the first neuron you load.

The path may be absolute, start with `~`, or — as above — be relative to the
project directory. Relative paths are only resolved for local projects.

!!! note "Not shareable"

    A `skeletons://` source points at your filesystem, so "Open in Neuroglancer"
    cannot include it. Layers whose only source is local are left out of the
    generated scene.

### Transforms

Bring datasets in different spaces into a common one when meshes are loaded.

```json
"transforms": [
    {
        "apply_to": "HbL",
        "type": "landmarks",
        "file": "landmarks.csv",
        "source_cols": ["x1", "y1", "z1"],
        "target_cols": ["x2", "y2", "z2"]
    }
]
```

| Field | Meaning |
|---|---|
| `apply_to` | Dataset name, a list of them, or a comma-separated string (`"HbL,HbR"`) |
| `type` | Only `landmarks` (thin-plate spline) is supported; anything else raises |
| `file` | CSV of landmark coordinates — project-relative path or URL |
| `source_cols` / `target_cols` | The x/y/z columns of the native and common spaces. Meshes map source → target. |

## Multiple projects

A directory whose `info` file is a **list of relative paths** is a collection:

```
/projects/
    info
    /clustering_1/
        info
        ...
    /clustering_2/
        info
        ...
```

```json
[
    "clustering_1",
    "clustering_2"
]
```

Point BigClust at `/projects/` and you get a project picker.

## Writing good Parquet

### General

- **Prefer Parquet over Feather** for distribution. Feather is fine as local
  scratch; Parquet compresses better and is more portable across versions.
- **Write with `pyarrow`** — `df.to_parquet(..., engine="pyarrow")`. Avoid the
  `fastparquet` engine for matrices with non-string column names; it handles them
  inconsistently.
- **Compress.** `snappy` (the pandas default) is a good speed/size trade-off;
  `zstd` gives the smallest files for slightly slower writes. Both decompress
  fast.
- **Keep row order consistent** across every file. See the warning at the top of
  this page.

### Per file

| File | Notes |
|---|---|
| `meta` | Store `id` as an integer where possible — smaller, faster, and it avoids whitespace and casing mismatches. Keep `label` and `dataset` as plain strings (`category` dtype works and survives Parquet, but strings work everywhere). Use one consistent colour representation across all rows. Don't write a meaningless index. |
| `distances` | `float32`. Index and columns must be the IDs — `df.to_parquet(...)` on a DataFrame indexed by ID does the right thing. Full square matrix, not a triangle. |
| `embeddings` | Exactly two float columns; `float32` is plenty. Name them clearly (`x`/`y`) and reference them with `"columns": [...]` if they live in `meta`. |
| `features` | Smallest dtype that fits. Build MultiIndex columns before writing and verify they round-trip. |
| KNN | `nn_idx_*` as `int32` (they are row positions, not floats), `nn_dist_*` as `float32`. Keep the columns in order. |

### Verify before shipping

```python
import pandas as pd

meta = pd.read_parquet("meta.parquet")
dist = pd.read_parquet("distances.parquet")
feat = pd.read_parquet("features.parquet")

# Shapes and dtypes
print(meta.shape, dist.shape, feat.shape)
print(dist.dtypes.unique())                    # expect float32

# Required columns present
assert {"id", "label", "dataset"} <= set(meta.columns)

# Row order consistent - the failure this catches is silent otherwise
assert list(dist.index) == list(meta["id"])
assert len(feat) == len(meta)

# Distances square and symmetric
assert dist.shape[0] == dist.shape[1]

# MultiIndex features survived the round trip
print(type(feat.columns), feat.columns[:3])
```

The row-order assertions are the important ones. Everything else fails loudly on
load; that one does not fail at all.
