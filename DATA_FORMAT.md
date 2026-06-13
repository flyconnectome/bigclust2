# Data Format

Example directory structure for a single dataset:

```
/my_clustering/
    info                <- JSON-formatted settings for the dataset
    meta.parquet        <- per-neuron metadata in Parquet (recommended) or Apache Arrow Feather format
    distances.parquet   <- pairwise distances in Parquet (recommended) or Apache Arrow Feather format (optional)
    embeddings.parquet  <- low-dimensional embeddings in Parquet (recommended) or Apache Arrow Feather format (optional)
    features.parquet    <- high-dimensional features in Parquet (recommended) or Apache Arrow Feather format (optional)
```

## `info` File (required)

The `info` file contains information about the dataset, including which files are present and how to interpret them:

```json
{
    "name": "my_clustering",
    "dataset": "hemibrain_v1.2",            # specify dataset for easier reference
    "description": "My clustering dataset", # brief description
    "version": "0.1",
    "date_created": "2024-06-01",
    # required: metadata about each observation (e.g. neuron)
    "meta": {
        "file": "meta.parquet",  # must at least have `id`, `label` and `dataset` columns
        "source": "neuprint://https://neuprint.janelia.org@hemibrain:v1.2",  # define source for updating metadata
        "last_updated": "2024-06-01",  # meta data can be update independently of the other data
        "color": "color"  # optional: column in meta to use for colors
    },
    # required: precomputed low-dimensional embeddings for scatter plot
    # (this can also be a list of embeddings - see "Multiple Embeddings" below)
    "embeddings": {
        "file": "embeddings.parquet"  # alternatively: "columns": ["x_coord", "y_coord"]
    },
    # optional: settings for neuroglancer viewer
    "neuroglancer": {
        # this can either be a single source, a column in `meta` or a dictionary mapping dataset names to sources
        "source": {
            "HbR": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
            "HbL": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation"
        },
        # this can be a local file, a URL or a mesh from a neuroglancer source (id@source)
        "neuropil_mesh": "https://github.com/navis-org/navis-flybrains/raw/refs/heads/main/flybrains/meshes/JRCFIB2018Fraw.ply",
        # this can be a single color, a column in `meta` or a dictionary mapping dataset names to colors
        "color": {
            "HbR": [255, 0, 0],
            "HbL": [0, 0, 255]
        }
    },
    # optional: pairwise (square) distance matrix
    "distances": {
        "type": "connectivity",
        "file": "distances.parquet",
        "metric": "cosine"
    },
    # optional alternative to a full distance matrix for large projects:
    # a precomputed k-nearest-neighbors graph (see "KNN graph" below)
    # "distances": {
    #     "type": "knn",
    #     "file": "knn.parquet",
    #     "metric": "cosine"
    # },
    # optional: feature vector
    "features": {
        "type": "connectivity",
        "file": "connections.parquet",
    },
}
```

## `meta` File (required)

The metadata file contains information about each observation (e.g. neuron) in the dataset:
- `id` (required): unique identifier for each observation (e.g. neuron)
- `label` (required): labels to use for the scatter plot
- `dataset` (required): which dataset the observation belongs to (e.g. "hemibrain", "FAFB", etc.); this is used for markers in the scatter plot but also for neuroglancer sources and updating/pushing annotation
- `color` (optional): colors used for markers in the scatter plot and for meshes in the 3D viewer
- `source` (optional): URLs or identifiers for the source of the observation (used for loading morphology in the 3D viewer)
- `x` / `y` (optional): pre-computed low-dimensional embeddings for the scatter plot (if not provided in a separate `embeddings` file)

The optional columns have to be explicitly specified in the `info` file (see above) so that BigClust knows how to use them!

Any additional columns are loaded and can be used for coloring, filtering, etc. in the GUI.

## `distances` File (optional)

The distances file contains pairwise distances between observations (e.g. neurons) in the dataset. This is optional but can be used for computing fidelity metrics and for feature selection. The file should be in Parquet (recommended) or Apache Arrow Feather format and contain a square matrix of shape `(n_observations, n_observations)`. Index and column names should be IDs matching the order of the `id` column in the `meta` file.

## KNN graph (optional alternative to a full distance matrix)

For large projects a full `(n_observations, n_observations)` distance matrix is often prohibitive to compute, store and load. Instead you can provide a precomputed **k-nearest-neighbors graph** by giving the `distances` spec `type: "knn"`:

```json
"distances": {
    "type": "knn",
    "file": "knn.parquet",
    "metric": "cosine"
}
```

The `type` may also be prefixed with the source the neighbors came from, in the form `"<source>:knn"` (e.g. `"nblast:knn"`); the prefix is for your own bookkeeping/display and is still recognised as a KNN graph.

The KNN file is a **wide** table with one row per observation, in the **same order as the `meta` file**. For `k` neighbors it must contain:

- `nn_idx_1, nn_idx_2, …, nn_idx_k` — the **0-based row indices** of each neighbor into the `meta` file (i.e. the neighbor's position, not its `id`), ordered **nearest-first**
- `nn_dist_1, nn_dist_2, …, nn_dist_k` — the distance to each corresponding neighbor

`k` is inferred from the number of `nn_idx_*` columns. Using row indices (rather than IDs) keeps the graph unambiguous even when a project has duplicate neuron IDs. When the dataset is filtered, neighbors that fall outside the kept subset are dropped automatically. The `metric` field is informational only.

When a project supplies a KNN graph instead of a full distance matrix, BigClust limits the options that genuinely require all pairwise distances and shows a red note in the affected tabs:

- The **distance heatmap** is disabled.
- **Recompute embeddings** is limited to **UMAP** and **t-SNE** (which accept the graph directly); MDS/PaCMAP, the feature metric, rebalancing and PCA are disabled. UMAP's `n_neighbors` and t-SNE's `perplexity` are capped to what `k` supports.
- **Clustering** is limited to **HDBSCAN** and **Spectral** (Agglomerative, K-Means and automatic "Find k" require a full matrix or feature vectors).
- **Neighborhood fidelity** and KNN edge lines are computed directly from the graph (capped at `k`).

## `features` File (optional)

The features file contains high-dimensional features for each observation (e.g. neuron) in the dataset. This is optional but can be used for feature selection and for exploring the relationship between features and embeddings. The file should be in Parquet (recommended) or Apache Arrow Feather format and contain a matrix of shape `(n_observations, n_features)`. The index should be integers matching the order of the `id` column in the `meta` file. You can use Multi-Index columns to organize features into groups (e.g. upstream vs downstream connections).

## Multiple Embeddings

A single project can contain more than one embedding. To do so, make `embeddings` a **list** of entries instead of a single object. Each entry may carry its own `features` and/or `distances` (the high-dimensional sources paired with that embedding); embeddings without sources are allowed, but features/distances must always belong to an embedding. Per-entry `features`/`distances` override the top-level ones (which act as a fallback/default).

```json
{
    ...
    "embeddings": [
        {
            "name": "connectivity (UMAP)",
            "file": "embeddings_conn.parquet",
            "features": {"file": "connections.parquet", "type": "connectivity"},
            "distances": {"file": "distances.parquet", "metric": "cosine"}
        },
        {
            "name": "morphology (NBLAST)",
            "file": "embeddings_morph.parquet",
            "distances": {"file": "nblast.parquet"}
        },
        {
            "name": "raw layout",
            "columns": ["x", "y"]
        }
    ]
}
```

In the GUI, switch the active embedding from the dropdown at the top of the "Embeddings" tab or by pressing the space bar to cycle through them. Transitions are animated and all embeddings are scaled into a shared frame (the first embedding's) so points stay in view; recomputing an embedding from its features/distances replaces (supersedes) that embedding. The single-object form shown above remains fully supported.

## Multiple Datasets

If you have multiple datasets, you can point BigClust to a top-level directory containing an `info` file with pointers:

```
/projects/
    info                    <- JSON-formatted metadata about the projects
    /clustering_1/
        info                <- JSON-formatted metadata about the dataset
        ...
    /clustering_2/
        info                <- JSON-formatted metadata about the dataset
        ...
    ...
```

The top-level `info` file would look like this:
```json
[
    "path/to/clustering_1",            # expected to be a relative path
    "path/to/clustering_2",            # expected to be a relative path
]
```

## Best Practices for Parquet Formatting

BigClust reads `meta`, `distances`, `embeddings` and `features` from Parquet (recommended) or Apache Arrow Feather files. Parquet is preferred because it is compact, self-describing, columnar and widely supported. The notes below help keep projects small, fast to load and unambiguous.

### General

- **Prefer Parquet over Feather** for distribution. Feather is fine for local scratch data but Parquet compresses better and is more portable across versions.
- **Write with `pyarrow`.** Both `pandas` (`df.to_parquet(..., engine="pyarrow")`) and `pyarrow` directly produce files BigClust can read. Avoid the `fastparquet` engine for matrices with non-string column names, as it handles them inconsistently.
- **Use compression.** `snappy` (the pandas default) is a good speed/size trade-off; use `zstd` (e.g. `compression="zstd"`) when you want the smallest files and can afford slightly slower writes. Both decompress quickly on load.
- **Keep row order consistent.** `meta`, `embeddings`, `features` and the KNN graph are aligned by **row position**, so every file must list observations in the **same order as the `meta` file**. Don't shuffle or re-sort one file independently.

### `meta`

- Store `id` as an integer when possible — it is smaller, faster and avoids accidental whitespace/casing mismatches. Only fall back to strings when IDs are genuinely non-numeric.
- Keep `label` and `dataset` as plain strings. For columns with few distinct values (e.g. `dataset`), pandas `category` dtype is fine and is preserved through Parquet, but plain strings work everywhere.
- If you store a `color` column, use a consistent representation (e.g. a hex string like `"#1f77b4"` or an `[r, g, b]` list) across all rows.
- Don't write the DataFrame index as a meaningful column unless it carries data — set `index=False` (or reset the index) so the on-disk schema matches what you expect.

### `distances`

- A full distance matrix grows with the **square** of the number of observations, so this is usually the largest file. Store values as **`float32`** rather than `float64` to halve the size with no practical loss of precision.
- The **index and column names must be the IDs** (matching the `meta` `id` order). Use `df.to_parquet(...)` on a DataFrame whose index/columns are those IDs — pyarrow preserves the index by default; pass `index=True` explicitly if in doubt.
- If the matrix is symmetric, still store it in full (BigClust expects a square matrix); don't store only the upper/lower triangle.
- For large projects where a full matrix is impractical, supply a **KNN graph instead** (see above) — it is far smaller and loads much faster.

### `embeddings`

- Two (or more) float columns are all that is needed; **`float32`** is plenty for 2D scatter coordinates.
- Give the coordinate columns clear names (e.g. `x`/`y`) and reference them via `"columns": [...]` in the `info` file if they are not in a dedicated embeddings file.

### `features`

- Feature matrices are often **wide and sparse** (e.g. connectivity vectors). Store counts/weights as the smallest dtype that fits (`float32`, or an integer type for raw counts) to keep files compact.
- If you use **Multi-Index columns** to group features (e.g. upstream vs downstream), build the DataFrame with a `pandas.MultiIndex` on the columns before writing; pyarrow preserves it. Verify it round-trips by reading the file back and checking `df.columns` is still a MultiIndex.
- The index should be integers matching the `meta` `id` order — keep it consistent with the other files.

### KNN graph

- Store the `nn_idx_*` columns as an **integer** dtype (e.g. `int32`) — they are 0-based row positions, not floats. The `nn_dist_*` columns should be **`float32`**.
- Keep the columns in order (`nn_idx_1 … nn_idx_k`, `nn_dist_1 … nn_dist_k`); `k` is inferred from the count of `nn_idx_*` columns.
- One row per observation, in the **same order as `meta`**.

### Verifying a file

A quick round-trip check before shipping a project:

```python
import pandas as pd

df = pd.read_parquet("distances.parquet")
print(df.shape, df.dtypes.unique())     # expected shape and dtypes (e.g. float32)
print(df.index[:5], df.columns[:5])     # IDs preserved on index/columns?
```
