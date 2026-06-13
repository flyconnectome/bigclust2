# BigClust 2.0

> [!NOTE]
> This project is under active development but you can already use it to explore your clustering datasets. If you have any feedback or questions, please open an issue here on GitHub.

`bigclust2` is a re-design of [`bigclust`](https://github.com/flyconnectome/bigclust), a graphical interface for interactively exploring clusterings of high-dimensional connectomic data.
Typically this means morphological or connectivity-based embeddings but it can be used for any kind of distances or features.

Highlights:
- **Interactive 2D scatter plots**: Explore large clusterings interactively with zoom, pan, and selection.
- **Neuroglancer-like 3D viewer**: Visualize neuron morphology in a 3D viewer.
- **Widgets for days**: All the tools you need to explore and cluster your data: cluster methods, dimensionality reduction, fidelity metrics, feature selection, connectivity explorer, etc.
- **Annotations**: Push annotation straight to supported backends (e.g. Clio or SeaTable) or export them as CSV/Parquet files.

![BigClust 2.0 GUI](./_static/screenshot.png)

## Version 2.0 Notes
A totally reworked GUI aside, this new version also fundamentally changes how data is represented:
previously, data artifacts (distances, features, etc.) had to be manually loaded and passed to BigClust widgets.
For this new version, we have switched to a Neuroglancer-like approach where data sources are whole directories
(local or remote) containing both the data itself as well as metadata files describing the setup. Here is a simple
example structure:

```
/my_clustering/
    info                <- JSON-formatted settings for the dataset
    meta.parquet        <- per-neuron metadata in Parquet (recommended) or Apache Arrow Feather format
    distances.parquet   <- pairwise distances in Parquet (recommended) or Apache Arrow Feather format (optional)
    embeddings.parquet  <- low-dimensional embeddings in Parquet (recommended) or Apache Arrow Feather format (optional)
    features.parquet    <- high-dimensional features in Parquet (recommended) or Apache Arrow Feather format (optional)
```

See the [Data Format](#data-format) section below for details on the expected files and how to set up your project.

## Usage

First make sure you have the Python package manager `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/). Then run:

```bash
uvx --from git+https://github.com/flyconnectome/bigclust2@main bigclust2
```

This will install the latest released version of `bigclust2` from this repository and start the GUI.

> [!TIP]
> Note the `@main` in above command? This is asking `uvx` to always use the latest version of `bigclust2`.
> Because of how `uvx` works, this can lead to slow start-up times when there are new releases. You can
> avoid this by pinning to a specific version (e.g. `@0.1.0`) or a specific commit (e.g. `@02ea911`).

### Controls

Most GUI elements are hopefully self-explanatory - when in doubt look for the tooltip.

#### Scatterplot

- left click to move the view
- shift + left click to draw a selection box (add + cmd to add to selection)
- shift + control + left click to draw a lasso selection (add + cmd to add to selection)
- `ESC` to deselect all points

- `C` to toggle the control panel
- `L` to toggle labels
- left/right arrows increase/decrease font size of labels
- up/down arrows increase/decrease marker size
- double-click on a label to highlight points with the same label
- shift + double-click on a label to select points with the same label
- CMD/control + shift + double-click on a label to add points with the same label to the current selection

#### 3D Viewer

- left click + hold to rotate the view
- middle button + hold to pan
- scroll to zoom in and out
- `C` to toggle the legend
- to align the view: `1` (front), `2` (side), `3` (top)

## Troubleshooting

| Error  | Solution |
| ------ | -------  |
| Running `uvx ...` fails with an error containing `realpath: command not found` | If you're on Mac, make sure your OS version is at least 13.x |

## Development

1. Clone this repository
2. `cd bigclust2` to change into this directory
3. `uv run run.py --debug` to start the GUI

## Data Format

Example directory structure for a single dataset:

```
/my_clustering/
    info                <- JSON-formatted settings for the dataset
    meta.parquet        <- per-neuron metadata in Parquet (recommended) or Apache Arrow Feather format
    distances.parquet   <- pairwise distances in Parquet (recommended) or Apache Arrow Feather format (optional)
    embeddings.parquet  <- low-dimensional embeddings in Parquet (recommended) or Apache Arrow Feather format (optional)
    features.parquet    <- high-dimensional features in Parquet (recommended) or Apache Arrow Feather format (optional)
```

### `info` File (required)

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

### `meta` File (required)

The metadata file contains information about each observation (e.g. neuron) in the dataset:
- `id` (required): unique identifier for each observation (e.g. neuron)
- `label` (required): labels to use for the scatter plot
- `dataset` (required): which dataset the observation belongs to (e.g. "hemibrain", "FAFB", etc.); this is used for markers in the scatter plot but also for neuroglancer sources and updating/pushing annotation
- `color` (optional): colors used for markers in the scatter plot and for meshes in the 3D viewer
- `source` (optional): URLs or identifiers for the source of the observation (used for loading morphology in the 3D viewer)
- `x` / `y` (optional): pre-computed low-dimensional embeddings for the scatter plot (if not provided in a separate `embeddings` file)

The optional columns have to be explicitly specified in the `info` file (see above) so that BigClust knows how to use them!

Any additional columns are loaded and can be used for coloring, filtering, etc. in the GUI.

### `distances` File (optional)

The distances file contains pairwise distances between observations (e.g. neurons) in the dataset. This is optional but can be used for computing fidelity metrics and for feature selection. The file should be in Parquet (recommended) or Apache Arrow Feather format and contain a square matrix of shape `(n_observations, n_observations)`. Index and column names should be IDs matching the order of the `id` column in the `meta` file.

### KNN graph (optional alternative to a full distance matrix)

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

### `features` File (optional)

The features file contains high-dimensional features for each observation (e.g. neuron) in the dataset. This is optional but can be used for feature selection and for exploring the relationship between features and embeddings. The file should be in Parquet (recommended) or Apache Arrow Feather format and contain a matrix of shape `(n_observations, n_features)`. The index should be integers matching the order of the `id` column in the `meta` file. You can use Multi-Index columns to organize features into groups (e.g. upstream vs downstream connections).

### Multiple Embeddings

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

### Multiple Datasets

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


### Ideas / TODOs
- [x] Enable multiple embeddings per dataset (with a dropdown to select embedding)
- [x] Allow selecting sets of features (e.g. upstream vs downstream or isomorphic vs dimorphic connections; this could simply use the multi-columns)
- [x] Support pushing updated annotations back to annotation backends (e.g. Clio/FlyTable)
- [ ] Add support for Graph Exploration
- [ ] Allow users to change project settings (e.g. neuroglancer sources) before or after loading
- [ ] Support loading up-to-date annotations when opening a project (e.g. from Clio or Neuprint)
- [ ] Add more detailed documentation about data formats and structure
- [ ] Caching system for remote data sources for faster start-up times
  - this turns out to not really be a problem from the client side; could still implement to reduce traffic on the server side
- [ ] Support sharing figure state (e.g. `uvx bigclust --state <state_id>`)
- [X] Fine-control over hover info
