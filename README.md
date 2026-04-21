# BigClust 2.0

> [!NOTE]
> `bigclust2` is work in progress. It's still under active development but it works and you can already use it to explore your clustering datasets. If you have any feedback or suggestions, please open an issue or reach out to us on Slack.

![BigClust 2.0 GUI](./_static/screenshot.png)

## Overview
A new GUI for BigClust built with PySide6. This update also fundamentally changes how data is represented:
previously, data artifacts (distances, features, etc.) had to be manually loaded and passed to BigClust widgets.
For this new version, we have switched to a Neuroglancer-like approach where data sources are whole directories
(local or remote) containing both the data itself and metadata files describing the setup. Here is a simple
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

You can start the BigClust 2.0 app using [`uv`](https://docs.astral.sh/uv/). First make sure you have `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/) and then run:

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

Use the left click + hold to move the view and shift + left click + hold to draw a
selection box around points.

- `ESC` to deselect all points
- `C` to toggle the control panel
- `L` to toggle labels
- left/right arrows increase/decrease font size of labels
- up/down arrows increase/decrease marker size
- double-click on a label to highlight points with the same label
- shift + double-click on a label to select points with the same label
- CMD/control + shift + double-click on a label to add points with the same label to the current selection

#### 3D Viewer

Use the left click + hold to rotate the view, middle button + hold to pan and scroll to zoom in and out.

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
    # optional: feature vector
    "features": {
        "type": "connectivity",
        "file": "connections.parquet",
    },
}
```

The metadata file contains information about each observation (e.g. neuron) in the dataset:
- `id` (required): unique identifier for each observation (e.g. neuron)
- `label` (required): labels to use for the scatter plot
- `dataset` (required): which dataset the observation belongs to (e.g. "hemibrain", "FAFB", etc.); this is used for markers in the scatter plot but also for neuroglancer sources and updating/pushing annotation
- `color` (optional): colors used for markers in the scatter plot and for meshes in the 3D viewer


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
- [ ] Add more detailed documentation about data formats and structure
- [ ] Caching system for remote data sources for faster start-up times
- [x] Enable multiple embeddings per dataset (with a dropdown to select embedding)
- [ ] Allow users to change project settings (e.g. neuroglancer sources) before or after loading
- [ ] Support loading up-to-date annotations when opening a project (e.g. from Clio or Neuprint)
- [x] Support pushing updated annotations back to annotation backends (e.g. Clio/FlyTable)
- [ ] Support sharing figure state (e.g. `uvx bigclust --state <state_id>`)
- [ ] Fine-control over hover info
- [x] Allow selecting sets of features (e.g. upstream vs downstream or isomorphic vs dimorphic connections; this could simply use the multi-columns)
