# BigClust 2.0

## Overview
A new GUI for BigClust built with PySide6. This update also fundamentally changes how data is represented:
previously, data had to be manually loaded and passed to BigClust widgets. For this new version, we have
switched to a Neuroglancer-like approach where data sources are whole directories (local or remote)
containing both the data itself and metadata files describing the setup. Here is a simple example structure:

```
/my_clustering/
    info                <- JSON-formatted settings for the dataset
    meta.parquet        <- per-neuron metadata in Apache Arrow Feather or Parquet format
    distances.parquet   <- pairwise distances in Apache Arrow Feather or Parquet format (optional)
    embeddings.parquet  <- low-dimensional embeddings in Apache Arrow Feather or Parquet format (optional)
    features.parquet    <- high-dimensional features in Apache Arrow Feather or Parquet format (optional)
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
        "file": "meta.feather",  # must at least have `id`, `label` and `dataset` columns
        "source": "neuprint://https://neuprint.janelia.org@hemibrain:v1.2",  # define source for updating metadata
        "last_updated": "2024-06-01",  # meta data can be update independently of the other data
        "color": "color"  # optional: column in meta to use for colors
    },
    # required: precomputed low-dimensional embeddings for scatter plot
    "embeddings": {
        "file": "embeddings.feather"  # alternatively: "columns": ["x_coord", "y_coord"]
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
        "file": "distances.feather",
        "metric": "cosine"
    },
    # optional: feature vector
    "features": {
        "type": "connectivity",
        "file": "connections.feather",
    },
    # optional: define how to update annotations for a given dataset from an annotation backend
    "push_annotations": {
        "hemibrain": "clio://https://clio.janelia.org@hemibrain"
    }
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
- [ ] Enable multiple embeddings per dataset (with a dropdown to select embedding)
- [ ] Allow users to change project settings (e.g. neuroglancer sources) before or after loading
- [ ] Support loading up-to-date annotations when opening a project (e.g. from Clio or Neuprint)
- [ ] Support pushing updated annotations back to annotation backends (e.g. Clio/FlyTable)
- [ ] Support sharing figure state (e.g. `uv run bigclust --state <state_id>`)
- [ ] Fine-control over hover info


## Usage

You can start the BigClust 2.0 GUI using [`uv`](https://docs.astral.sh/uv/). First make sure you have `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/) and then run:

```bash
uvx --from git+https://github.com/flyconnectome/bigclust2@main bigclust
```

> [!TIP]
> Note the `@main` in above command? We're asking `uvx` to always use the latest version of `jumpstart` but we could also point it at a specific release using e.g. `@02ea911`.


## Troubleshooting

| Error  | Solution |
| ------ | -------  |
| Running `uvx ...` fails with an error containing `realpath: command not found` | If you're on Mac, make sure your OS version is at least 13.x |


## Development

1. Clone this repository
2. `cd bigclust-jump-start` to change into this directory
3. `uv run run.py --debug` to start the GUI
