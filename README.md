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

See the [Data Format](./DATA_FORMAT.md) documentation for details on the expected files and how to set up your project.

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

Data sources are whole directories (local or remote) containing the data itself plus metadata files describing the setup. The full specification — the `info` file, the `meta`/`distances`/`embeddings`/`features` files, KNN graphs, multiple embeddings, multiple datasets, and best practices for Parquet formatting — lives in [DATA_FORMAT.md](./DATA_FORMAT.md).

## Ideas / TODOs
- [x] Enable multiple embeddings per dataset (with a dropdown to select embedding)
- [x] Allow selecting sets of features (e.g. upstream vs downstream or isomorphic vs dimorphic connections; this could simply use the multi-columns)
- [x] Support pushing updated annotations back to annotation backends (e.g. Clio/FlyTable)
- [ ] Add support for Graph Exploration
- [ ] Allow users to change project settings (e.g. neuroglancer sources) before or after loading
- [ ] Support loading up-to-date annotations when opening a project (e.g. from Clio or Neuprint)
- [x] Add more detailed documentation about data formats and structure
- [ ] Caching system for remote data sources for faster start-up times
  - this turns out to not really be a problem from the client side; could still implement to reduce traffic on the server side
- [ ] Support sharing figure state (e.g. `uvx bigclust --state <state_id>`)
- [X] Fine-control over hover info
