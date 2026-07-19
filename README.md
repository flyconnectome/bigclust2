# BigClust 2.0

> [!NOTE]
> This project is under active development but that shouldn't stop you from using it to explore your clustering datasets. If you have any feedback or questions, please open an issue here on GitHub.

`bigclust2` is a re-design of [`bigclust`](https://github.com/flyconnectome/bigclust), a graphical interface for interactively exploring clusterings of high-dimensional connectomic data.
Typically this means morphological or connectivity-based embeddings but it can be used for any kind of distances or features.

📖 **[Documentation](https://flyconnectome.github.io/bigclust2/)** — guides, reference and the data format specification.

Highlights:
- **Interactive 2D scatter plots**: Explore large clusterings interactively with zoom, pan, and selection.
- **Neuroglancer-like 3D viewer**: Visualize neuron morphology in a 3D viewer.
- **Widgets for days**: All the tools you need to explore and cluster your data: cluster methods, dimensionality reduction, fidelity metrics, feature selection, connectivity explorer, etc.
- **Annotations**: Push annotation straight to supported backends (e.g. Clio or SeaTable) or export them as CSV/Parquet files.

![BigClust 2.0 GUI](./_static/screenshot.png)

## Quick start

First make sure you have the Python package manager `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/). Then run:

```bash
uvx bigclust2@latest
```

### Try the example dataset

Don't have a project yet? Open the public example — no account, no credentials, nothing to download first:

```bash
uvx bigclust2@latest --from https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/bigclust_data/examples/MaleCNS_FlyWire_hemibrain_central_brain_bigclust
```

That's a co-clustering of **87,263 central brain neurons** from FlyWire (32,384), MaleCNS (32,164) and Hemibrain (22,715), laid out by connectivity and streamed over HTTP (~63 MB). Select a cluster and the 3D viewer shows those neurons from all three connectomes together, in a common space.

See [the example dataset](https://flyconnectome.github.io/bigclust2/get-started/example-dataset/) for things to try.

### Open your own

```bash
uvx bigclust2@latest --from /path/to/my_clustering
uvx bigclust2@latest --from https://example.org/my_clustering
```

To build one, see [create a local dataset](https://flyconnectome.github.io/bigclust2/how-to/create-a-local-dataset/).

To work with the latest development version:

```bash
uvx --from git+https://github.com/flyconnectome/bigclust2@main bigclust2
```

> [!TIP]
> Note the `@main` in above command? This is asking `uvx` to always use the latest version of `bigclust2`.
> Because of how `uvx` works, this can lead to slow start-up times when there are new releases. You can
> avoid this by pinning to a specific version (e.g. `@0.1.0`) or a specific commit (e.g. `@02ea911`).

See the [installation guide](https://flyconnectome.github.io/bigclust2/get-started/installation/) for the other options, and [your first project](https://flyconnectome.github.io/bigclust2/get-started/first-project/) for a walkthrough.

## Data Format

Data sources are whole directories (local or remote) containing both the data itself as well as metadata files describing the setup — a Neuroglancer-like approach. A simple example:

```
/my_clustering/
    info                <- JSON-formatted settings for the dataset
    meta.parquet        <- per-neuron metadata (required)
    embeddings.parquet  <- low-dimensional embeddings for the scatter plot
    distances.parquet   <- pairwise distances (optional)
    features.parquet    <- high-dimensional features (optional)
```

The full specification — the `info` file, the `meta`/`distances`/`embeddings`/`features` files, KNN graphs, multiple embeddings, multiple datasets, and best practices for Parquet formatting — is in the [data format reference](https://flyconnectome.github.io/bigclust2/reference/data-format/).

## Controls

Most UI elements are hopefully self-explanatory — when in doubt look for the tooltip. The essentials:

| Key | Does |
| --- | ---- |
| `CMD/ctrl + shift + P` | Open the command palette — search and run any command |
| `C` | Toggle the control panel (scatter) / legend (3D viewer) |
| `L` | Toggle labels |
| shift + drag | Draw a selection box (add `CMD` to add to the selection) |
| shift + ctrl + drag | Draw a lasso selection (add `CMD` to add to the selection) |
| `ESC` | Deselect all points |
| `SPACE` | Cycle through embeddings |

If you only remember one, make it the command palette: it lists every command in the menu bar with fuzzy search (`gs` → "Grow Selection", `dh` → "Distance Heatmap") and shows each one's shortcut, so you pick those up as you go.

The complete list is in the [keyboard and mouse reference](https://flyconnectome.github.io/bigclust2/reference/shortcuts/), and in the app under **Help → Keyboard Shortcuts**.

## Documentation

The docs are built with [Zensical](https://zensical.org/) and live in [`docs/`](./docs).

```bash
uv run --extra docs zensical serve -o    # live preview
uv run --extra docs zensical build       # -> site/
```

They are published to GitHub Pages from `main` by [`.github/workflows/docs.yml`](./.github/workflows/docs.yml). Pull requests are built and link-checked with `--strict` but not published.

The landing page hero is [`docs/javascripts/clusters.js`](./docs/javascripts/clusters.js) — a dependency-free WebGL point cloud that morphs between six layouts of the same 9,000 observations (UMAP-style blobs, a t-SNE sunflower, two lobes, filaments, a horseshoe and a continuum). Without JavaScript or WebGL it falls back to the blurred screenshot.

Screen recordings referenced by the how-to pages are tracked in [`scripts/DOCS_MEDIA.md`](./scripts/DOCS_MEDIA.md); [`scripts/record_docs_media.sh`](./scripts/record_docs_media.sh) converts a capture into a GIF.

## Development

1. Clone this repository
2. `cd bigclust2` to change into this directory
3. `uv run run.py --debug` to start the GUI

Note `run.py` only understands `--debug`; use `uv run bigclust2 --from ...` if you need the other [CLI flags](https://flyconnectome.github.io/bigclust2/reference/cli/).

## Troubleshooting

See [troubleshooting](https://flyconnectome.github.io/bigclust2/reference/troubleshooting/).

| Error  | Solution |
| ------ | -------  |
| Running `uvx ...` fails with an error containing `realpath: command not found` | If you're on Mac, make sure your OS version is at least 13.x |

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
- [ ] Record the screen captures listed in `scripts/DOCS_MEDIA.md`
