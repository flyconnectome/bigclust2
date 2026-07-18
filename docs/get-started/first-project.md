# Your first project

This page opens a dataset and gets you to the point where selecting points in the
scatter plot shows you neurons in the 3D viewer.

It assumes you have something to open. If you do not, use the [example
dataset](example-dataset.md) — a public URL, nothing to build — and follow along
with that. To make a project of your own, see [create a local
dataset](../how-to/create-a-local-dataset.md).

## 1. Open a project

Start BigClust, then **File → Open Project** (++cmd+o++ / ++ctrl+o++).

In the dialog:

1. Type a path or URL into the box at the top, or press **Browse…** to pick a
   local folder. The dropdown remembers your last ten locations.
2. If the directory holds several projects, pick one under **Select Project**.
   **Project Details** below shows the `info` file's contents so you can confirm
   you have the right one.
3. Leave **Filters** and **Embeddings** alone for now.
4. Press **Load**.

You can skip the dialog entirely by naming the dataset on the command line:

```bash
uvx bigclust2@latest --from /path/to/my_clustering
```

!!! tip "Remote projects are just a URL"

    ```bash
    uvx bigclust2@latest --from https://example.org/clusterings/hemibrain_v1.2
    ```

    Only `http://` and `https://` are supported — `gs://` and `s3://` URLs are
    *not*. See [loading a remote dataset](../how-to/load-a-remote-dataset.md).

## 2. Find your way around

The window is two panes and a sidebar:

- The **scatter plot** on top is your embedding: one point per neuron, laid out
  in 2D.
- The **3D viewer** below (or beside — see the layout buttons in the top-right of
  each pane) shows the actual morphology of whatever is selected.
- The **control panel** on the left has five tabs: General, Embeddings, Fidelity,
  Cluster, Settings. Press ++c++ over the scatter plot to show and hide it.

The [annotated screenshot on the front page](../index.md) maps every region to
its reference page.

Two keys are worth learning immediately:

| Key | Does |
|---|---|
| ++c++ | Toggle the control panel (over the scatter) or the legend (over the viewer) |
| ++l++ | Toggle labels on the scatter plot |

## 3. Select some points

Selection is how you drive everything else in the app.

- **Shift + drag** draws a selection box.
- **Shift + Ctrl + drag** draws a freehand lasso.
- Hold ++cmd++ as well to *add* to the existing selection rather than replace it.
- ++esc++ clears the selection.

<figure class="bc-clip" markdown>
  <div class="bc-clip--todo" markdown>
:material-video-outline: &nbsp;**Clip pending:** `lasso-selection.gif`<br>
Shift + Ctrl + drag a lasso around one cluster; the 3D viewer fills in behind it.
  </div>
  <figcaption>Lasso-selecting a cluster.</figcaption>
</figure>

The status bar in the bottom-right keeps a running count — `Selected: 2,604`.

As soon as you select something, the 3D viewer starts fetching those neurons. For
a big selection that takes a moment; the viewer says `Loading 2,079 neurons`
while it works.

!!! warning "Selecting tens of thousands of points will hurt"

    Each selected neuron is a mesh to fetch and render. BigClust asks for
    confirmation before very large selections for exactly this reason. If you
    only want to know *which* neurons are in a region, turn off
    **View → Synchronize Viewer** first — the scatter selection then stops driving
    the 3D viewer, and selecting is instant.

## 4. Look at one cluster properly

Now that you have a selection, the other widgets have something to work on. All
four live under the **View** menu:

| Widget | Shortcut | Shows |
|---|---|---|
| Connectivity Table | ++shift+cmd+c++ | Up- and downstream partners of the selection |
| Distance Heatmap | ++shift+cmd+d++ | Pairwise distances within the selection |
| Feature Comparison | ++shift+cmd+f++ | What separates two groups of neurons |
| Meta Data Explorer | ++shift+cmd+m++ | The full meta table, filterable |

Each opens as its own window and follows the selection. See the [widget
reference](../reference/widgets.md) for what every control in them does.

## 5. Colour by something meaningful

In the control panel's **General** tab:

- **Color by** picks a meta column to colour points by. Any column in your `meta`
  table is available, not just the ones named in `info`.
- **Labels** picks the column drawn as text over the plot. Double-click a label
  to highlight every point sharing it; shift + double-click to *select* them.
- **Palette** switches the colour map.

This is usually the fastest way to find out whether your embedding agrees with
your existing annotations — colour by `label` and look for clusters that are not
one colour.

## Where to go next

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### The embedding is not fixed

If the project ships distances or features, you can recompute the layout with a
different method or metric without leaving the app.

[Recompute the embedding &rarr;](../how-to/recompute-embeddings.md)

</div>

<div class="bc-card" markdown>

### Neither is the clustering

HDBSCAN, agglomerative, k-means and spectral clustering all run in the Cluster
tab, and the result can be applied as labels.

[Recluster &rarr;](../how-to/recluster.md)

</div>

<div class="bc-card" markdown>

### Decisions can go back upstream

Once you are confident about a group, write the type straight back to Clio or
FlyTable.

[Push annotations &rarr;](../how-to/push-annotations.md)

</div>

</div>

If something is behaving oddly, the [concepts pages](../concepts/index.md) explain
the model the app is built on — particularly [what "embedding" versus "distances"
versus "features" means here](../concepts/data-model.md), which is the single
most common source of confusion.
