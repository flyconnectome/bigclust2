# The example dataset

There is a public project you can open right now, without building anything:

```bash
uvx bigclust2@latest --from https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/bigclust_data/examples/MaleCNS_FlyWire_hemibrain_central_brain_bigclust
```

That is the whole setup. No account, no credentials, no download step — BigClust
streams the project over HTTP. First load pulls about 63 MB and takes a few
seconds on a decent connection.

You can also paste the URL into **File → Open Project** (++cmd+o++) if you already
have the app running.

## What it is

A **co-clustering of central brain neurons from three different connectomes**,
laid out by connectivity:

| Dataset | Neurons |
|---|---:|
| FlyWire | 32,384 |
| MaleCNS | 32,164 |
| Hemibrain | 22,715 |
| **Total** | **87,263** |

The point of a co-clustering is that neurons from different brains are embedded
in *one* space, so the same cell type from three specimens should land in the
same place. Whether it does is exactly what the app is for.

The project ships:

- **An embedding** — `connectivity (UMAP, cosine)`.
- **A feature matrix** — 87,263 × 8,346 connectivity features, split into
  `upstream` and `downstream` groups.
- **Meta data** — `label`, `dataset`, `side`, `hemilineage`, `mapping` and
  `cn_frac_used`, with 11,677 distinct labels.
- **3D sources** — a per-neuron Neuroglancer source per dataset, with FlyWire and
  Hemibrain registered into MaleCNS space, plus a brain outline mesh.

That last point is what makes the 3D viewer worth looking at here: select a
cluster and you get neurons from all three connectomes rendered together, in a
common space, without any transform step of your own.

!!! note "Features, not distances"

    This project has no `distances` file — a full pairwise matrix for 87,263
    neurons would be a 7.6-billion-cell array. Everything that needs all pairwise
    distances is therefore unavailable: the **Distance Heatmap** stays disabled.

    Everything that works from features works fine, and there is more of it —
    you can change the metric, use feature subsets, and rank features between
    groups. See [the data model](../concepts/data-model.md#why-this-decides-what-the-app-can-do).

## Things to try

**Colour by `dataset`.** In the control panel's **General** tab, set **Color by**
to `dataset`. This is the first question to ask of any co-clustering: if the three
datasets separate into their own regions, the co-clustering has failed and you are
looking at batch effect. If they intermingle *within* clusters, it worked.

**Select a cluster and look at it.** Shift + Ctrl + drag a lasso around a tight
group. The 3D viewer fills with those neurons — from all three connectomes at
once. Matching morphology across specimens is the visual confirmation that a
cluster is a real cell type rather than an artefact of one dataset.

**Colour by `hemilineage`.** Developmental origin is independent of the
connectivity that produced the layout, so agreement between the two is meaningful
rather than circular.

**Colour by `side`.** Left and right homologues of the same cell type should sit
together — a cluster that splits cleanly down the middle by `side` is usually
telling you about asymmetric reconstruction rather than about biology.

**Find a type you know.** Use the **Search** box in the General tab, or turn on
labels with ++l++ and shift + double-click one to select every neuron sharing it.
The Kenyon cells are the largest groups here — `KCg-m` alone has 4,121 neurons.

**Ask what separates two clusters.** Select one group, **Copy Selection** into
Group A of the [Feature Comparison widget](../how-to/compare-features.md), then do
the same for a neighbouring group into Group B. Because the features carry
`upstream` and `downstream` levels, you can ask whether two types differ by what
they listen to or by what they talk to.

**Re-embed on half the data.** In the **Embeddings** tab, untick `downstream`
under **Feature Subset** and press **Re-calculate positions**. That lays the same
neurons out by their *inputs* alone. Clusters that survive both are on firmer
ground than clusters that only exist in one.

## Working comfortably at this size

87,263 points is a real dataset, not a toy, and two things are worth knowing.

**Selecting a lot of neurons is expensive.** Every selected neuron is a mesh the
3D viewer fetches and renders. If you want to explore the scatter plot quickly,
untick **View → Synchronize Viewer** — selection then becomes instant, at the cost
of not seeing morphology. Turn it back on when you have a group worth looking at.

**Load a subset instead.** The **Filters** box in the Open Project dialog is
applied while reading, so a subset loads proportionally faster:

```
hemilineage == "MBp3"
```

gives you 2,863 neurons — 1,284 from FlyWire, 1,038 from MaleCNS and 541 from
Hemibrain. Set **Embeddings** to `calculate from features` at the same time, so
the layout is recomputed for the subset rather than inherited from a plot that was
arranged around 87,000 points. See [loading part of a
dataset](../how-to/filter-a-dataset.md).

!!! tip "Check a column's values before you filter on it"

    A filter that matches nothing, or matches only part of what you meant, fails
    silently — an empty result looks exactly like a genuinely empty category. Open
    the **Meta Data Explorer** (++shift+cmd+m++) and look at the column first.

    The columns here are already reconciled across the three connectomes: `side`
    is `left` / `right` / `center` throughout, so `side == "left"` means the same
    thing in all of them. That is not free — each source brain spells its
    annotations differently, and lining them up is exactly what [column
    mapping](../how-to/update-meta-data.md#configure-the-sources) is for. Expect
    to do it in projects of your own.

## Then what

Once you have a feel for the app on real data:

- [Your first project](first-project.md) walks through the interface properly.
- [The how-to guides](../how-to/index.md) cover each task in turn.
- [Create a local dataset](../how-to/create-a-local-dataset.md) builds a project
  of your own, which is where most people go next.
