# Recompute the embedding

The scatter plot's layout is not fixed. If your project ships `distances` or
`features`, BigClust can recompute the 2D coordinates from them with a different
method, metric or parameter set, and animate the points to their new positions.

If your project ships *only* an `embeddings` file, there is nothing to recompute
from and the Embeddings tab will have no data source to offer. See [what
embeddings, distances and features mean here](../concepts/data-model.md).

## Steps

1. Press ++c++ over the scatter plot to show the control panel, then open the
   **Embeddings** tab.
2. Under **Input**, pick a **Method** and a **Data** source. The source-info label
   underneath tells you what you just selected — its shape and type.
3. Adjust the method's parameters under **Method Settings**.
4. Press **Re-calculate positions**.

On a large project you get a **Confirm re-embedding** dialog first, because the
computation is not cheap and not cancellable.

<figure class="bc-clip" markdown>
  <div class="bc-clip--todo" markdown>
:material-video-outline: &nbsp;**Clip pending:** `recompute-embedding.gif`<br>
Switch Method from UMAP to t-SNE, press Re-calculate positions, points animate to the new layout.
  </div>
  <figcaption>Recomputing the embedding with a different method.</figcaption>
</figure>

Tick **Auto run** to have every parameter change trigger a recompute
automatically. Useful for sweeping one slider on a small project; painful on a
large one.

## Choosing a method

| Method | Available when | Good for |
|---|---|---|
| **UMAP** | always | The default. Preserves local neighbourhoods and gives usable global structure. |
| **t-SNE** | always | Tighter, more separated clusters. Distances *between* clusters mean less than in UMAP. |
| **MDS** | full distance matrix only | Tries to preserve actual distances rather than neighbourhoods. Slow, but the axes mean something. |
| **PaCMAP** | feature matrix only | A middle ground between local and global structure. Not parameterised in the UI. |

!!! note "A KNN graph limits your options"

    If the project supplies a [k-nearest-neighbours graph](../reference/data-format.md#knn-graph)
    instead of a full distance matrix, only **UMAP** and **t-SNE** are offered —
    they can consume the graph directly. MDS and PaCMAP need distances the graph
    does not contain, and the feature metric, rebalancing and PCA controls are
    disabled for the same reason. UMAP's `Number of neighbors` and t-SNE's
    `Perplexity` are also capped to what your `k` supports.

## Parameters

### UMAP

| Control | Default | Range | What it does |
|---|---|---|---|
| `Number of neighbors:` | 10 | 1–200 | How much of the data each point is laid out against. Low values chase local detail, high values favour global shape. |
| `Minimum distance:` | 0.1 | 0.0–100.0 | How tightly points may pack. Lower gives denser clumps. |
| `Spread:` | 1 | 0.0–1000.0 | The overall scale points are spread over. Read together with `Minimum distance`. |
| `Set operation mix ratio:` | 1.0 | 0.0–1.0 | 1.0 is a fuzzy union, 0.0 a fuzzy intersection of local neighbourhoods. |
| `DensMAP:` | off | | Preserve local *density* as well as position — clusters that are genuinely tight look tight. |
| `DensMAP lambda:` | 2.0 | 0.0–10.0 | How hard DensMAP pushes. Only active when DensMAP is on. |

### t-SNE

| Control | Default | Range |
|---|---|---|
| `Perplexity:` | 30.0 | 1.0–200.0 |
| `Learning rate:` | 200.0 | 10.0–1000.0 |
| `Iterations:` | 1000 | 250–10000 |

Perplexity is the one that matters — it is roughly "how many neighbours each
point should care about". If your clusters are shattering into small fragments,
raise it.

### MDS

| Control | Default | Range |
|---|---|---|
| `Number of initializations:` | 4 | 1–200 |
| `Max iterations:` | 300 | 1–10000 |
| `Relative tolerance:` | 0.001 | 0.0000–1.0000 |

### Reproducibility

`Random seed:` is empty-means-random and shows the placeholder `random
initialization`; it is pre-filled with `42`. Set it to a fixed number if you want
the same layout twice, clear it if you want to check that a structure you are
looking at is not an artefact of one particular initialisation.

## Working from features

When the data source is a feature matrix, three extra groups appear.

**Feature Subset** gives you a checkbox per top-level feature group — this only
appears if your `features` file uses MultiIndex columns. This is where you say
"lay these neurons out by their *upstream* connectivity only", and it is often the
single most informative thing you can do with a connectivity feature matrix.

**Feature Options**:

| Control | Options | Notes |
|---|---|---|
| `Feature metric:` | `cosine`, `euclidean`, `manhattan`, `correlation`, `chebyshev` | Cosine is the sensible default for connectivity vectors — it compares *pattern* rather than magnitude, so a strongly-connected neuron is not automatically far from a weakly-connected one with the same partners. |
| `Feature rebalancing:` | `none`, `z-score`, `robust (median/IQR)`, `log1p + z-score` | Applied per feature before the metric. `log1p + z-score` is the usual choice for raw synapse counts, whose distribution is heavily skewed. |
| `Enable PCA` | off, 100 components | Project to N principal components first. Speeds up the embedding on wide matrices and suppresses noise; costs you whatever the discarded components held. |

## Reading the result

Recomputing **supersedes** the embedding it was computed from — the new layout
replaces that entry rather than being added alongside.

If your project ships [several embeddings](../reference/data-format.md#multiple-embeddings),
press ++space++ over the scatter plot to cycle through them. Transitions are
animated and every embedding is scaled into a shared frame, so points stay on
screen and you can actually see what moved.

That animation is the point. A cluster that holds together across a connectivity
embedding and a morphology embedding is telling you something a single static
layout cannot.

!!! tip "Check the layout before you trust it"

    The **Fidelity** tab exists to answer "is this embedding lying to me?".
    **Neighborhood Fidelity** scores how well each point's 2D neighbours match its
    neighbours in the original high-dimensional space. Points that score badly are
    ones the layout has misplaced — colour by the fidelity score and you can see
    which regions of the plot to distrust.

Next: [recluster the data](recluster.md).
