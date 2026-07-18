# Recluster the data

The **Cluster** tab runs a clustering over your distances or features and turns
the result into labels you can colour by, refine by hand and export.

This is separate from [recomputing the embedding](recompute-embeddings.md). The
embedding decides *where points are drawn*; the clustering decides *which group
each point belongs to*. Both read the same underlying data, and they can disagree
— which is informative rather than a bug.

## Steps

1. Press ++c++ over the scatter plot, then open the **Cluster** tab.
2. Under **Input**, pick a **Data** source and a **Metric**.
3. Under **Algorithm**, pick a **Method** and set its parameters.
4. Press **Run**. The **Result:** label under **Output** reports what you got.
5. Press **Apply labels** to colour the scatter plot by the new clustering.

**Clear** discards the result. **Export** saves it as a CSV with `id` and
`bigclust_cluster` columns, and the folder button next to **Run** loads such a
CSV back in — which is how you carry a clustering between sessions.

<figure class="bc-clip" markdown>
  <div class="bc-clip--todo" markdown>
:material-video-outline: &nbsp;**Clip pending:** `recluster.gif`<br>
Set HDBSCAN min cluster size, press Run, press Apply labels; the scatter recolours.
  </div>
  <figcaption>Running HDBSCAN and applying the result as labels.</figcaption>
</figure>

## Choosing a method

| Method | Available when | You have to say |
|---|---|---|
| **HDBSCAN** | always | How small a cluster is allowed to be — not how many there are |
| **Agglomerative** | full distances or features | Either how many clusters, or where to cut the tree |
| **K-Means** | full distances or features | Exactly how many clusters |
| **Spectral** | always | Exactly how many clusters |

With a [KNN graph](../reference/data-format.md#knn-graph) only **HDBSCAN** and
**Spectral** are available, and a red note in the tab says so. Agglomerative and
k-means need the full pairwise structure the graph does not carry.

For connectomics data HDBSCAN is usually the right first thing to reach for,
because the honest answer to "how many cell types are in this data?" is normally
"I don't know, that's what I'm trying to find out" — and it is the only one of
the four that does not make you answer that up front. It is also the only one
that will tell you a point belongs to *no* cluster.

### HDBSCAN

| Control | Default | What it does |
|---|---|---|
| `Min cluster size:` | 5 | The smallest group that counts as a cluster. The main knob — raise it to get fewer, larger clusters. |
| `Min samples:` | 0 | How conservative the algorithm is about noise. `0` means "use `Min cluster size`". Higher values push more points to noise. |
| `Cluster epsilon:` | 0.0 | Merge clusters closer than this. Useful for stopping one real cluster being split in two. |
| `Selection method:` | `eom` | `eom` (excess of mass) prefers larger stable clusters; `leaf` gives you the finest split the hierarchy supports. |
| `Force noise point assignment` | off | Assign every noise point to its nearest cluster instead of leaving it as `-1`. |
| `Noise threshold:` | off | With the above, only assign noise points within this distance. |

Noise points get cluster `-1` and appear as `noise (-1)` in the manual-refinement
dropdown. A large noise fraction is not necessarily a failure — in cell-typing it
often means those neurons genuinely do not sit in a tight group.

### Agglomerative

`Stop criterion:` decides which of the next two controls is live:

- `N clusters` — cut the tree to give exactly `N clusters (k):` groups.
- `Distance threshold` — cut at `Distance threshold:` instead, giving however
  many clusters that produces.
- `Homogeneous composition` — cut adaptively so that clusters are homogeneous
  with respect to a chosen `Composition labels:` column, bounded by `Max split
  distance:`, `Min split distance:` and `Merge distance diff:` (0 means
  unbounded on each).

`Linkage:` is `ward`, `complete`, `average` or `single`.

!!! warning "Ward and precomputed distances are incompatible"

    Ward linkage is only defined for Euclidean distances. Selecting `ward`
    together with a precomputed distance matrix — or any non-Euclidean metric —
    **raises an error**; it does not quietly fall back to something else.

    Use `average` for precomputed distances. (An older tooltip and docstring claim
    a silent substitution happens. They are wrong; the code raises.)

### K-Means and Spectral

Both need `N clusters:` up front. Both have a **Find k** dropdown next to it that
will search for a good value by `Silhouette`, `Calinski-Harabasz` or
`Davies-Bouldin` score.

**Find k** needs to be able to recompute the clustering many times over raw
coordinates, so it is disabled for precomputed distance matrices and KNN graphs.

Spectral additionally takes `Affinity:` (`nearest_neighbors` or `rbf`), `Gamma:`,
`N neighbors:` and `N init:`.

## Singletons become noise

Whatever method you use, clusters with exactly one member are relabelled to `-1`
(noise). A cluster of one is not a cluster, and leaving them in makes every
downstream count misleading.

If you are seeing far more noise than you expect, this is often why — try a
larger `Min cluster size`, or a coarser cut.

## Fixing it by hand

Automatic clustering will get some things wrong that are obvious on screen. The
**Manual Refinement** group lets you fix them without re-running anything:

1. Select the points you want to move in the scatter plot.
2. Either pick an existing cluster under `Target cluster:` and press **Set
   Cluster**, or press **New Cluster** to put them in a fresh one.
3. **Reset** discards all manual edits and returns to the computed result.

Manual edits are part of the result, so they are included when you **Apply
labels** or **Export**.

## Checking whether a split is real

Before you commit to a boundary the algorithm drew, test it:

- The **Fidelity** tab's **Evaluate Labels** group scores your current labels by
  `Silhouette` or `Neighbor consistency`, against a data source and metric you
  choose. Tick `Sync w/ labels` to have it follow the label column.
- The [Feature Comparison widget](compare-features.md) takes two groups and tells
  you both *what* separates them and *how well* — its **Eval. separation** dialog
  is exactly the "is this one cluster or two?" question.

A split that no metric supports and that you cannot name a distinguishing
feature for is probably a split you should not keep.

## Persisting a clustering

Clusters live in the session. To keep them:

- **Export** in the Cluster tab writes `id` / `bigclust_cluster` as CSV, loadable
  again with the folder button.
- **Apply labels** followed by **Export → Meta Data → To CSV** gives you the
  labels alongside the rest of the meta table.
- If the clustering *is* your answer, [push it to an annotation
  backend](push-annotations.md).

Next: [find what separates two groups](compare-features.md).
