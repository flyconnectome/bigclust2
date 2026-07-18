# Embeddings, distances and features

A BigClust project can carry three different representations of the same
neurons. They are not interchangeable, they sit at different levels of
derivation, and which ones your project ships determines what the application can
do. This is the single most common source of "why is that greyed out?".

## The three levels

```
features          →      distances          →      embedding
(n × f matrix)           (n × n matrix)             (n × 2 coordinates)

what each neuron         how different            where each neuron
actually is              each pair is             is drawn
```

**Features** are the raw high-dimensional description of each neuron. In practice
this is usually a connectivity vector: one column per potential partner (or
partner type), holding the number of synapses. It can be anything — NBLAST score
vectors, morphological measurements, expression levels. The defining property is
that it is *per neuron*, and that the columns mean something individually.

**Distances** are pairwise: for every two neurons, one number saying how
different they are. This is what you get by applying a metric — cosine,
Euclidean, correlation — to the feature vectors. The columns no longer mean
anything individually; a distance matrix has thrown away *what* neurons are and
kept only *how they compare*.

**An embedding** is the 2D layout: for each neuron, an x and a y. This is what the
scatter plot draws. It is produced by running UMAP, t-SNE, MDS or PaCMAP over
either the distances or the features, and it is a lossy summary of them — a
hundred thousand pairwise relationships squeezed into a plane.

Each arrow is one-way. You can go from features to distances by picking a metric;
you cannot go back, because the metric discarded information. You can go from
either to an embedding; you cannot recover distances from 2D coordinates.

## Why this decides what the app can do

| Project ships | You can | You cannot |
|---|---|---|
| `embeddings` only | Look, select, colour, filter, annotate, export | Recompute the layout, recluster, rank features, show a heatmap |
| `+ distances` | …also recompute the embedding (any method), recluster (any method), show a distance heatmap | Rank features, change the metric, choose feature subsets |
| `+ features` | …also change the metric, use feature subsets, rebalance, run PCA, compare feature rankings between groups | |

Adding features to a project is worth it. Distances fix a metric at build time,
and a metric is a modelling choice you often want to revisit — comparing
connectivity by cosine versus correlation can move a cluster boundary. With
features in the project, that becomes a dropdown; with only distances, it becomes
a rebuild.

The trade is size. Distances grow with the **square** of the number of neurons —
100,000 neurons is a 10-billion-cell matrix — while features grow linearly with
the number of feature columns. For large projects a full distance matrix is
simply impractical, which is what the [KNN graph
option](../reference/data-format.md#knn-graph) exists for: keep each neuron's
*k* nearest neighbours and throw away the rest of the matrix.

A KNN graph is a further-degraded form of distances. It supports the operations
that only need local neighbourhoods (UMAP, t-SNE, HDBSCAN, spectral clustering,
neighbourhood fidelity) and not the ones that need all pairs (MDS, k-means,
agglomerative clustering, the distance heatmap).

## The embedding is a hypothesis

It is worth being explicit about this, because the scatter plot is so convincing.

The embedding you are looking at is one 2D projection of a space with hundreds or
thousands of dimensions, produced by an algorithm with parameters you or someone
else chose. Two points being close on screen is *evidence* they are similar. It is
not proof, and the failure mode is not rare — dimensionality reduction routinely
places points near each other that are nowhere near each other in the original
space.

BigClust gives you three ways to check rather than trust:

**Neighborhood fidelity** (Fidelity tab) scores each point on how well its 2D
neighbours match its true high-dimensional neighbours. Colour by that score and
you can see which regions of the plot are trustworthy.

**Recompute with different parameters.** A cluster that survives UMAP at
`n_neighbors=5` and `n_neighbors=50`, and survives t-SNE, is real. One that only
appears at one setting is an artefact of that setting.

**Cycle embeddings** with ++space++. If your project ships several — say a
connectivity embedding and a morphology one — a group that holds together in both
is a much stronger claim than one that holds together in either.

This is also why clustering runs on the **distances or features**, not on the 2D
coordinates. Clustering an embedding would compound the projection's errors
instead of checking them. If a clustering computed on the real data disagrees with
what the embedding shows you, that disagreement is information — usually about
the embedding.

## Multiple embeddings

A project can ship several embeddings, each optionally with its own
features/distances. This is how you put "these neurons by connectivity" and
"these neurons by morphology" in one project, and it makes the ++space++ key a
genuine analysis tool rather than a convenience.

All embeddings are scaled into a shared frame — the first one's — so points stay
on screen when you switch, and transitions are animated. You are meant to watch
what moves.

Recomputing an embedding *supersedes* the one it was derived from rather than
adding a new one, so cycling does not fill up with your experiments.

See [multiple embeddings](../reference/data-format.md#multiple-embeddings) for
how to declare them.

Next: [why data sources are directories](data-sources.md).
