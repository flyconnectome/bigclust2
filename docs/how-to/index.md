# How-to guides

Each page here is one task, as a numbered recipe. They assume you can already
[open a project](../get-started/first-project.md) and select points.

## Getting data in

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### [Create a local dataset](create-a-local-dataset.md)

Build a project from scratch — a worked NBLAST of hemibrain neurons, with the 3D
viewer wired up. Start here if you have data but no project yet.

</div>

<div class="bc-card" markdown>

### [Load a remote dataset](load-a-remote-dataset.md)

Open a clustering published at a URL, and what to do when it won't open.

</div>

<div class="bc-card" markdown>

### [Load part of a dataset](filter-a-dataset.md)

Filter at load time with a query or an ID list, so you work with the 3,000
neurons you care about instead of 300,000.

</div>

</div>

## Changing what you see

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### [Recompute the embedding](recompute-embeddings.md)

Swap UMAP for t-SNE, change the metric, drop feature groups — and watch the
points move.

</div>

<div class="bc-card" markdown>

### [Recluster the data](recluster.md)

Run HDBSCAN, agglomerative, k-means or spectral clustering and apply the result
as labels.

</div>

<div class="bc-card" markdown>

### [Find what separates two groups](compare-features.md)

Rank the features that distinguish one set of neurons from another, and test
whether a split is real.

</div>

</div>

## Getting data out

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### [Refresh the meta data](update-meta-data.md)

Pull current annotations from neuPrint, Clio or FlyTable into your project, or
merge in a local file.

</div>

<div class="bc-card" markdown>

### [Push annotations](push-annotations.md)

Write the types you settled on back to Clio, FlyTable or a CSV.

</div>

<div class="bc-card" markdown>

### [Export your work](export.md)

Meta data, cluster assignments, an interactive Plotly page, or a full project
snapshot.

</div>

</div>

## Working comfortably

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### [Work in several views](work-in-several-views.md)

Tabs, tear-off windows, and keeping a subset of neurons in its own view.

</div>

</div>

---

If you are looking for what a specific control does rather than how to complete a
task, that is the [widget reference](../reference/widgets.md). If a recipe says
something you don't have the background for, the [concepts
pages](../concepts/index.md) are the missing half.
