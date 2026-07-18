# Why data sources are directories

BigClust does not have a "load distance matrix" button. It opens a *directory*
containing an `info` file that describes what else is in there. This is a
deliberate change from BigClust 1.x, where you assembled the pieces in Python and
handed them to widgets, and it is worth explaining because it shapes everything
else.

## The Neuroglancer model

The idea is borrowed from
[Neuroglancer](https://github.com/google/neuroglancer), which is how connectomics
already shares volumetric data. A Neuroglancer source is a URL. Behind it sits a
directory with a small JSON manifest — also called `info` — describing the voxel
size, the scales, the data type. The client fetches the manifest, learns the
shape of what is there, then fetches what it needs.

BigClust does the same thing one level up, for derived data:

```
/my_clustering/
    info                <- JSON: what is here and how to read it
    meta.parquet        <- one row per neuron
    embeddings.parquet  <- the 2D coordinates
    distances.parquet   <- pairwise distances        (optional)
    features.parquet    <- the high-dimensional data (optional)
```

```bash
uvx bigclust2@latest --from https://example.org/clusterings/hemibrain_v1.2
```

The consequence is that a clustering becomes something with an **address**. You
can put one behind a URL and send someone the link. They do not need your script,
your environment, or a copy of the data — and critically, they cannot open a
*slightly different* version of it by accident.

## What the `info` file buys you

The manifest is what makes the data self-describing. Without it, a directory of
Parquet files is ambiguous: which one holds the coordinates? Are those distances
or a feature matrix? What metric produced them? What do the IDs refer to?

The `info` file answers all of that up front, which has three practical effects.

**The app can adapt before loading anything.** The Open Project dialog shows you
the project's details, and the Embeddings dropdown offers `calculate from
distances` only if the project declares distances. The UI is derived from the
manifest, so it cannot offer you an operation the data does not support.

**Only what is needed is fetched.** The manifest is read first, and it is tiny.
For a remote project you find out the URL is wrong, or the version is not the one
you wanted, before a gigabyte of distance matrix starts downloading.

**Semantics travel with the data.** `"metric": "cosine"` on a distance matrix,
`"type": "connectivity"` on a feature matrix, the colour column, the Neuroglancer
source for the 3D viewer — all of that lives in the file rather than in the
analyst's head or in a README nobody reads.

## What it costs

Being honest about the trade-offs:

**You have to build a project.** There is no "just point it at this DataFrame".
Producing a project directory is an extra step at the end of your analysis
pipeline. It is a small step — write four Parquet files and a JSON manifest, as
in [this worked example](../how-to/create-a-local-dataset.md) — but it is a step.

**Consistency is on you.** `meta`, `embeddings`, `features` and the KNN graph are
aligned by **row position**, not by joining on ID. Re-sorting one file
independently silently corrupts the project. The [data format
reference](../reference/data-format.md) says this repeatedly for good reason.

**There is no caching.** A remote project is re-fetched every run. This is a
deliberate simplification — a cache means invalidation, and a stale cache of a
clustering is worse than a slow load. **Export → Project** is the manual escape
hatch: it writes a complete local snapshot, which you then open instead.

## Composition

Because a project is just a directory, projects compose by nesting. A directory
whose `info` file is a list of relative paths is a collection:

```
/projects/
    info                    <- ["clustering_1", "clustering_2"]
    /clustering_1/
        info
        ...
    /clustering_2/
        info
        ...
```

Point BigClust at `/projects/` and you get a picker. This is how a lab publishes
a whole set of clusterings at one URL.

The same nesting shows up inside a single project: `embeddings` can be a list,
each entry carrying its own `features` and `distances`. One project can therefore
hold "these neurons by connectivity" and "these neurons by morphology" as two
embeddings over one shared `meta` table — which is what makes cycling between
them with ++space++ meaningful, since it is the *same* neurons being laid out two
ways.

## Live data stays live

One thing deliberately does not get frozen into the directory: annotations.

The `meta` table is a snapshot, and snapshots go stale. So `info` can also declare
**meta sources** — per-dataset pointers at live annotation backends, with a column
mapping. BigClust can then [pull current values](../how-to/update-meta-data.md)
into the in-memory table, and warn you with a status-bar banner when the snapshot
is more than a day old.

That refresh is read-only and stays in memory. Writing back is [pushing
annotations](../how-to/push-annotations.md), which is a separate flow with its own
guard rails — and it writes to the *backend*, never to the project directory.

The project directory is a published artefact. It describes what was true when it
was built, and BigClust never modifies it in place.

Next: [how selection works](selection.md).
