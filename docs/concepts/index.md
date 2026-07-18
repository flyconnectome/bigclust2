# Concepts

Three pages of background. Nothing here tells you which button to press — that is
what the [how-to guides](../how-to/index.md) are for. This is the model the app is
built on, and it is what makes the buttons make sense.

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### :material-cube-outline: [Embeddings, distances, features](data-model.md)

The three kinds of data a project can hold, how they relate, and why what your
project ships decides what the app can do. Start here.

</div>

<div class="bc-card" markdown>

### :material-folder-network-outline: [Why data sources are directories](data-sources.md)

BigClust takes a directory, not a matrix. Where that idea comes from and what it
buys you.

</div>

<div class="bc-card" markdown>

### :material-lasso: [How selection works](selection.md)

Selection is the app's central verb. What it propagates to, and how growing and
shrinking actually work.

</div>

</div>

## The short version

BigClust holds **one set of observations** — normally neurons, one row in `meta`
each — and shows them **three ways at once**: as points in a 2D scatter plot, as
morphology in a 3D viewer, and as rows in whichever widgets you have open.

Everything else follows from that:

- The three views are joined by **selection**. Selecting points is how you ask
  every other part of the app to talk about the same neurons.
- The 2D positions come from an **embedding**, which is derived from
  **distances** or **features**. Which of those your project ships decides what
  can be recomputed.
- The whole thing is loaded from a **directory** described by an `info` file,
  which is why a clustering can be published at a URL.

If you read only one of these pages, read [embeddings, distances,
features](data-model.md) — it is the distinction people trip over.
