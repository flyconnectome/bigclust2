# Large embeddings

Most projects are small enough that BigClust draws every point, every frame,
and nothing on this page applies.

Two things change as a project grows. Past about **200,000** points the scatter
is split into **chunks** and stops drawing the ones you are not looking at —
which changes nothing you can see. Only once what is *in view* exceeds the draw
budget (**3 million** points by default) does it start drawing a **sample** of
them. That second step is worth understanding, because it changes what you see
while deliberately not changing what you get.

(The 200,000 is counted per marker symbol, so a project that draws several
datasets with different markers reaches it later than its total suggests.)

## Why drawing everything stops working

The cost of drawing a scatter is close to linear in the number of points
submitted to the GPU, and it does not care that most of them are invisible.
Zoomed out, tens of millions of points land on the same few thousand pixels: the
GPU draws all of them, and you see the handful that happen to be on top.

So the expensive work buys nothing. The fix is to stop submitting points you
cannot distinguish anyway.

## Chunks

The points are divided into **chunks** — groups of a few tens of thousands that
are near each other in the embedding. Chunks hold roughly equal numbers of
points, so a chunk covering a dense cluster is physically small while one over
empty space is large.

That gives the renderer two things it can do per chunk:

- **Skip chunks you are not looking at.** Panning or zooming in means most
  chunks fall outside the viewport, and those cost nothing at all.
- **Draw only part of a chunk.** The points inside a chunk are shuffled when the
  project loads, so the first half of a chunk is a fair sample of the whole
  chunk — not its left half, or its oldest neurons, or anything else
  systematic.

## The draw budget

`Max points drawn` in the Settings tab caps how many points may be drawn per
frame. If everything in view fits under the cap, everything in view is drawn.
If it does not, each visible chunk contributes a share in proportion to its
size.

The effect when you zoom in is that fewer chunks are in view, so each one gets a
bigger share of the budget and is drawn more densely — until you are close
enough that everything in view is on screen. Detail appears as you approach it,
which is roughly what you would expect from a map.

Lower the budget for a smoother frame rate on a busy scene, raise it for more
detail when you are holding still.

!!! note "What the budget does not affect"

    The budget is a **rendering** setting. The things that work from your data
    rather than from the screen are unaffected, whatever is currently drawn:

    - **Selection.** A lasso or rectangle selects every point in the region,
      including ones that were not drawn. The count in the corner is the real
      count, and so is everything it propagates to — the 3D viewer, the
      widgets, annotations. What you *see* of a selection does depend on the
      budget (see [below](#selection-highlights)), but what is selected does
      not.
    - **Labels**, which are placed from point positions (see
      [below](#labels)) rather than from what was drawn.
    - **Export.** A Plotly export writes the points, not the sample.

    So a sparse-looking scatter is a drawing of your data, not a subset of it.

    The exception is **hover**, which asks the renderer what is under the
    cursor and so can only find points that were actually drawn. Zoom in — or
    raise the budget — if you want to inspect a point the scatter is currently
    sampling out.

## Selection highlights

Selected points are drawn with a ring around them, and how much of that
highlight you get depends on how big the selection is relative to
`Max points drawn`.

**Up to a tenth of the budget** — 300,000 points at the default — the selection
is ringed in full. Rings are never sampled away, so you can see all of a modest
selection even where the scatter underneath it is sampled. These rings are not
culled either, so zoomed right in you can be drawing more of them than the
scatter is drawing points; what bounds the cost is the threshold itself, which
moves with `Max points drawn`. Lower the budget and this limit drops with it.

**Above it**, the highlight follows the scatter. Ringing ten million points
over a scatter showing one million costs more than the scatter does and shows
nothing extra, so instead a selected point that is drawn gets its ring and one
that was sampled out does not. For these the rings and the dots agree exactly:
you never see a visible point that looks unselected, and zooming in fills both
in together.

The difference between the two is deliberate — a small selection is worth
showing you completely, a huge one is not — and in neither case does any of it
touch what is *selected*. Only the drawing changes.

## Labels

Labels work from the points themselves, not from the sample, so the budget does
not change which ones you get. What does change them is how crowded the view is:
labels only appear once a few hundred points are on screen, and a view holding
more than that shows none at all. Zooming in is what brings them back, exactly
as on a small project.

That threshold is why a large embedding shows no labels at first. It is not a
limit on the project — it is a limit on how many labels fit on a screen before
they stop being readable.

**Show label lines**, which outlines each label's points, is the one label
feature whose cost does scale with the project: it builds an outline per
distinct label value over all of their points. On an embedding of tens of
millions of points that takes a while to appear after you tick it — and it is
rebuilt whenever the points move, so leaving it on makes every layout change
and embedding switch pay the same cost again.

## What this costs

Chunking is built when a project loads, which is why a very large project takes
longer to open than its file size suggests — the points have to be sorted into
chunks and shuffled within them. It is rebuilt whenever the points change:
switching embeddings pays the cost again, as does a layout change once its
animation settles.

Memory is the other limit, and it scales with the number of points rather than
with the budget: every point stays resident on the GPU, and only the number
*drawn* changes. A project of tens of millions of points needs several GB of RAM
on top of whatever its metadata occupies.

## What a very large project cannot do

Drawing scales. Analysis does not, and no amount of sampling helps: these tools
read your data, not the screen, so they have to touch every point.

Where that bites, the tool says so rather than appearing to work and then
freezing. Most of these are a question of patience, so you are *asked*, with
the cost stated and "no" as the default — if you are willing to wait ninety
seconds, that is your call to make. One is not a question of patience, and is
refused outright: there is no useful way to agree to running out of memory.

| Tool | Limit | Why |
|---|---|---|
| **Clustering — Agglomerative** | 25,000 points **(refused)** | Builds a table of every pair of distances, so memory grows with the *square* of the point count. Past the limit it does not get slow, it runs the machine out of memory — so it is greyed out rather than offered. |
| **Clustering — Spectral** | 50,000 points | Cost grows faster than the point count. |
| **Clustering — HDBSCAN** | 100,000 points | Same, less steeply: 50,000 points take seconds, 200,000 take a minute and a half. |
| **Clustering — K-Means** | none | Scales with the point count, so it stays usable: five million points cluster in a few seconds. |
| **Choose K** | as the method above | Runs the method once per candidate k, so it costs several times a single run. |
| **Evaluating cluster labels — Silhouette** | 50,000 points | Scoring compares every pair of points. |
| **Evaluating cluster labels — Neighbor consistency** | 250,000 points | Searches each point's neighbours, which is far cheaper than every pair. |
| **Fidelity scores** | 250,000 points | Compares every point's neighbours before and after embedding, twice. |
| **KNN edges, *All*** | 250,000 points | Searches every point's neighbours and then draws a line per edge. *Selected only* is bounded by your selection and is never limited. |
| **Distance heatmap** | 5,000 selected | A heatmap holds a cell for every pair, and more rows than that is unreadable regardless. |
| **Feature comparison** | 1,000,000 points | Keeps a row index per point to build its two groups. Above this the **View → Feature Comparison** entry is greyed out. |

A clustering method that cannot run at all is greyed out with its limit shown
beside the name, so you can see what you would have to do rather than guessing.

The point counts above were measured on two-dimensional input. Clustering
feature vectors of higher dimension costs more at the same number of points,
so treat them as upper bounds.

### Working on part of a project

The way round all of these is to make the project smaller rather than to wait:

- **[Filter the dataset](../how-to/filter-a-dataset.md)** to the neurons you
  care about. A filtered project holds fewer points, so everything above
  applies to the smaller number.
- **[Open a view](../how-to/work-in-several-views.md)** of a subset and work
  there, leaving the full embedding open alongside it.
- For the **distance heatmap**, select the points you want to compare — it
  works from the selection, not the whole project.

Re-computing an embedding is not capped, because how long it takes depends on
the input rather than only on the point count. It warns you with an estimate
first, and defaults to cancelling.

## If it still feels slow

- Lower `Max points drawn`.
- Try the other remedies in [troubleshooting](../reference/troubleshooting.md).
- Use [filters](../how-to/filter-a-dataset.md) or
  [views](../how-to/work-in-several-views.md) to work on a subset. A filtered
  view holds fewer points, so it is cheaper in every respect, not just to draw.
