# Widgets

Every panel, tab and window, control by control. For what to *do* with them, see
the [how-to guides](../how-to/index.md).

## The main window

One window holds one or more **views**. A view is a scatter plot, a 3D viewer and
their shared control panel — see [working in several
views](../how-to/work-in-several-views.md).

### Pane layout

Three buttons sit in the top-right corner of each pane:

| Button | Tooltip | Effect |
|---|---|---|
| :material-square-outline: | **Show only scatter** | One pane fills the window |
| :material-view-split-vertical: | **Stack widgets vertically** | Scatter above, viewer below (the default) |
| :material-view-split-horizontal: | **Place widgets side by side** | Scatter left, viewer right |

Drag the splitter between the panes to change the ratio. **View → Reset Layout**
restores the default stacked 50/50 arrangement.

Each pane also has a chevron toggle for its own sidebar.

### The status bar

Bottom of the window. Shows a running selection count (`Selected: 2,604`) and is
where the [meta staleness banner](../how-to/update-meta-data.md#the-staleness-banner)
appears when a project's snapshot is more than a day old.

## The control panel

The sidebar on the left of the scatter pane. Toggle with ++c++ over the plot, or
**View → Toggle Figure Controls**. Five tabs.

### General

<div class="bc-dense" markdown>

| Group | Control | Does |
|---|---|---|
| **Search** | search field | Find points. Hover the field for the syntax. |
| | **Previous** / **Next** | Step through matches |
| | **Select** | Select all matches |
| **Labels** | `Labels:` | Which meta column is drawn as text on the plot |
| | **Show label counts** | Append the number of points to each label |
| | **Show label outlines** | Outline labels for legibility over dense points |
| **Colors** | `Color by:` | Which meta column drives point colour |
| | `Palette:` | The colour map, with previews |
| | range slider | Clamp the colour range. Three handles (min / centre / max) over a histogram of the values. |
| **Point Size** | `Size by:` | Which meta column drives point size |
| | range slider | Clamp the size range |
| **Scope** | **+ Add filter** | Add a filter row hiding non-matching points. Rows combine with AND/OR; a live count shows how many match. |
| **Selection Behavior** | **Add as group** | Track each addition as a distinct group (on by default) |
| | **Deselect on double-click** | Double-clicking empty space clears the selection |
| | **Deselect on empty selection** | An empty box/lasso clears the selection |
| | **Configure Tab toggle** | Set what ++tab++ flips between — see [shortcuts](shortcuts.md#keys) |

</div>

The column dropdowns have an inline filter box, which matters when your meta
table has a hundred columns.

Labels only appear once you have zoomed in far enough that a manageable number
of points is in view. They are then automatically arranged around their points
to avoid covering points or each other — a label may shift to the other side of
its point, and in very crowded spots labels that cannot be placed without
overlap are hidden until you zoom in further. A short connector line ties each
label to its point. Labels of selected points and of search matches get placed
first. All of this is configurable via [**View → Labels**](menus.md#view):
decluttering and the connector lines can be turned off, and labels that don't
fit can be hidden (the default), shown dimmed (tucked behind the points), or
shown normally. These choices are remembered across sessions (the **Show
Labels** toggle itself is not — it resets to on, like the camera).

As an alternative to labeling every point, **View → Labels → Declutter Mode →
One Label Per Group** shows a single label per unique value instead: the label
sits at the edge of its group of points, kept within the current view, with
connector lines fanning out to all of its points (drawn behind them). If a
value's points form several clearly separated islands, each island gets its
own copy of the label with its own local connector lines — rather than one
label with lines running across the whole view. Labeling every unique value
takes precedence: the extra island labels are only added where space is left,
so they never crowd out another value's only label. Double-clicking a group label
highlights/selects its points just like a regular label.

### Embeddings

Recomputes the scatter plot's layout. See [recompute the
embedding](../how-to/recompute-embeddings.md) for the full parameter reference.

<div class="bc-dense" markdown>

| Group | Control | Does |
|---|---|---|
| | **Re-calculate positions** | Run it |
| | **Auto run** | Recompute on every parameter change |
| **Input** | `Method:` | UMAP, MDS, t-SNE, PaCMAP (PaCMAP needs features; a KNN graph limits you to UMAP and t-SNE) |
| | `Data:` | Which data source to embed |
| **Feature Subset** | checkboxes | One per top-level feature group. Only present with MultiIndex feature columns. |
| **Feature Options** | `Feature metric:` | `cosine`, `euclidean`, `manhattan`, `correlation`, `chebyshev` |
| | `Feature rebalancing:` | `none`, `z-score`, `robust (median/IQR)`, `log1p + z-score` |
| | **Enable PCA** + components | Reduce to N principal components first |
| **Method Settings** | `Random seed:` | Empty means random initialisation |
| | *method-specific* | See [the parameter tables](../how-to/recompute-embeddings.md#parameters) |

</div>

### Fidelity

Answers "is this embedding lying to me?".

<div class="bc-dense" markdown>

| Group | Control | Does |
|---|---|---|
| **Input** | `Data:` / `Metric:` | The ground truth to compare the layout against |
| **Neighborhood Fidelity** | `k neighbors:` | How many neighbours to compare |
| | **Use rank** | Score by neighbour rank rather than raw overlap |
| | **Compute** | Run it. Results become a colourable column. |
| **K-Nearest Neighbors** | `Show:` | `Off`, `Selected only`, `All points` — draws edges to each point's true nearest neighbours |
| | `k neighbors:` | How many edges per point |
| **Evaluate Labels** | **Show:** | Score the current labels |
| | `Method:` | `Silhouette` or `Neighbor consistency` |
| | `Metric:` / `Data:` | What to score against |
| | **Sync w/ labels** | Follow the label column automatically |
| **Distances** | **Show:** + `Threshold:` | Draw lines between points closer than a threshold |

</div>

Drawing KNN edges for **All points** on a large project will slow the plot
noticeably. `Selected only` is usually what you want.

### Cluster

See [reclustering](../how-to/recluster.md) for the full method reference.

<div class="bc-dense" markdown>

| Group | Control | Does |
|---|---|---|
| | **Run** / **Clear** | Run or discard the clustering |
| | folder button | Load cluster assignments from a previously exported CSV |
| **Input** | `Data:` / `Metric:` | What to cluster |
| **Algorithm** | `Method:` | HDBSCAN, Agglomerative, K-Means, Spectral (a KNN graph limits you to HDBSCAN and Spectral) |
| | *method-specific* | See [the method reference](../how-to/recluster.md#choosing-a-method) |
| **Output** | `Result:` | What you got |
| | **Apply labels** | Colour the scatter plot by the clustering |
| | **Export** | Save as CSV with `id` and `bigclust_cluster` |
| **Manual Refinement** | `Target cluster:` + **Set Cluster** | Move the current selection into an existing cluster |
| | **New Cluster** | Move the selection into a fresh cluster |
| | **Reset** | Discard manual edits |

</div>

### Settings

<div class="bc-dense" markdown>

| Control | Does |
|---|---|
| `Render trigger:` | `Continuous`, `Reactive` or `Active Window`. `Reactive` redraws only on change — use it on a laptop on battery. |
| `Max frame rate:` | Cap the frame rate |
| `Font size:` | Label size (also ++left++ / ++right++) |
| `Point scale:` | Marker size (also ++up++ / ++down++) |
| `Max visible labels:` | How many labels may be drawn at once |
| **Cache neurons** + `Max cache size:` | Keep fetched neuron meshes in memory |
| **Clear cache** | Drop the cache |

</div>

## The 3D viewer

A Neuroglancer-style viewer showing the morphology of whatever is selected.
Controls are in [shortcuts](shortcuts.md#3d-viewer). Press ++c++ over the viewer
for its legend, and use its own sidebar toggle for viewer settings.

**View → Synchronize Viewer** (on by default) is what couples the scatter
selection to the viewer. Untick it when working with very large selections — see
[the cost of a big selection](../concepts/selection.md#the-cost-of-a-big-selection).

## Connectivity Table

**View → Connectivity Table** (++shift+cmd+c++). Up- and downstream partners of
the current selection, as a table or a graph.

<div class="bc-dense" markdown>

| Tab / group | Control | Does |
|---|---|---|
| **Table Controls → Rows** | row-label dropdown | `ID` or any meta column |
| | sort dropdown | `No sort`, `By label`, `By distance` |
| | search field | Filter rows by name |
| | **Collapse rows by label** | Aggregate rows sharing a label |
| **→ Columns** | **Upstream** / **Downstream** | Which directions to show |
| | synapse threshold | Hide connections below N synapses |
| | sort dropdown | `No sort`, `By synapse count`, `By label`, `By distance` |
| | search field | Filter columns |
| | **Copy table to clipboard** | |
| **Display** | **Hide zero values** | |
| | **Normalize** | Show fractions rather than raw counts |
| | cell-size slider | Scale cells relative to content |
| | **Always on top** | Keep the window above the main one |
| **Graph** | `Color scheme:` | `Up/Downstream`, `ID`, or any meta column |
| | `Line width:` / `Max partners:` / `Max rows:` | Graph density limits |

</div>

## Distance Heatmap

**View → Distance Heatmap** (++shift+cmd+d++). Pairwise distances within the
selection, as a heatmap with rotated column headers.

Two modes: reading a precomputed matrix, or computing distances for the current
selection from features. The second is capped at 5,000 selected neurons, with a
hint shown when you exceed it. Disabled entirely for [KNN-graph
projects](data-format.md#knn-graph).

<div class="bc-dense" markdown>

| Group | Control | Does |
|---|---|---|
| **Compute** *(feature mode)* | metric dropdown | `euclidean`, `cosine`, `manhattan`, `correlation`, `chebyshev` |
| | **Normalize per neuron** | |
| **Display** | row-label dropdown | `ID` or any meta column |
| | ordering dropdown | `None`, `Label` or `Linkage`. `Linkage` (the default) orders rows and columns by a Ward-linkage dendrogram, which is what makes blocks visible. |
| | **Hide diagonal** | Drop the always-zero diagonal |
| | **Hide upper triangle** | The matrix is symmetric |
| | **Color cells** | Colour by value |
| | decimals spin | Digits shown per cell |
| | cell-size slider | |
| | **Always on top** | |
| | **Copy table to clipboard** | |

</div>

## Feature Comparison

**View → Feature Comparison** (++shift+cmd+f++). Ranks the features separating two
groups. See [finding what separates two groups](../how-to/compare-features.md) for
the full control reference including scoring.

<div class="bc-dense" markdown>

| Group | Control | Does |
|---|---|---|
| **Groups** | **Group A** / **Group B** → **Copy Selection** | Take the current figure selection |
| | **Edit** | Pick members by ID or label |
| | **Eval. separation** | Score whether the split is real |
| **Data** | `Aggregation`, `Top-level filters`, `Min synapse count`, **Normalize per neuron** | Prepare the features |
| **Scoring** | `Metric` | L1 logistic regression or permutation importance |
| **Table** tab | feature filter, `Show Top N:`, `Top N by:` | |
| **Graph** tab | `Orientation:`, **Transpose graph**, `Feature jitter:`, **Show KDE** | |

</div>

Click any table row for a per-feature detail dialog with its own distribution
plot.

## Meta Data Explorer

**View → Meta Data Explorer** (++shift+cmd+m++). The full meta table, filterable,
and the entry point to both meta-update routes.

<div class="bc-dense" markdown>

| Group | Control | Does |
|---|---|---|
| **Filters** | **Add Filter** | Add a column / operator / value row |
| | **Clear Filters** | |
| | `Logic:` | Combine rows with `AND` or `OR` |
| | count label | `Showing n / total rows` |
| | selection-mode combo | `Filtered rows` or `Highlighted rows` |
| | copy combo | `To Clipboard`, `Selected rows`, `IDs only` |
| | **Save** | Write the selected rows to CSV |
| | **Update** ▸ **From Remote…** | [Meta Data Sources](#meta-data-sources) dialog |
| | **Update** ▸ **From Local…** | [Update Meta From Local File](#update-meta-from-local-file) dialog |
| | **Select in Main Window** | Push these rows to the scatter plot as a selection |
| | **Open in New View** | Open them in their own view |

</div>

The **Update** button is the *only* way into the two meta dialogs — they are not
in the menu bar.

## Meta Data Sources

**Meta Data Explorer → Update → From Remote…**. Configures where each dataset's
annotations are read from, and pulls them. **Never writes to a backend.** See
[refreshing meta data](../how-to/update-meta-data.md#route-1-from-a-live-backend).

Table columns: `Dataset`, `Rows`, `Backend`, `Configuration`, `Column mapping`.

| Control | Does |
|---|---|
| backend combo | Pick a [backend](backends.md) per dataset |
| **Edit mapping…** | Map your meta columns onto the source's column names |
| **Auto-map** | Guess the obvious mappings |
| **Save sources** | Store the definitions in the project `info` |
| **Update meta now** | Pull fresh values, on a background thread |

In the mapping dialog, `DO_NOT_UPDATE` is the first choice for every column and
means "leave this alone".

## Update Meta From Local File

**Meta Data Explorer → Update → From Local…**. Merges a local file into the meta
table. Accepts `.csv`, `.tsv`, `.parquet` and `.feather`. See [route
2](../how-to/update-meta-data.md#route-2-from-a-local-file).

| Group | Control | Does |
|---|---|---|
| | `File:` + **Browse…** + **Load** | Read the file and list its columns |
| **Join on** | column list | Which columns to match rows by |
| **Columns to import** | table | `Import`, `File column`, `Import as`, `Status` |
| | **Only fill empty cells** | Never overwrite existing values |
| | **Empty file cells clear existing values** | A blank means "delete" rather than "no opinion" |
| **Sanity check** | **Check merge** | Preview without changing anything |
| | **Apply** | Apply the merge. Disabled until **Check merge** has run. |

Meta rows never change; unmatched file rows are reported and ignored. Nothing is
written to disk.

## Push Annotations

**Selection → Set Annotations** (++cmd+a++ / ++ctrl+a++). The only dialog that
writes to a live backend. See [pushing
annotations](../how-to/push-annotations.md) — read it before using this.

| Tab | Control | Does |
|---|---|---|
| **Configuration** | **Group same backend/repository** | Batch writes sharing a destination |
| | dataset table | `Dataset`, `Selected neurons`, `Backend`, `Configuration` |
| | **Validate** | Connect and check config + credentials. Unlocks the Submit tab. |
| **Submit Annotations** | `Value` | What to write |
| | `Fields` | Comma-separated field names, per dataset |
| | **Clear fields** | Write empty instead of the value |
| | recent field plans | Re-apply one of your last five setups |
| | **Submit** | Write. No further confirmation. |

## Other dialogs

| Dialog | Reached from | Does |
|---|---|---|
| **Open Project** | **File → Open Project** (++cmd+o++) | Pick a project, filter it, choose an embedding strategy |
| **Credentials** | **Window → Credentials…** (macOS: **Preferences**) | Set or clear service tokens — see [credentials](../get-started/credentials.md) |
| **Project Details** | **Window → Show Project Details** | The `info` file as a tree |
| **Annotation Log** | **Window → Show Annotation Log** | Every push this session, as plain text, JSON or CSV |
| **Keyboard Shortcuts** | **Help → Keyboard Shortcuts** | The list on [this page](shortcuts.md) |

!!! note "`has features` / `has distances` in Project Details can read `False` incorrectly"

    Those two rows only inspect the **top level** of the `info` file. In a
    [multiple-embeddings project](data-format.md#multiple-embeddings) where the
    sources are declared per entry, they report `False` even though the sources
    are present and working. Trust the Embeddings tab's `Data:` dropdown instead.
