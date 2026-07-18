# Refresh the meta data

The `meta` table your project shipped with is a snapshot, taken whenever the
project was built. Annotations upstream keep moving. This page pulls current
values in — from a live backend, or from a file someone sent you.

**Both routes are read-only with respect to the backend.** Nothing here writes
anything anywhere. Writing is [pushing annotations](push-annotations.md), which is
a different dialog with a different set of guard rails.

## The staleness banner

If your project declares meta sources and the snapshot is more than a day old — or
carries no date at all — a banner appears in the status bar:

> Project's meta data may be out of date &nbsp; **Update** &nbsp; **Dismiss**

**Update** takes you straight into the refresh below. **Dismiss** hides it for the
session. A project with no declared sources never shows the banner, because there
would be nothing to refresh from.

## Route 1: from a live backend

### Configure the sources

1. **View → Meta Data Explorer** (++shift+cmd+m++).
2. Press **Update** and choose **From Remote…**. The **Meta Data Sources** dialog
   opens with one row per dataset in your project.
3. For each dataset, pick a **Backend** and fill in its **Configuration**. The
   fields change per backend — see the [backend reference](../reference/backends.md).
4. Press **Edit mapping…** to say which source column feeds which of your meta
   columns. **Auto-map** guesses the obvious ones for you.
5. Press **Save sources** to store this setup in the project's `info`, so you do
   not have to redo it. (This writes to the in-memory `info` — see [persisting](#persisting-a-refresh).)

Column mapping is the part worth care. It maps **your** meta column onto the
**source's** column name, which is what lets a project spanning several datasets
line up despite each of them calling the same thing something different —
`soma_side` here, `somaSide` there, `soma` in the third. A mapped column that does
not yet exist in your meta table is created on refresh. Any column you leave on
`DO_NOT_UPDATE` is left alone.

`id` and `dataset` are never mapped and never updated. Neither is anything whose
name starts with `_`.

### Run it

Press **Update meta now**. The pull runs on a background thread, so the window
stays responsive.

Refreshed values land in the in-memory meta table, matched by ID and scoped per
dataset. **The table's shape never changes** — same rows, same order, same index.
Only cell values move. A neuron that has vanished from the backend keeps its old
values rather than disappearing from your project.

If some datasets fail and others succeed you get a **Meta update** warning with a
per-source summary; a total failure gives you **Meta update failed**.

## Route 2: from a local file

Use this when someone sends you a spreadsheet of corrections, or when your
authoritative annotations live somewhere BigClust has no backend for.

1. **View → Meta Data Explorer** (++shift+cmd+m++).
2. Press **Update** and choose **From Local…**. The **Update Meta From Local
   File** dialog opens.
3. Set **File:** to a `.csv`, `.tsv`, `.parquet` or `.feather` file and press
   **Load**. Its columns are listed below.
4. Under **Join on**, pick the columns to match rows by — `id`, or `id` +
   `dataset` for a multi-dataset project.
5. Under **Columns to import**, tick the columns you want and set what each is
   **Import as**.
6. Choose how conflicts resolve:
    - **Only fill empty cells (never overwrite existing values)** — additive
      merge, safest.
    - **Empty file cells clear existing values** — a blank in the file means
      "delete this", rather than "no opinion".
7. Press **Check merge**. This previews the result and changes nothing.
8. Only if the preview looks right, press **Apply**.

**Apply** stays disabled until you have run **Check merge**. This is deliberate:
a merge on the wrong join column can silently rewrite thousands of cells, and the
preview is the only place you will catch it.

As with a remote refresh, your meta rows never change. File rows that match
nothing are reported and ignored — and that report is the most useful thing in
the preview. A join that matches 12 of 4,000 file rows means you picked the wrong
join column.

## Persisting a refresh

!!! warning "Refreshed values live in the session only"

    Neither route writes to your project directory. Close BigClust and the
    refreshed values are gone; the next run reads the same old `meta` file.

    To keep them, run **Export → Project**. That writes a full snapshot —
    including the updated meta table, the refreshed `last_updated` stamps and any
    source definitions you saved — to a directory you choose. Open *that* from
    then on.

    See [exporting](export.md#a-full-project-snapshot).

## Checking what changed

The **Meta Data Explorer** is the fastest way to see the state of things:

- **Column Filters** with AND/OR logic narrow the table down; the count label
  keeps a running `Showing n / total rows`.
- **Select in Main Window** pushes the filtered or highlighted rows to the scatter
  plot as a selection — so "show me every neuron whose type changed" becomes a
  visible group of points.
- **Save** writes the filtered rows to CSV.

Next: [push annotations](push-annotations.md).
