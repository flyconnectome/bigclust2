# Export your work

Everything you do in BigClust — recomputed embeddings, clusterings, refreshed
meta data, manual refinements — lives in the session. This page is how it leaves.

## What is where

| You want | Do this | You get |
|---|---|---|
| The selected IDs | **Selection → Copy to Clipboard → IDs** (++cmd+c++) | Newline-separated IDs |
| The selected rows | **Selection → Copy to Clipboard → Meta Data** | The meta rows for the selection |
| The whole meta table | **Export → Meta Data → To CSV** | `<project>_meta_data.csv` |
| The whole meta table, pasteable | **Export → Meta Data → To Clipboard** | Tab-separated, ready for a spreadsheet |
| A shareable interactive plot | **Export → Embedding → To Plotly** | A single self-contained HTML file |
| The same, with side panels | **Export → Embedding → To Dashboard** | HTML with selection-linked panels |
| A cluster assignment | Cluster tab → **Export** | CSV with `id` and `bigclust_cluster` |
| A filtered meta subset | Meta Data Explorer → **Save** | `selected_meta.csv` |
| A feature ranking | Feature Comparison → export | `feature_ranking.csv` |
| **Everything, reopenable** | **Export → Project** | A complete project directory |

## A full project snapshot

**Export → Project** is the one that matters. Pick a destination folder and
BigClust writes a complete, self-contained project there: the `info` file as it
currently stands in memory, plus every data file it references, downloading
remote ones as it goes.

That "as it currently stands in memory" is the point. A snapshot captures:

- Meta data you [refreshed from a backend or merged from a
  file](update-meta-data.md), which otherwise vanishes at exit.
- Meta source definitions you saved, and their `last_updated` stamps.
- The project as a whole, so a slow remote project becomes a fast local one.

Open the exported directory from then on and you pick up where you left off.

The `neuroglancer` block is preserved verbatim rather than being downloaded —
meshes and segmentation stay remote, which is what keeps a snapshot to a sensible
size.

If the destination exists and is not empty, you are asked before anything is
overwritten. A download that fails partway removes its partial file rather than
leaving a truncated one behind.

!!! note "A snapshot does not capture everything"

    Cluster assignments and recomputed embeddings are **not** included — they are
    session state, not project files. Export those separately (Cluster tab →
    **Export**) if you want them.

## Sharing a plot with someone who doesn't have BigClust

**Export → Embedding → To Plotly** writes one HTML file containing your current
embedding, with the colouring and labels you have set up. It opens in any
browser, zooms and pans, and shows hover info. Plotly itself is loaded from a
CDN, so the file is small but needs a network connection to render.

**To Dashboard** adds panels linked to the selection, so the recipient can pick
points and see what they are.

This is the right way to put a figure in a slide deck or send a colleague
something to look at. It is not a substitute for sharing the project directory —
the HTML is a static picture of one embedding, with no data behind it.

## Reproducing a session

There is no single "save my session" file. To pick up exactly where you left off,
the reliable combination is:

1. **Export → Project** — the data, including any refreshed meta.
2. Cluster tab → **Export** — the cluster assignments, reloadable with the folder
   button next to **Run**.
3. Note the Embeddings tab settings you used, including the `Random seed:`. With
   the seed fixed, a recompute is reproducible.

Next: [work in several views](work-in-several-views.md).
