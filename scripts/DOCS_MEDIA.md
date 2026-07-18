# Screen recordings for the docs

Six short clips are referenced by the documentation. Until each one exists, its
page renders a dashed `bc-clip--todo` placeholder naming the missing file — so a
missing clip is visible rather than silent.

These notes live in `scripts/` rather than under `docs/` because everything under
`docs/` is published as a page, and a shot list is not documentation. The GIFs
themselves go in `docs/_static/media/`, next to `record_docs_media.sh` here.

## Shot list

| File | Page | What to show |
|---|---|---|
| `lasso-selection.gif` | [get-started/first-project](../docs/get-started/first-project.md) | Shift + Ctrl + drag a lasso around one cluster; the 3D viewer fills in behind it. Let the neurons finish loading. |
| `open-remote-project.gif` | [how-to/load-a-remote-dataset](../docs/how-to/load-a-remote-dataset.md) | Paste a URL into the Open Project dialog; Project Details populates; press Load; the plot appears. |
| `recompute-embedding.gif` | [how-to/recompute-embeddings](../docs/how-to/recompute-embeddings.md) | Switch Method from UMAP to t-SNE, press **Re-calculate positions**, points animate to the new layout. The animation is the whole point — do not cut it short. |
| `recluster.gif` | [how-to/recluster](../docs/how-to/recluster.md) | Set an HDBSCAN `Min cluster size`, press **Run**, press **Apply labels**; the scatter recolours. |
| `feature-comparison.gif` | [how-to/compare-features](../docs/how-to/compare-features.md) | Select cluster A → **Copy Selection**; select cluster B → **Copy Selection**; **Refresh**; the ranked table fills. |
| `push-annotations.gif` | [how-to/push-annotations](../docs/how-to/push-annotations.md) | Configuration tab → pick backend → **Validate** → Submit tab enables → enter value and fields → **Submit**. **Use the CSV backend.** Never record against a live database. |

## Recording them

`scripts/record_docs_media.sh` in the repository root wraps the steps below.

**1. Capture.** On macOS, ++cmd+shift+5++ records a window or region to `.mov`.
Record at the window's native size and let the GIF conversion do the scaling —
recording small and scaling up looks terrible.

Keep clips to **8–15 seconds**. Longer than that and nobody watches to the end;
split it into two clips instead.

**2. Convert.** [`gifski`](https://gif.ski/) gives much better results than
ffmpeg's palette filter for the kind of content here (thousands of small coloured
points on black):

```bash
ffmpeg -i raw.mov -vf "fps=15,scale=1280:-1:flags=lanczos" -f yuv4mpegpipe - \
  | gifski -o lasso-selection.gif --fps 15 --quality 90 -
```

Target **under 3 MB** per clip. If you cannot get there, drop the frame rate to
12 or the width to 1000 before dropping quality.

**3. Drop it in** this directory and replace the placeholder in the page:

```html
<figure class="bc-clip" markdown>
  <div class="bc-clip--todo" markdown>
:material-video-outline: &nbsp;**Clip pending:** `lasso-selection.gif`<br>
Shift + Ctrl + drag a lasso around one cluster; the 3D viewer fills in behind it.
  </div>
  <figcaption>Lasso-selecting a cluster.</figcaption>
</figure>
```

becomes

```html
<figure class="bc-clip" markdown>
  <img src="../_static/media/lasso-selection.gif" alt="Lasso-selecting a cluster in the scatter plot.">
  <figcaption>Lasso-selecting a cluster.</figcaption>
</figure>
```

Mind the relative path — pages in `how-to/`, `get-started/` etc. need
`../_static/media/…`.

## Conventions

- **Dark theme, default layout.** The docs' own screenshot uses the stacked
  layout; clips should match so the two read as the same application.
- **Move deliberately.** Pause for a beat before and after each click. A
  recording that races through the interaction is unreadable at 15fps.
- **No real credentials on screen**, and no live annotation backends. The push
  clip uses the CSV backend.
- **Write a real `alt`.** Someone reading with images off should still learn what
  the clip demonstrates.
- **Recycle a project.** Using the same dataset across all clips makes the
  documentation feel like one continuous session rather than six unrelated ones.
