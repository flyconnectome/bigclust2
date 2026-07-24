# Create a local dataset

BigClust does not ship with data. Most projects you open will be ones you built,
and a project is just a directory you write four files into — there is no
database, no server and no import step.

This page builds one end to end: an all-by-all NBLAST of a few hundred hemibrain
neurons, with the 3D viewer wired up to the hemibrain segmentation. It is a toy,
but it is a complete toy — every piece a real project has, at a size that
finishes in a couple of minutes.

The [data format reference](../reference/data-format.md) is the specification;
this is the worked example.

!!! tip "Only want to look around?"

    You do not need to build anything to try BigClust — the [example
    dataset](../get-started/example-dataset.md) is a public URL. Come back here
    when you want a project of your own.

## What you will end up with

```
hemibrain_nblast/
    info                    <- what is here and how to read it
    meta.parquet            <- one row per neuron
    embeddings.parquet      <- the 2D coordinates
    distances.parquet       <- the NBLAST distance matrix
    hemibrain_neuropil.ply  <- the brain outline for the 3D viewer
```

## Before you start

You need [`navis`](https://navis-org.github.io/navis/) and
`neuprint-python`. Both are already dependencies of BigClust, so if you have
BigClust you have them; otherwise:

```bash
pip install "navis[all]" neuprint-python
```

You also need a neuPrint token, from
<https://neuprint.janelia.org/account>:

```bash
export NEUPRINT_APPLICATION_CREDENTIALS="eyJhbGciOi..."
```

!!! tip "No token? Use the bundled neurons"

    `navis.example_neurons()` returns five hemibrain skeletons that ship with
    navis, in the same units and ID space as the real thing. Substitute it for
    step 1 and the rest of this page works unchanged, offline. Five neurons is
    too few to embed meaningfully, but it is enough to prove your pipeline runs.

## 1. Fetch the neurons

```python
import navis
import navis.interfaces.neuprint as neu

client = neu.Client("https://neuprint.janelia.org/", dataset="hemibrain:v1.2.1")

# A few olfactory projection neuron types - enough to see structure, small
# enough to finish quickly.
criteria = neu.NeuronCriteria(
    type="^(DA1|DL3|DM1|VA1v|DM4|VM2).*PN.*",
    regex=True,
    status="Traced",
)

nl = neu.fetch_skeletons(criteria, heal=True)
meta_np, _ = neu.fetch_neurons(criteria)

print(len(nl), nl[0].units)
```

```
239 8 nanometer
```

`heal=True` matters: neuPrint skeletons occasionally arrive as several
disconnected fragments, and NBLAST on a fragment is NBLAST on the wrong neuron.

## 2. Get the units right

Note what that printed — **`8 nanometer`**. Hemibrain skeletons come in voxels,
not physical units.

NBLAST is not scale-invariant. Its scoring matrix was trained on neurons measured
in **microns**, so feeding it 8nm voxels means every distance is off by a factor
of 125 and every score collapses toward zero. You get a matrix that looks fine
and means nothing.

```python
nl = nl.convert_units("um")
print(nl[0].units)
```

```
1.0 micrometer
```

This is the single most common way to get a wrong answer in this whole pipeline.
Check the units; do not assume them.

## 3. Run the NBLAST

```python
dps = navis.make_dotprops(nl, k=5, resample=1)
scores = navis.nblast_allbyall(dps, progress=True)
```

`make_dotprops` turns each skeleton into points-plus-tangent-vectors, which is
what NBLAST actually compares. `resample=1` resamples to one point per micron —
now meaningful, because of step 2.

`nblast_allbyall` returns a square DataFrame of **similarity** scores, indexed by
body ID, normalised so a neuron scored against itself is 1.0.

## 4. Turn scores into distances

BigClust wants distances, where **0 means identical**. NBLAST gives you the
opposite, and its matrix is not quite symmetric — NBLAST(a→b) and NBLAST(b→a)
differ slightly.

```python
import numpy as np

sym = (scores + scores.T) / 2          # NBLAST is not symmetric; average it
dist = (1 - sym).clip(lower=0)         # similarity -> distance
np.fill_diagonal(dist.values, 0.0)     # a neuron is at distance 0 from itself

dist.index = dist.index.astype(np.int64)
dist.columns = dist.columns.astype(np.int64)
dist = dist.astype(np.float32)         # halves the file, loses nothing
```

Keeping the body IDs on the index and columns is not cosmetic — it is what lets
BigClust subset the matrix when you [filter at load
time](filter-a-dataset.md#the-id-column-trap).

## 5. Build the meta table

Here is the discipline that keeps a project correct: **derive the row order from
one place and use it everywhere**.

```python
import pandas as pd

ids = dist.index.to_numpy()            # the one true ordering

meta = (
    meta_np.set_index("bodyId")
           .reindex(ids)               # <- reorder to match the distance matrix
           .reset_index()
)

meta = pd.DataFrame({
    "id": ids,
    "label": meta["type"].fillna("unknown").to_numpy(),
    "dataset": "hemibrain",
    "instance": meta["instance"].to_numpy(),
    "status": meta["status"].to_numpy(),
    "soma_side": meta["somaSide"].to_numpy() if "somaSide" in meta else None,
    "n_synapses": (meta["pre"].fillna(0) + meta["post"].fillna(0)).to_numpy(),
})
```

`id`, `label` and `dataset` are the three columns BigClust needs. Everything else
is yours — and worth shipping, because every extra column is another thing you
can colour by, filter on or hover over. `n_synapses` above costs nothing and
immediately lets you ask "are my clusters just sorting neurons by size?".

!!! danger "The `reindex` is the whole ballgame"

    `meta`, `embeddings` and `distances` are matched by **row position**, not by
    joining on `id`. If `meta_np` comes back from neuPrint in a different order
    than the NBLAST matrix — and it will — then without that `reindex` every
    neuron gets another neuron's label, silently, with no error at any point.

    Deriving `ids` from `dist.index` and reindexing everything onto it makes the
    mistake impossible rather than merely unlikely.

## 6. Compute the embedding

You can skip this. If a project ships `distances` but no `embeddings`, the
[Open Project dialog](filter-a-dataset.md#recompute-rather-than-reuse) offers
**calculate from distances** and lays it out for you.

Shipping one is still nicer — the project opens straight into a picture:

```python
from umap import UMAP

emb = UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.1,
    metric="precomputed",             # we already have distances
    random_state=42,                  # reproducible
).fit_transform(dist.values.astype(np.float64))

emb_df = pd.DataFrame(emb.astype(np.float32), columns=["x", "y"])
```

!!! warning "Exactly two columns, and no ID column"

    An embeddings file must have **exactly two** columns. Add the body IDs
    alongside them and loading fails with `Embeddings must have exactly 2
    dimensions`. The IDs are already in `meta`, at the same row positions.

## 7. Write the files

```python
from pathlib import Path

out = Path("hemibrain_nblast")
out.mkdir(exist_ok=True)

meta.to_parquet(out / "meta.parquet", index=False)
emb_df.to_parquet(out / "embeddings.parquet", index=False)
dist.to_parquet(out / "distances.parquet")      # note: index NOT dropped
```

That last line is the exception. `meta` and `embeddings` have a meaningless
integer index worth dropping; `distances` carries the body IDs on its index, and
dropping them breaks filtering.

## 8. Write the `info` file

```python
import json

info = {
    "name": "hemibrain PN NBLAST",
    "dataset": "hemibrain:v1.2.1",
    "description": "All-by-all NBLAST of olfactory projection neurons.",
    "version": "0.1",
    "meta": {"file": "meta.parquet"},
    "embeddings": {"file": "embeddings.parquet"},
    "distances": {
        "type": "morphology",
        "file": "distances.parquet",
        "metric": "nblast",
    },
}

(out / "info").write_text(json.dumps(info, indent=4))
```

The file is named `info`, with no extension. `type` and `metric` are labels for
your own benefit — they are shown in the UI so that six months later you can tell
which of your projects is the NBLAST one.

At this point you have a working project:

```bash
uvx bigclust2@latest --from hemibrain_nblast
```

## Build it with `ProjectBuilder`

Steps 7 and 8 wrote the parquet files and the `info` JSON by hand so you can see
exactly what a project is. Once you have the pieces in memory, `ProjectBuilder`
does that bookkeeping for you — file naming, the `info` schema, and writing a
shared distance matrix only once when several embeddings reference it:

```python
from bigclust2 import ProjectBuilder

project = ProjectBuilder(
    "hemibrain_nblast",
    name="hemibrain PN NBLAST",
    dataset="hemibrain:v1.2.1",
    description="All-by-all NBLAST of olfactory projection neurons.",
)
project.set_meta(meta, color_column="color")     # needs id, label, dataset
project.add_embedding(
    emb_df,                                       # a 2-column x/y DataFrame
    name="morphology (NBLAST)",
    distances=dist,                               # square, IDs on the index
    distances_type="morphology",
    distances_metric="nblast",
)
project.set_neuroglancer(
    source="precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
    neuropil_mesh="hemibrain_neuropil.ply",       # see step 9
)
project.save()
```

That single call replaces steps 7–9. `add_embedding` can be called more than once
for a [multiple-embeddings project](../reference/data-format.md#multiple-embeddings);
`project.register_embeddings()` will landmark-align the extra ones onto the first.
The class is Qt-free, so it also runs in headless data-prep scripts.

!!! tip "Prefer clicking to scripting?"

    **File → Build Project** in the app is a form over this same class — point it
    at your tables and it writes the project for you. See
    [Build Project](../reference/widgets.md#build-project).

## 9. Wire up the 3D viewer

Selecting points now works, but the 3D viewer is empty — nothing has told
BigClust where the neurons live. For hemibrain that is one line you need to add to the `info` file:

```json
"neuroglancer": {
    "source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation"
}
```

Body IDs in your `meta` are segment IDs in that volume, so meshes resolve
directly. This is also the point of using real body IDs as your `id` column
rather than inventing your own.

### The neuropil outline

Neurons floating in empty space are hard to orient. A brain outline fixes that,
and hemibrain's is published — but it needs converting first.

```python
import requests, trimesh

URL = ("https://github.com/navis-org/navis-flybrains/raw/refs/heads/main/"
       "flybrains/meshes/JRCFIB2018Fraw.ply")

r = requests.get(URL, timeout=120)
r.raise_for_status()
Path("JRCFIB2018Fraw.ply").write_bytes(r.content)

mesh = trimesh.load("JRCFIB2018Fraw.ply")
mesh.vertices = mesh.vertices * 8          # 8nm voxels -> nanometres
mesh.export(out / "hemibrain_neuropil.ply")

print((mesh.bounds[1] - mesh.bounds[0]) / 1000)
```

```
[276. 241. 297.]
```

!!! warning "The `* 8` is not optional"

    `JRCFIB2018Fraw` is in **raw 8nm voxels**, but meshes fetched from the
    Neuroglancer source arrive in **nanometres**. Skip the scaling and your
    neuropil is one eighth the size of the neurons inside it — a tiny blob at the
    origin while the neurons sprawl off into the distance.

    The printed extent is the check: **roughly 276 × 241 × 297 microns**. If you
    see numbers around 34 × 30 × 37, you forgot to scale.

This is also why the mesh is downloaded and stored in the project rather than
referenced by URL. `neuropil_mesh` can take a URL, but it would fetch the file
*unscaled*, and there is nowhere to express "multiply by 8" in the `info` file.
Convert once, ship the result.

Now extend the `neuroglancer` block:

```json
"neuroglancer": {
    "source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
    "neuropil_mesh": "hemibrain_neuropil.ply",
    "color": "#ff7043"
}
```

`neuropil_mesh` is resolved relative to the project directory, so the project
stays self-contained and moves as one folder.

## 10. Check it before you trust it

```python
from bigclust2.data import parse_directory

loader = parse_directory("hemibrain_nblast")
compiled = loader.compile()

print(len(compiled["meta"]), compiled["embeddings"].shape, compiled["distances"].shape)
assert list(compiled["distances"].index) == list(compiled["meta"]["id"])
```

That assertion is the row-order check from step 5, verified against what BigClust
actually loaded. It is worth running on every project you build, because it is the
one class of error the application cannot detect for you.

Then open it:

```bash
uvx bigclust2@latest --from hemibrain_nblast
```

## The whole thing

```python
"""Build a BigClust project from an all-by-all NBLAST of hemibrain neurons."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import trimesh
import navis
import navis.interfaces.neuprint as neu
from umap import UMAP

OUT = Path("hemibrain_nblast")
OUT.mkdir(exist_ok=True)

# 1. neurons -----------------------------------------------------------------
client = neu.Client("https://neuprint.janelia.org/", dataset="hemibrain:v1.2.1")
criteria = neu.NeuronCriteria(
    type="^(DA1|DL3|DM1|VA1v|DM4|VM2).*PN.*", regex=True, status="Traced"
)
nl = neu.fetch_skeletons(criteria, heal=True)
meta_np, _ = neu.fetch_neurons(criteria)

# 2. units - NBLAST expects microns, hemibrain ships 8nm voxels ---------------
nl = nl.convert_units("um")

# 3. NBLAST ------------------------------------------------------------------
dps = navis.make_dotprops(nl, k=5, resample=1)
scores = navis.nblast_allbyall(dps, progress=True)

# 4. similarity -> distance --------------------------------------------------
sym = (scores + scores.T) / 2
dist = (1 - sym).clip(lower=0)
np.fill_diagonal(dist.values, 0.0)
dist.index = dist.index.astype(np.int64)
dist.columns = dist.columns.astype(np.int64)
dist = dist.astype(np.float32)

# 5. meta, reindexed onto the distance matrix's ordering ----------------------
ids = dist.index.to_numpy()
m = meta_np.set_index("bodyId").reindex(ids).reset_index()
meta = pd.DataFrame({
    "id": ids,
    "label": m["type"].fillna("unknown").to_numpy(),
    "dataset": "hemibrain",
    "instance": m["instance"].to_numpy(),
    "status": m["status"].to_numpy(),
    "n_synapses": (m["pre"].fillna(0) + m["post"].fillna(0)).to_numpy(),
})

# 6. embedding ---------------------------------------------------------------
emb = UMAP(
    n_components=2, n_neighbors=10, min_dist=0.1,
    metric="precomputed", random_state=42,
).fit_transform(dist.values.astype(np.float64))
emb_df = pd.DataFrame(emb.astype(np.float32), columns=["x", "y"])

# 7. neuropil mesh, scaled from 8nm voxels to nanometres ----------------------
URL = ("https://github.com/navis-org/navis-flybrains/raw/refs/heads/main/"
       "flybrains/meshes/JRCFIB2018Fraw.ply")
raw = OUT / "JRCFIB2018Fraw.ply"
raw.write_bytes(requests.get(URL, timeout=120).content)
mesh = trimesh.load(raw)
mesh.vertices = mesh.vertices * 8
mesh.export(OUT / "hemibrain_neuropil.ply")
raw.unlink()

# 8. write -------------------------------------------------------------------
meta.to_parquet(OUT / "meta.parquet", index=False)
emb_df.to_parquet(OUT / "embeddings.parquet", index=False)
dist.to_parquet(OUT / "distances.parquet")   # keep the index: it holds the IDs

(OUT / "info").write_text(json.dumps({
    "name": "hemibrain PN NBLAST",
    "dataset": "hemibrain:v1.2.1",
    "description": "All-by-all NBLAST of olfactory projection neurons.",
    "version": "0.1",
    "meta": {"file": "meta.parquet"},
    "embeddings": {"file": "embeddings.parquet"},
    "distances": {"type": "morphology", "file": "distances.parquet",
                  "metric": "nblast"},
    "neuroglancer": {
        "source": ("precomputed://gs://neuroglancer-janelia-flyem-hemibrain"
                   "/v1.2/segmentation"),
        "neuropil_mesh": "hemibrain_neuropil.ply",
        "color": "#ff7043",
    },
}, indent=4))

# 9. verify ------------------------------------------------------------------
from bigclust2.data import parse_directory
c = parse_directory(str(OUT)).compile()
assert list(c["distances"].index) == list(c["meta"]["id"]), "row order mismatch!"
print(f"wrote {OUT} - {len(c['meta'])} neurons")
```

## Where to go next

**Add features.** This project ships distances, so BigClust can re-embed and
recluster it — but it cannot rank *what* makes two groups different, because
distances have thrown that away. Add a `features` table (connectivity vectors,
say) and the [Feature Comparison widget](compare-features.md) comes alive. See
[the data model](../concepts/data-model.md) for why the two are not
interchangeable.

**Add a second embedding.** Put morphology and connectivity side by side as
[multiple embeddings](../reference/data-format.md#multiple-embeddings) and press
++space++ to flip between them. A cluster that holds in both is a much stronger
claim than one that holds in either.

**Scale up.** A full distance matrix grows with the *square* of your neuron
count — 100,000 neurons is a 10-billion-cell matrix. Past a few tens of thousands,
ship a [KNN graph](../reference/data-format.md#knn-graph) instead.

**Publish it.** A project directory served over HTTP is a project anyone can open
with a link. See [loading a remote dataset](load-a-remote-dataset.md#serving-a-project-yourself).
