# Load a remote dataset

A BigClust project is a directory, and a directory served over HTTP is a project
you can hand someone as a link. Nothing has to be downloaded first.

This page is about opening someone else's. To build your own, see [create a local
dataset](create-a-local-dataset.md).

## From the command line

```bash
uvx bigclust2@latest --from https://example.org/clusterings/hemibrain_v1.2
```

The window opens with the project already loaded. If the URL points at a
directory containing *several* projects, you get a **Select Project** picker
before the main window appears.

For a real URL to try this with, the [example
dataset](../get-started/example-dataset.md) is public:

```bash
uvx bigclust2@latest --from https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/bigclust_data/examples/MaleCNS_FlyWire_hemibrain_central_brain_bigclust
```

## From the dialog

1. **File → Open Project** (++cmd+o++ / ++ctrl+o++).
2. Type or paste the URL into the box at the top. Do **not** press Browse — that
   is for local folders only.
3. Wait a moment. BigClust fetches the `info` file and fills in **Select
   Project** and **Project Details** from it. If the URL is wrong you find out
   here, before anything large is downloaded.
4. Check **Project Details** — it is the `info` file rendered as a table, so this
   is where you confirm the version, the date and which files the project
   declares.
5. Press **Load**.

<figure class="bc-clip" markdown>
  <div class="bc-clip--todo" markdown>
:material-video-outline: &nbsp;**Clip pending:** `open-remote-project.gif`<br>
Paste a URL into the Open Project dialog; details populate; press Load.
  </div>
  <figcaption>Opening a project from a URL.</figcaption>
</figure>

The last ten locations you opened are kept in the dropdown, so a project you use
regularly is two clicks away on subsequent runs.

## What "remote" means

!!! warning "HTTP and HTTPS only"

    `gs://` and `s3://` URLs do **not** work. They look like URLs to BigClust, so
    they are routed to the HTTP client, which then fails with a schema error
    rather than falling back to anything sensible.

    If your data lives in a bucket, either expose it over HTTPS (both GCS and S3
    can serve public objects over `https://storage.googleapis.com/…` and
    `https://…s3.amazonaws.com/…`) or sync it down locally first.

    `file://` URLs do not work either — use a plain path.

Everything in the project is fetched over HTTP as it is needed: the `info` file
first, then `meta`, then whichever of `embeddings` / `distances` / `features` the
project declares. There is **no local cache** — each run re-downloads. For a
large project on a slow link that is the dominant cost of start-up.

Two ways to avoid paying it repeatedly:

- **Filter at load time.** A [filter](filter-a-dataset.md) is applied while
  reading, so a 5,000-neuron subset of a 300,000-neuron project loads
  correspondingly faster.
- **Take a snapshot.** **Export → Project** writes a complete local copy of
  everything the project references. Open that copy from then on. See
  [exporting](export.md#a-full-project-snapshot).

## Serving a project yourself

Any static file server will do — there is no server-side component. The only
requirements are that the `info` file is reachable at `<url>/info` and that the
files it names resolve relative to it.

For a quick test on your own machine:

```bash
cd /path/to/my_clustering/..
python -m http.server 8000
```

```bash
uvx bigclust2@latest --from http://localhost:8000/my_clustering
```

If you are serving from a different origin than the one the request comes from,
you may need CORS headers; `python -m http.server` does not set them.

## When it won't open

| What you see | What it means |
|---|---|
| `No info file found in selected directory` | The directory exists but has no `info` at its root. Check for a trailing path component — pointing at `…/my_clustering/meta.parquet` will not work. |
| `Malformed info file. The 'info' file must contain valid JSON.` | The `info` file is not valid JSON. The most common cause is copying an example out of the docs — [the specification's examples carry `#` comments](../reference/data-format.md#the-info-file) for readability and are not valid JSON as written. |
| An HTTP error | The `info` URL returned a non-200. Check it in a browser first; `curl -I <url>/info` is faster. |
| It loads, but filtering fails | See [the id-column trap](filter-a-dataset.md#the-id-column-trap). |

Next: [load part of a dataset](filter-a-dataset.md).
