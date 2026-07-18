# Get started

Four pages, in order. The first three are enough to open a dataset and look at
it; the last only matters once you want to write annotations back somewhere.

!!! tip "In a hurry?"

    ```bash
    uvx bigclust2@latest --from https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/bigclust_data/examples/MaleCNS_FlyWire_hemibrain_central_brain_bigclust
    ```

    That opens a public [example dataset](example-dataset.md) — 87,000 central
    brain neurons co-clustered across three connectomes. No account, no
    credentials, nothing to build.

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### :material-download: [Installation](installation.md)

`uvx bigclust2@latest` and you are running. What `uvx` actually does, how to pin
a version, and how to run from a clone.

</div>

<div class="bc-card" markdown>

### :material-database-outline: [The example dataset](example-dataset.md)

A public co-clustering of FlyWire, MaleCNS and Hemibrain neurons you can open
with one command. The fastest way to see what the app does.

</div>

<div class="bc-card" markdown>

### :material-cursor-default-click: [Your first project](first-project.md)

Open a dataset, find your way around the window, select some points and see the
neurons behind them.

</div>

<div class="bc-card" markdown>

### :material-key-outline: [Credentials](credentials.md)

Tokens for neuPrint, FlyTable and Clio — needed for refreshing meta data and for
pushing annotations, and for nothing else.

</div>

</div>

## What you need before you start

**Something to open.** BigClust opens a directory — local or served over
HTTP — containing an `info` file and at minimum a `meta` table. It does not open
a bare `.parquet` or a distance matrix you hand it.

You do not need one of your own to start: the [example
dataset](example-dataset.md) is a public URL and needs nothing but the command
above. When you are ready to build your own, [create a local
dataset](../how-to/create-a-local-dataset.md) walks through a complete worked
example, [the data format reference](../reference/data-format.md) is the
specification behind it, and [why data sources are
directories](../concepts/data-sources.md) explains the reasoning.

**Python 3.10 or newer**, though if you install with `uvx` you never have to
think about it — `uv` fetches an interpreter itself.

**A GPU that can run WebGPU.** The scatter plot and the 3D viewer are both
rendered with [pygfx](https://github.com/pygfx/pygfx) on
[wgpu](https://github.com/pygfx/wgpu-py). Any machine from the last decade is
fine; a headless server without a display is not.

Nothing else. No account, no configuration file, no credentials — those only
enter the picture at the point where BigClust needs to *talk* to an annotation
backend, which is [refreshing meta data](../how-to/update-meta-data.md) and
[pushing annotations](../how-to/push-annotations.md).

Next: [installation](installation.md).
