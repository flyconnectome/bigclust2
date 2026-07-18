# Installation

BigClust needs **Python 3.10 or newer**. The recommended route does not ask you
to have Python at all.

=== "uvx (recommended)"

    ```bash
    uvx bigclust2@latest
    ```

    [`uv`](https://docs.astral.sh/uv/getting-started/installation/) downloads
    BigClust and its dependencies into a throwaway environment, fetches a
    suitable Python if you don't have one, and runs it. Nothing is installed into
    your system or your active environment.

=== "uv (persistent install)"

    ```bash
    uv tool install bigclust2
    bigclust2
    ```

    Same isolation, but the environment is kept, so start-up is instant after the
    first run. Upgrade with `uv tool upgrade bigclust2`.

=== "pip"

    ```bash
    pip install bigclust2
    bigclust2
    ```

    Works, but pulls a large dependency tree (PySide6, pygfx, numba,
    cloud-volume, navis) into whatever environment is active. Use a dedicated
    virtualenv unless you have a reason not to.

=== "From source"

    ```bash
    git clone https://github.com/flyconnectome/bigclust2
    cd bigclust2
    uv run run.py --debug
    ```

    `run.py` is the development entry point. Note it only understands `--debug` —
    it does not go through the argument parser, so [`--from`](../reference/cli.md)
    is not available there. Use `uv run bigclust2 --from …` if you need it.

## Why `@latest`

```bash
uvx bigclust2@latest
```

Without `@latest`, `uvx` will happily reuse a cached older version indefinitely.
With it, you always start the current release.

The cost is start-up time: whenever a new version has been published, `uvx` has
to resolve and download it before the window appears. If that gets annoying, pin
instead:

```bash
uvx bigclust2@0.1.1          # a specific release
```

To run the development version straight from the repository:

```bash
uvx --from git+https://github.com/flyconnectome/bigclust2@main bigclust2
```

The `@main` is doing the same job as `@latest` above, and has the same
trade-off. Pin to a commit if you want reproducibility:

```bash
uvx --from git+https://github.com/flyconnectome/bigclust2@02ea911 bigclust2
```

## Optional extras

One extra exists:

```bash
uvx --from "bigclust2[clio]" bigclust2
```

`clio` pulls in [`clio-py`](https://github.com/schlegelp/clio-py), which is
needed to read from or write to [Clio](../reference/backends.md#clio). Everything
else — neuPrint, SeaTable/FlyTable, CSV — is covered by the core dependencies.

## Check it worked

```bash
uvx bigclust2@latest --version
```

```
bigclust2 0.1.1
```

That is the fastest confirmation that the install resolved, since it exits before
any GUI or GPU code is touched.

To confirm the *graphics* stack works you have to actually draw something, so
open the [example dataset](example-dataset.md):

```bash
uvx bigclust2@latest --from https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/bigclust_data/examples/MaleCNS_FlyWire_hemibrain_central_brain_bigclust
```

You should get a window with a dense scatter plot of ~87,000 points. If the
window appears but stays empty, that is a graphics problem rather than an install
one — see [troubleshooting](../reference/troubleshooting.md#installing-and-starting).

## Updating

BigClust checks PyPI for a newer release on start-up and shows an **Update
Available** dialog if it finds one. You can also check on demand from
**Help → Check for Updates…**.

How you act on that depends on how you installed:

| Installed with | Update by |
|---|---|
| `uvx bigclust2@latest` | nothing to do — the next run picks it up |
| `uv tool install` | `uv tool upgrade bigclust2` |
| `pip` | `pip install --upgrade bigclust2` |
| a clone | `git pull` |

## Troubleshooting

!!! warning "`realpath: command not found` on macOS"

    If `uvx` fails with an error mentioning `realpath`, your macOS is older than
    13.x. Upgrade the OS — this is a `uv` bootstrap issue, not a BigClust one.

For anything else, see [troubleshooting](../reference/troubleshooting.md).

Next: [your first project](first-project.md).
