# Annotation backends

Backends are how BigClust talks to annotation databases. They serve two different
flows:

- **Reading** — [refreshing meta data](../how-to/update-meta-data.md) from a live
  source. Always safe.
- **Writing** — [pushing annotations](../how-to/push-annotations.md). Not safe.
  Read that page before using it.

!!! danger "Default configurations point at live, shared datasets"

    Every backend below ships with defaults describing production databases. A
    write is immediately visible to everyone else and there is no undo.

    For testing, use the `CSV` backend or point at a scratch table.

## At a glance

| Backend | Read | Write | Credentials |
|---|---|---|---|
| [`neuPrint`](#neuprint) | ✅ | ❌ | `NEUPRINT_APPLICATION_CREDENTIALS` |
| [`Clio`](#clio) | ✅ | ✅ | Clio token (`clio-py`) |
| [`FlyTable`](#flytable) | ✅ | ✅ | `SEATABLE_TOKEN` |
| [`FlyWire @ FlyTable`](#flywire-flytable) | ✅ | ✅ | `SEATABLE_TOKEN` |
| [`Hemibrain @ FlyTable`](#hemibrain-flytable) | ✅ | ✅ | `SEATABLE_TOKEN` |
| [`CSV`](#csv) | ✅ | ✅ | none |

See [credentials](../get-started/credentials.md) for how to supply tokens.

## neuPrint

**Read-only.** Available as a [meta source](../how-to/update-meta-data.md) but not
as a push target — attempting to write raises, and field validation always fails.

| Config field | Default | Notes |
|---|---|---|
| `dataset` | — | e.g. `hemibrain:v1.2.1` |
| `server` | `https://neuprint.janelia.org` | |

Get a token from <https://neuprint.janelia.org/account>.

## Clio

Requires the `clio` extra:

```bash
uvx --from "bigclust2[clio]" bigclust2
```

| Config field | Default | Notes |
|---|---|---|
| `dataset_name` | — | e.g. `CNS` |
| `auto_fix_instances` | `true` | See below |

**Cannot write:** `bodyid`.

!!! warning "`auto_fix_instances` rewrites a field you did not name"

    With it on (the default), writing `type` **also** rewrites `instance` to
    `{type}_{soma_side}` — using `soma_side` if present, otherwise `root_side`.
    Clearing `type` clears `instance` too.

    This is usually what you want, since the two fields are meant to stay
    consistent. It is not what you want if `instance` carries information that is
    not derivable from the type. Turn it off in that case.

Get a token from <https://clio.janelia.org/settings>. Clio can also fall back to
an authenticated `gcloud` on your `PATH`.

## FlyTable

The generic SeaTable backend.

| Config field | Default | Notes |
|---|---|---|
| `table_name` | — | The table to read/write |
| `base_name` | — | Optional |
| `id_column` | — | Which column holds the neuron ID |
| `server` | `https://flytable.mrc-lmb.cam.ac.uk` | Hidden in the UI |

**Cannot write:** `root_id`, `root_783`, or whatever you set as `id_column`.

## FlyWire @ FlyTable

A preconfigured FlyTable for FlyWire. Only `user_initials` and `id_column` are
shown in the UI; the rest are fixed.

| Config field | Default |
|---|---|
| `user_initials` | — (**required**) |
| `table_name` | `info` |
| `base_name` | `main` |
| `id_column` | `root_783` |

**Side effect:** writing any of `cell_type`, `hemibrain_type` or `malecns_type`
also writes a companion `_source` column stamped with your `user_initials`. This
is intentional — it records who made each call.

**Value restrictions:** `side` accepts only `left`, `right`, `center` or empty.
`dimorphism` accepts only a fixed set of values.

## Hemibrain @ FlyTable

The same pattern for hemibrain.

| Config field | Default |
|---|---|
| `user_initials` | — (**required**) |
| `table_name` | `hb_info` |
| `base_name` | `hemibrain` |
| `id_column` | `bodyId` |

**Side effect:** writing `type_corrected` also stamps its `_source` column.

**Value restrictions:** `side` accepts only `left`, `right`, `center` or empty.

## CSV

Reads and writes a local file. No credentials, no network, no shared state.

| Config field | Notes |
|---|---|
| `filepath` | Path to the CSV |

**Cannot write:** `id`.

This is the backend to use when you are learning the [push
flow](../how-to/push-annotations.md), testing a field mapping, or working
somewhere BigClust has no native backend for. The output is a plain CSV you can
diff, review and import elsewhere.

## Declaring a backend in `info`

Backends used as **meta sources** are declared per dataset in the project's `info`
file, so a project can carry its own refresh configuration:

```json
"meta": {
    "file": "meta.parquet",
    "sources": {
        "hemibrain": {
            "backend": "neuPrint",
            "config": { "dataset": "hemibrain:v1.2.1" },
            "columns": { "label": "type", "soma_side": "somaSide" }
        }
    }
}
```

The `backend` value is the name exactly as it appears in the table above,
including capitalisation and spacing (`FlyWire @ FlyTable`). See [meta
sources](data-format.md#meta-sources).

Push targets are **not** stored in the project — they are configured per session
in the Push Annotations dialog and remembered in your local settings. A project
you hand someone cannot cause them to write anywhere.
