# Push annotations

This is the one operation in BigClust that changes something outside your own
machine. It writes a value into a field, for a set of neurons, in a live
annotation backend — Clio, FlyTable/SeaTable, or a local CSV.

!!! danger "The defaults point at production"

    The backend configurations that ship with BigClust describe **live, shared
    datasets**. There is no sandbox and no dry-run toggle in the UI. A push is
    immediately visible to everyone else using that database, and BigClust has no
    undo.

    If you are testing or developing, point the backend at a scratch table or use
    the `CSV` backend. Do not use the defaults to "see what happens".

## Before you start

You need:

- A **selection** in the scatter plot. The push applies to exactly those neurons.
- **Credentials** for the backend — see [credentials](../get-started/credentials.md).
  You will be prompted if they are missing.
- To know which **field** you are writing and what value goes in it.

## Steps

1. Select the neurons in the scatter plot.
2. **Selection → Set Annotations** (++cmd+a++ on macOS, ++ctrl+a++ elsewhere).
   The **Push Annotations** dialog opens on its **Configuration** tab. The
   **Submit Annotations** tab is disabled, and stays that way until step 5.
3. For each dataset in your selection, pick a **Backend** and complete its
   **Configuration**. The table shows how many of your selected neurons belong to
   each dataset. Datasets you do not want to write to get `Not Writeable`.
4. Leave **Group same backend/repository** ticked unless you have a reason not to
   — it batches writes that share a destination.
5. Press **Validate**. This actually connects to each backend and checks that the
   configuration and your credentials work. On failure you get **Backend
   validation failed** and stay on this tab.
6. The **Submit Annotations** tab is now enabled. Switch to it.
7. Enter the **Value** — for example a cell type like `DA1_lPN`.
8. Under **Fields**, enter the comma-separated field names to write it into, per
   dataset (e.g. `type, flywire_type`).
9. Press **Submit**.

<figure class="bc-clip" markdown>
  <div class="bc-clip--todo" markdown>
:material-video-outline: &nbsp;**Clip pending:** `push-annotations.gif`<br>
Configuration tab → pick backend → Validate → Submit tab enables → enter value and fields → Submit.
  </div>
  <figcaption>The validate-then-submit flow.</figcaption>
</figure>

!!! warning "There is no final confirmation"

    **Submit** writes. The gate is **Validate**, not a "are you sure?" dialog — so
    the moment to check your selection count and your field names is *before* you
    press it, not after.

    Validation is remembered for the whole session, across dialog openings. The
    second push of a session goes straight to Submit with nothing standing
    between you and the write.

## Clearing values

Tick **Clear fields** on the Submit tab. The named fields are set to empty rather
than to the value in the box. The status line changes to `Ready to clear
annotations` so you can tell the two modes apart at a glance.

## Reusing a setup

The **Re-use recent field plans** table keeps your last five field
configurations. Click a row to apply it. This matters more than it sounds — the
per-dataset field lists are the fiddly part, and retyping `type, flywire_type`
for the eleventh cluster of the afternoon is where mistakes come from.

Backend configurations are also remembered between sessions.

## Backend-specific behaviour

Some backends do more than write the field you asked for. Know these before you
push.

**Clio** — with `auto_fix_instances` on (the default), writing `type` *also*
rewrites `instance` to `{type}_{soma_side}`. Clearing `type` clears `instance`
too. This is usually what you want and occasionally very much not; the field's
own tooltip warns to use it with caution. You cannot write `bodyid`.

**FlyWire @ FlyTable** and **Hemibrain @ FlyTable** — writing a user-owned field
also stamps a companion `_source` column with your `user_initials`. Some fields
accept only certain values: `side` must be one of `left`, `right`, `center` or
empty. You cannot write the ID column.

**neuPrint is read-only.** It is available as a [meta
source](update-meta-data.md) but not as a push target; selecting it here will not
give you a writable backend.

**CSV** writes to a local file and is the safe option for practising the flow.

The full field-by-field detail is in the [backend
reference](../reference/backends.md).

## After a push

Writes run on a background thread, batched per backend and field set. You get a
summary — `Successfully pushed annotations for 47 neurons to 2 backends` — or a
per-backend error.

Two things record what happened:

- **Window → Show Annotation Log** lists every push of the session, exportable as
  plain text, JSON or CSV.
- Annotated neurons are marked in the scatter plot, and the marking follows them
  across every view of the project — so you can see at a glance which clusters
  you have already dealt with.

If a push failed partway, check the log before retrying: the batches that
succeeded have already been written, and re-running the whole thing will write
them again.

Next: [export your work](export.md).
