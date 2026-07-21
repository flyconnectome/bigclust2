# Load part of a dataset

Filtering happens at load time, in the **Open Project** dialog. The filter is
applied while the files are read, so a subset loads proportionally faster and
uses proportionally less memory — this is the main lever you have on start-up
time for a large project.

## Steps

1. **File → Open Project** (++cmd+o++ / ++ctrl+o++) and pick your project as
   usual.
2. In the **Options** box, type a query into **Filters**.
3. Press the **?** button next to it to see which columns your project actually
   has — it lists them from the loaded `meta` table, so you are not guessing.
4. Set **Embeddings** to `calculate from distances` or `calculate from features`
   (see [below](#recompute-rather-than-reuse)).
5. Press **Load**.

Your last filter is remembered and pre-filled next time.

## Filter syntax

Two forms are accepted.

**A query expression** over your meta columns:

```
superclass == "central"
```

```
superclass == "central" & side != "left"
```

```
cell_class in ("ALPN", "ALLN") | flow == "intrinsic"
```

| Supported | |
|---|---|
| Comparisons | `==` `!=` `>` `>=` `<` `<=` |
| Membership | `in (...)` or `in [...]` |
| Combination | `&` / `and`, `\|` / `or` |
| Grouping | parentheses |
| Literals | single- or double-quoted strings, integers, floats |

Comparisons bind tightest, then `&`, then `|` — so `a == 1 & b == 2 | c == 3`
means `(a == 1 & b == 2) | c == 3`. Parenthesise anything you would have to think
about.

**A bare list of IDs**, comma-separated:

```
1234567, 2345678, 3456789
```

!!! warning "The ID list is digits-only"

    This form is recognised by checking that every entry is numeric. If your
    project uses non-numeric IDs, a pasted list will be parsed as a *query* and
    fail. Use `id in ("abc", "def")` instead.

## Recompute rather than reuse

The **Embeddings** dropdown offers three choices, and the dialog nudges you
toward the last two whenever a filter is set:

| Choice | What you get |
|---|---|
| `use precomputed` | The 2D coordinates from the project's `embeddings` file, subset to your rows |
| `calculate from distances` | A fresh UMAP over the filtered distance matrix |
| `calculate from features` | A fresh UMAP over the filtered feature matrix |

Reusing precomputed coordinates for a filtered subset is usually the wrong
choice, and it is worth understanding why. The published embedding was laid out
to separate *everything* in the project. When you keep 2% of the points, that
layout is still answering the old question: your neurons stay where they were,
scattered across a plot whose structure was determined by 98% of points that are
no longer there. Recomputing asks the layout question again, over the subset you
actually care about, and the result is generally far more legible.

The load-time recompute uses fixed UMAP settings (`n_neighbors=10`,
`min_dist=0.1`, `spread=1`, `random_state=42`). They are deliberately not
adjustable here — get into the app first, then tune properly in the
[Embeddings tab](recompute-embeddings.md), which is what the dialog's hint is
pointing at.

## The id-column trap

!!! warning "Filtering can fail on a project that opens fine unfiltered"

    When a filter is active, BigClust has to subset the `distances` and
    `features` matrices by row *and* column. To do that it needs to find the ID
    column in those files, and it will only look at the **first or last**
    column — and only under the names `id`, `index` or `__index_level_0__`.

    A project whose distance matrix has its ID column somewhere in the middle
    loads perfectly with no filter and fails the moment you set one, with:

    ```
    Distance file must have the 'id', 'index' or '__index_level_0__' column as the
    first or last column for correct filtering
    ```

    The fix is on the data side — rewrite the file with the index preserved
    properly. See [the data format notes on
    `distances`](../reference/data-format.md#distances).

## Narrowing down after loading

Filtering at load time is permanent for that session. If you want to *explore*
subsets, do it in the app instead:

- **Scope filters** in the control panel's General tab (**+ Add filter**) hide
  points without unloading them. Reversible, stackable with AND/OR, and the
  match count updates as you type. To find the gaps in a column, either uncheck
  everything but *(empty)* (few distinct values) or set the selector next to the
  substring field to `empty` — `non-empty` gives you the complement.
- **Hide Selection** (++h++) removes the current selection from view;
  ++alt+h++ brings everything back.
- **Open in New View** (++shift+cmd+ctrl+n++) puts the current selection in its
  own tab with everything else gone. See [working in several
  views](work-in-several-views.md).

Use a load-time filter when the full project is too big to hold; use scope
filters and views when it isn't.

Next: [recompute the embedding](recompute-embeddings.md).
