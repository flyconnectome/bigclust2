# How selection works

Selection is the central verb. Almost everything you do in BigClust is: select
some points, then ask a question about them. Understanding what selection
propagates to — and what it costs — is most of understanding the app.

## One selection, every view

A selection made in the scatter plot is immediately the subject of everything
else in that view:

- The **3D viewer** loads the morphology of the selected neurons.
- The **connectivity table** shows their partners.
- The **distance heatmap** shows distances among them.
- The **feature comparison** takes them as a group.
- **Set Annotations** writes to exactly them.
- **Copy to Clipboard** copies exactly them.

There is no separate "apply" step and no per-widget selection to keep in sync.
This is what makes the app feel like a single instrument rather than a collection
of panels: you point at something once.

The selection is **per view**. Two views of the same project have independent
selections, which is what makes [comparing two things](../how-to/work-in-several-views.md)
possible.

## Making a selection

| Gesture | Effect |
|---|---|
| ++shift++ + drag | Box selection |
| ++shift+ctrl++ + drag | Freehand lasso |
| add ++cmd++ to either | Add to the current selection instead of replacing it |
| double-click a label | Highlight points sharing that label |
| ++shift++ + double-click a label | Select points sharing that label |
| ++cmd+shift++ + double-click a label | Add those points to the selection |
| ++cmd+a++ | Select all |
| ++cmd+i++ | Invert |
| ++esc++ | Deselect all |

The label gestures are worth internalising. If your `meta` table already has cell
types, shift + double-clicking a type name selects every member of it — which is
the fastest possible way to ask "where does this known type sit in my embedding,
and is it in one place?".

`Add as group` in the General tab (on by default) means each new addition is
tracked as a distinct group rather than merged into an undifferentiated set.

## Growing and shrinking

++cmd+plus++ grows the selection, ++cmd+minus++ shrinks it back.

Growing pulls in the nearest points to what is already selected — in the
**original high-dimensional space**, not in the 2D plot. This distinction matters.
Growing does not simply take the points that look nearby on screen; it takes the
neurons that are actually most similar, which may include points the embedding has
placed elsewhere. If growing repeatedly pulls in something from across the plot,
that is your embedding telling on itself.

**Selection → Grow/Shrink Options** controls how:

- Which **data source** the neighbourhood is computed from, and with which
  **feature metric**.
- **Grow By** — a fixed number of points per step, or everything within a
  neighbour distance, optionally scaled by a **Distance ×** factor.

Shrinking undoes the last grow rather than eroding the boundary, so
grow-grow-shrink returns you exactly to where one grow left you.

This is the intended way to work outward from a seed. Select one neuron you are
sure about, grow until the additions stop making sense, and you have delineated a
type by hand with the data doing the work.

## Hiding rather than deselecting

Three different operations that look similar:

| Operation | Effect | Reversible |
|---|---|---|
| ++esc++ | Clears the selection; points stay | — |
| ++h++ (**Hide Selection**) | Points stay loaded but are not drawn | ++alt+h++ shows all hidden |
| ++backspace++ (**Remove from View**) | Points leave the view entirely | No — reopen the view |
| **Scope filters** (General tab) | Points hidden by a rule, not by hand | Yes, edit or remove the filter |

Hiding is how you work through a large dataset: deal with a cluster, hide it, and
what remains is what you have not looked at yet. Scope filters do the same thing
declaratively — "hide everything that already has a type" is a filter, not a
gesture.

**Remove from View** only makes sense in a [selection
view](../how-to/work-in-several-views.md#putting-a-selection-in-its-own-view),
which is why it is bound to ++backspace++ there.

## The cost of a big selection

Selection is cheap. Its consequences are not.

Every selected neuron is a mesh the 3D viewer has to fetch from a Neuroglancer
source and render. Selecting 50,000 points asks for 50,000 meshes. BigClust
confirms before very large selections for this reason.

Two ways to work with big selections comfortably:

- **View → Synchronize Viewer** — untick it and the scatter selection stops
  driving the 3D viewer. Selection becomes instant. Use this whenever the question
  is about *which* neurons rather than what they look like.
- The **Settings** tab has a neuron cache with a size limit and a **Clear cache**
  button. Caching helps a lot when you keep returning to the same groups.

## Why plain letters are shortcuts

++h++, ++l++, ++c++, ++shift+c++ and ++backspace++ are bound as bare keys, which
would normally be hostile — they would hijack typing in a search box.

They are scoped to the canvas widget rather than the whole window, so they only
fire when the scatter plot has focus. Typing `hello` into the Meta Data Explorer's
filter field does not hide your selection. The menu entries exist so the shortcuts
are discoverable; the canvas is where they actually fire.

---

That is the last concepts page. For the full inventory of controls, see the
[reference](../reference/index.md).
