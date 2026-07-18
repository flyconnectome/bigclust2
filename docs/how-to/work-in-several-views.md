# Work in several views

A **view** is one scatter plot plus its 3D viewer and its own control panel.
Views live as tabs, and a tab can be torn off into its own window. They are how
you compare two things without losing your place in either.

## Opening and switching

| Action | Shortcut |
|---|---|
| New view | ++cmd+t++ |
| New window | ++shift+cmd+n++ |
| Close view | ++cmd+w++ |
| Close window | ++shift+ctrl+w++ |
| Switch to the *n*-th view | ++cmd+1++ … ++cmd+9++ (9 = last) |

The tab bar hides itself when there is only one view, so a single-view session
looks exactly as it did before.

Each view gets its own accent colour and a coloured dot in its tab. That colour
is repeated in the window title and in the title bar of every widget the view
owns — which is the only practical way to tell which of three open connectivity
tables belongs to which view.

## Putting a selection in its own view

This is the main reason views exist.

1. Select a group of points in the scatter plot.
2. **Selection → Open in New View** (++shift+cmd+ctrl+n++).

The new view contains only those neurons, titled `selection (247)`. Everything
else is gone — the embedding is redrawn for the subset, so a cluster that was a
smear at full scale becomes structure you can actually see.

Inside a selection view, ++backspace++ (**Selection → Remove from View**) drops
the selected neurons out of it. Work down a cluster by removing what you have
dealt with until the view is empty.

The original view is untouched, so you can always go back.

## Tearing off a window

Drag a tab **vertically** out of the tab bar and it becomes its own window.
Dragging horizontally reorders tabs instead.

Two windows side by side is the natural way to compare a connectivity embedding
with a morphology embedding of the same neurons — one view each, both visible.

**Window → Move View to New Window** does the same thing from the menu, and
**Window → Merge Window into Parent** puts a detached window's views back where
they came from.

## Shared state

Views of the same project are not fully independent. What is shared:

- **Annotation state.** Neurons you have [pushed annotations
  for](push-annotations.md) are marked in every view, so you can see what has
  been dealt with regardless of which view you did it in.
- **The project itself** — the meta table, the data sources, the credentials.

What is not shared: the selection, the embedding, the clustering, the colouring,
the scope filters, the control panel state. Those are per-view, which is what
makes views useful.

## Keeping it manageable

Every view can own a connectivity table, a distance heatmap, a feature comparison
and a meta explorer. Four views means potentially sixteen windows.

**Window → Hide Widgets of Inactive Views** deals with this: with it ticked, the
widgets belonging to background views are hidden and come back when you switch to
that view. The widgets keep their state — they are hidden, not closed.

**View → Reset Layout** restores the default stacked 50/50 arrangement of the
scatter and viewer panes if you have dragged the splitter somewhere unhelpful.

## Turning off the 3D viewer

When you are working with large selections, the 3D viewer is the expensive part —
every selected neuron is a mesh to fetch and render.

**View → Synchronize Viewer** (ticked by default) is what couples the scatter
selection to the viewer. Untick it and selection becomes instant, at the cost of
not seeing morphology. The viewer keeps whatever it last loaded.

For a session that is purely about the scatter plot, the layout button in the
top-right of the scatter pane (**Show only scatter**) gives the plot the whole
window.

---

That is the last how-to. If you want the model behind all of this rather than the
steps, the [concepts pages](../concepts/index.md) are next; if you want the full
inventory of controls, that is the [reference](../reference/index.md).
