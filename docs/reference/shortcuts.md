# Keyboard and mouse

The same list is available in the app under **Help → Keyboard Shortcuts**.

Modifiers are written with macOS names — see [the conventions
note](index.md#conventions-in-this-documentation) for the mapping to Linux and
Windows.

## General

| Key | Does |
|---|---|
| ++cmd+shift+p++ | Open the [command palette](#command-palette) |

### Command palette

If you only remember one shortcut, make it this one. ++cmd+shift+p++ opens a
search box listing **every** command in the menu bar — type a few letters, press
++enter++ to run the highlighted one.

Matching is fuzzy and skips ahead, so initials work: `gs` finds **Grow
Selection**, `dh` finds **Distance Heatmap**, `csv` finds **Export ▸ Meta Data ▸
To CSV**. You can also search by the menu a command lives in — typing `export`
lists everything under the Export menu.

Each row shows the command's own shortcut on the right, which makes the palette a
practical way to *learn* the shortcuts for the things you do often. Commands you
have run recently are offered first.

Commands that are unavailable right now — the Distance Heatmap before a project
is loaded, say — are still listed, greyed out, rather than hidden, so you can see
that a feature exists and that it is merely waiting on something.

Navigate with ++up++ / ++down++, and ++esc++ to dismiss.

Not everything in the palette comes from the menu bar: a few control-panel
buttons are in there too, listed under the panel they belong to — **Run
UMAP/MDS/t-SNE** and **Compute Neighborhood Fidelity** under *Embedding*, the
clustering **Run** / **Clear** / **Apply** / **Export** actions under
*Clustering*, and so on. The dimensionality-reduction entry follows whichever
method is selected, so it reads "Run t-SNE" when t-SNE is chosen — but searching
`umap` finds it either way.

!!! info "Adding your own"

    Any panel can contribute commands by defining a `palette_commands()` method
    returning `Command` objects; `command_from_button()` wraps an existing
    `QPushButton` in one line. See
    `ScatterControls._PALETTE_BUTTONS` in
    [`bigclust2/gui/controls/scatter_control.py`](https://github.com/flyconnectome/bigclust2/blob/main/bigclust2/gui/controls/scatter_control.py)
    for the pattern. Everything already in the menu bar is picked up
    automatically and needs no registration.

## Scatter plot

Bare-letter shortcuts only fire when the scatter plot has focus, so they never
hijack typing in a filter or search field.

### Mouse

| Gesture | Does |
|---|---|
| left-click + drag | Move the view |
| ++shift++ + drag | Draw a selection box |
| ++shift+cmd++ + drag | Add a box selection to the current selection |
| ++shift+ctrl++ + drag | Draw a lasso selection |
| ++shift+ctrl+cmd++ + drag | Add a lasso selection to the current selection |
| scroll | Zoom |
| double-click a label | Highlight points with the same label |
| ++shift++ + double-click a label | Select points with the same label |
| ++cmd+shift++ + double-click a label | Add same-label points to the selection |

### Keys

| Key | Does |
|---|---|
| ++esc++ | Deselect all points |
| ++shift+c++ | Centre the view on the current selection |
| ++c++ | Toggle the control panel |
| ++l++ | Toggle labels |
| ++tab++ | Flip between two configured property states (Color / Labels) |
| ++space++ | Cycle through embeddings |
| ++right++ | Increase label font size |
| ++left++ | Decrease label font size |
| ++up++ | Increase marker size |
| ++down++ | Decrease marker size |
| ++f++ | Toggle the FPS counter |

++tab++ is configured under **General → Selection Behavior → Configure Tab
toggle**: pick a property (Color by, Labels) and a column for state A and state B,
and ++tab++ flips between them. Comparing two colourings by flicking back and
forth is far more effective than looking at them side by side.

## Selection

| Key | Does |
|---|---|
| ++cmd+a++ | Select all points |
| ++cmd+i++ | Invert the selection |
| ++cmd+plus++ | Grow the selection (add nearest points) |
| ++cmd+minus++ | Shrink the selection (undo the last grow) |
| ++h++ | Hide the selected neurons |
| ++alt+h++ | Show all hidden neurons |
| ++backspace++ | Remove the selected neurons (selection views only) |
| ++cmd+c++ | Copy selected IDs to the clipboard |

!!! note "`Set Annotations` shares a chord with `Select All`"

    **Selection → Set Annotations** is ++cmd+a++ on macOS and ++ctrl+a++
    elsewhere. **Select All** is ++cmd+a++ on macOS only — on other platforms it
    has no shortcut, precisely to avoid the collision.

See [how selection works](../concepts/selection.md) for what growing and shrinking
actually do.

## Views and windows

| Key | Does |
|---|---|
| ++cmd+t++ | Open a new view |
| ++cmd+w++ | Close the current view |
| ++shift+cmd+n++ | Open a new window |
| ++shift+ctrl+w++ | Close the current window |
| ++shift+cmd+ctrl+n++ | Open the current selection in a new view |
| ++cmd+1++ … ++cmd+9++ | Switch to the *n*-th view (9 = last view) |
| ++cmd+m++ | Minimise the window |

## Widgets

| Key | Opens |
|---|---|
| ++shift+cmd+c++ | Connectivity Table |
| ++shift+cmd+d++ | Distance Heatmap |
| ++shift+cmd+f++ | Feature Comparison |
| ++shift+cmd+m++ | Meta Data Explorer |

All four are disabled until a project is loaded.

## Files

| Key | Does |
|---|---|
| ++cmd+o++ | Open Project |

## 3D viewer

| Gesture / key | Does |
|---|---|
| left-click + hold | Rotate the view |
| middle-click + hold | Pan |
| scroll | Zoom |
| ++c++ | Toggle the legend |
| ++1++ | Align view: front |
| ++2++ | Align view: side |
| ++3++ | Align view: top |

Note ++c++ means different things depending on which pane has focus — the control
panel over the scatter plot, the legend over the viewer.
