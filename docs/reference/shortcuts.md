# Keyboard and mouse

The same list is available in the app under **Help → Keyboard Shortcuts**.

Modifiers are written with macOS names — see [the conventions
note](index.md#conventions-in-this-documentation) for the mapping to Linux and
Windows.

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
