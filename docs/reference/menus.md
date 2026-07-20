# Menus

The menu bar, item by item. Shortcuts use macOS modifier names — see [the
conventions note](index.md#conventions-in-this-documentation).

## File

| Item | Shortcut | Does |
|---|---|---|
| **Open Project** | ++cmd+o++ | Open the [project dialog](widgets.md#other-dialogs) |
| **New View** | ++cmd+t++ | Add a view to this window |
| **New Window** | ++shift+cmd+n++ | Open a second window |
| **Close View** | ++cmd+w++ | |
| **Close Window** | ++shift+ctrl+w++ | |
| **Open Recent** ▸ | | Your recent projects, ending in **Clear Menu** |

## View

| Item | Shortcut | Does |
|---|---|---|
| **Command Palette…** | ++cmd+shift+p++ | Search and run any command — see [shortcuts](shortcuts.md#command-palette) |
| **Connectivity Table** | ++shift+cmd+c++ | [Widget](widgets.md#connectivity-table) |
| **Distance Heatmap** | ++shift+cmd+d++ | [Widget](widgets.md#distance-heatmap) |
| **Feature Comparison** | ++shift+cmd+f++ | [Widget](widgets.md#feature-comparison) |
| **Meta Data Explorer** | ++shift+cmd+m++ | [Widget](widgets.md#meta-data-explorer) |
| **Center** ▸ **Scatter** | | Reset the scatter view |
| **Center** ▸ **3D Viewer** | | Reset the viewer camera |
| **Center** ▸ **Selection** | ++shift+c++ | Centre the scatter on the selection |
| **Toggle Figure Controls** | | Show/hide the control panel (also ++c++) |
| **Toggle Viewer Controls** | | Show/hide the viewer sidebar |
| **Reset Layout** | | Restore the default stacked 50/50 panes |
| **Synchronize Viewer** | | *Checkable, on by default.* Couple scatter selection to the 3D viewer |
| **Labels** ▸ **Show Labels** | ++l++ | *Checkable, on by default.* Show/hide the point labels |
| **Labels** ▸ **Declutter Labels** | | *Checkable, on by default.* Automatically arrange labels so they cover neither points nor each other |
| **Labels** ▸ **Declutter Mode** ▸ | | **Individual Labels** (default): every visible point gets its own label. **One Label Per Group**: a single label per unique value, with connector lines fanning out to all its points |
| **Labels** ▸ **Connector Lines** | | *Checkable, on by default.* Draw a short line from each label to the point it belongs to |
| **Labels** ▸ **When Labels Don't Fit** ▸ | | What to do with labels that can't be placed without overlap: **Hide** (default), **Show Dimmed** (faded, drawn behind the points) or **Show Normally** |
| **Show Hoverinfo** | | *Checkable, on by default.* Tooltips on hover |
| **Hover Columns** ▸ | | Which meta columns appear in the tooltip, plus **Show All** / **Hide All** |

The four widget entries are disabled until a project is loaded. The **Labels**
settings (except **Show Labels** itself) are remembered across sessions and
apply to new tabs and windows.

## Selection

| Item | Shortcut | Does |
|---|---|---|
| **Select All** | ++cmd+a++ *(macOS only)* | |
| **Invert Selection** | ++cmd+i++ | |
| **Deselect All** | | Also ++esc++ over the plot |
| **Set Annotations** | ++cmd+a++ / ++ctrl+a++ | [Push annotations](../how-to/push-annotations.md) |
| **Grow Selection** | ++cmd+plus++ | Add nearest points |
| **Shrink Selection** | ++cmd+minus++ | Undo the last grow |
| **Grow/Shrink Options** ▸ | | Data source, feature metric, and how much to grow by |
| **Hide Selection** | ++h++ | |
| **Show Hidden** | ++alt+h++ | |
| **Remove from View** | ++backspace++ | Selection views only |
| **Open in New View** | ++shift+cmd+ctrl+n++ | |
| **Open in Neuroglancer** | | Open the selection in a Neuroglancer scene |
| **Copy to Clipboard** ▸ **IDs** | ++cmd+c++ | |
| **Copy to Clipboard** ▸ **Meta Data** | | |

!!! note "`Select All` and `Set Annotations` share ++cmd+a++ on macOS"

    Not a conflict in practice — `Select All` is only bound on macOS, and
    `Set Annotations` uses ++ctrl+a++ elsewhere. But it does mean the same chord
    does different things depending on your platform.

### Grow/Shrink Options

| Submenu | Contains |
|---|---|
| *(data sources)* | Exclusive group — which source the neighbourhood comes from |
| **Feature Metric** | Only when a feature source is active |
| **Grow By** | `N points` presets plus a custom entry; **Within neighbour distance**; and, in distance mode, a **Distance ×** factor submenu |

## Export

| Item | Produces |
|---|---|
| **Meta Data** ▸ **To Clipboard** | Tab-separated meta table |
| **Meta Data** ▸ **To CSV** | `<project>_meta_data.csv` |
| **Embedding** ▸ **To Plotly** | A self-contained interactive HTML page |
| **Embedding** ▸ **To Dashboard** | The same, with selection-linked panels |
| **Project** | A full project snapshot directory |

See [exporting your work](../how-to/export.md).

## Window

| Item | Shortcut | Does |
|---|---|---|
| **Zoom** | | Toggle maximise |
| **Minimize** | ++cmd+m++ | |
| **Move View to New Window** | | Same as dragging a tab out vertically |
| **Merge Window into Parent** | | Put a detached window's views back |
| **Hide Widgets of Inactive Views** | | *Checkable.* Hide background views' widgets, restoring them with their view |
| **Show Annotation Log** | | Every push this session |
| **Show Project Details** | | The `info` file as a tree |
| **Credentials...** | | [Service tokens](../get-started/credentials.md). On macOS this appears under **BigClust → Preferences**. |

## Help

| Item | Does |
|---|---|
| **Documentation** | Open this site |
| **About** | Version and links. On macOS this appears under **BigClust → About BigClust**. |
| **Check for Updates…** | Query PyPI for a newer release |
| **GitHub Repository** | <https://github.com/flyconnectome/bigclust2> |
| **Report a Problem** | <https://github.com/flyconnectome/bigclust2/issues> |
| **Debug** ▸ **Scatter** / **Viewer** / **All** | *Checkable, per view.* Render debug overlays |
| **Debug** ▸ **Labels…** | Diagnose label placement for the current view: which labels were placed where, and why the rest were dropped (every rejected candidate position with its reason). Copy the report when reporting label issues |
| **Debug** ▸ **Tracebacks** | *Checkable, app-wide.* Include full tracebacks in the log |
| **Keyboard Shortcuts** | The list on [this page](shortcuts.md) |

**Debug → Tracebacks** is what to turn on before [reporting a
problem](troubleshooting.md#reporting-a-problem).
