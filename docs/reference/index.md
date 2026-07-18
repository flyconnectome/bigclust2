# Reference

Look-up material. For "how do I do X", see the [how-to
guides](../how-to/index.md); for "why is it built this way", the
[concepts](../concepts/index.md).

<div class="bc-grid" markdown>

<div class="bc-card" markdown>

### :material-file-tree: [Data format](data-format.md)

The complete `info` file specification and the layout of every data file a
project can contain. This is what you need to *build* a project.

</div>

<div class="bc-card" markdown>

### :material-keyboard: [Keyboard and mouse](shortcuts.md)

Every shortcut, grouped by where it applies.

</div>

<div class="bc-card" markdown>

### :material-view-dashboard-outline: [Widgets](widgets.md)

Every panel, tab and window, control by control.

</div>

<div class="bc-card" markdown>

### :material-menu: [Menus](menus.md)

The menu bar, item by item, with shortcuts.

</div>

<div class="bc-card" markdown>

### :material-console: [Command line](cli.md)

`bigclust2` flags and what they do.

</div>

<div class="bc-card" markdown>

### :material-database-arrow-up-outline: [Annotation backends](backends.md)

Clio, neuPrint, FlyTable and CSV — configuration fields, what each can write, and
the side effects.

</div>

<div class="bc-card" markdown>

### :material-lifebuoy: [Troubleshooting](troubleshooting.md)

Error messages and what they mean.

</div>

</div>

## Conventions in this documentation

Keyboard shortcuts are written with the **macOS** modifier names, since that is
where most of BigClust's development happens. On Linux and Windows:

| Written | macOS | Elsewhere |
|---|---|---|
| ++cmd++ | ⌘ Command | ++ctrl++ Control |
| ++ctrl++ | ⌃ Control | ++ctrl++ Control |
| ++alt++ | ⌥ Option | ++alt++ Alt |

Where the two platforms genuinely differ — and there are a couple of places — it
is called out on the page.

Names of user-interface elements are written in **bold** and quoted exactly as
they appear in the app, so you can search for them. Column names, file names and
values are written in `code`.

## Not documented here

There is no API reference. BigClust is an application, and its internal modules
are not a supported interface — they change without notice. If you want to build a
project directory programmatically, write the files described in [the data
format](data-format.md) with pandas or pyarrow; that format *is* the stable
contract.
