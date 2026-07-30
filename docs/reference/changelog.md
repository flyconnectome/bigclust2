# Changelog

Notable changes to BigClust, newest first. Versions follow
[semantic versioning](https://semver.org/); the format is loosely based on
[Keep a Changelog](https://keepachangelog.com/).

For the complete commit-level history, see the
[releases on GitHub](https://github.com/flyconnectome/bigclust2/releases).

## Unreleased

### Fixed

- Meta refresh counts are now reported as **values** (row/column intersections)
  alongside the number of rows they sit in — "cells" read as the neurons
  themselves. The share that merely populated a newly added column is called out
  separately, since that fills every matched row at once and can dwarf the real
  edits.
- A value whose source representation differs but which converts back to what is
  already stored (a source sending `"111"` for an integer column) is no longer
  counted as a change. It never reached the table, but it was reported on every
  single run.

### Changed

- `import bigclust2` no longer loads the GUI stack. Qt, pygfx and wgpu are
  imported the first time the app itself is started, so the Python helpers
  (`ProjectBuilder`, `update_project_meta`) and `bigclust2 update` now work — and
  start noticeably faster — on machines with no display or GPU.

### Added

- **Headless meta-data refresh** — `bigclust2 update /path/to/project` pulls
  current annotations from the project's declared sources and writes them into
  the project, no window required. `--auto-map` matches columns by name,
  `--dry-run` shows what would change. Also available as
  `bigclust2.update_project_meta()` from Python. See [the command line
  reference](cli.md#update-project) and [refreshing meta
  data](../how-to/update-meta-data.md#route-3-from-the-command-line).

## [0.2.0](https://github.com/flyconnectome/bigclust2/compare/v0.1.1...v0.2.0) — 2026-07-25

A large feature release centred on new interactive widgets, a from-scratch
project-building workflow, and this documentation site.

### Added

- **Project builder** (`File` → `Build Project`) — a guided dialog for turning raw data into a BigClust
  project without hand-writing the [`info` file](data-format.md). See the
  [how-to guides](../how-to/index.md).
- **Find Nearest widget** (`Selection` → `Find Nearest...`)— search for a neuron's nearest neighbours in feature
  space, and [grow the current selection](../concepts/selection.md) by nearest
  neighbour. See [Widgets](widgets.md).
- **Command palette** (press `Cmd+Shift+P`) to search and run any command from
  the keyboard. See also [Keyboard and mouse](shortcuts.md).
- **Custom metadata** — bring your own metadata sources and merge them into a
  loaded dataset. See [Refresh the meta data](../how-to/update-meta-data.md).
- **Credential handling** — centralized management of service credentials for the
  annotation backends (Clio, neuPrint, FlyTable). See [Credentials](../get-started/credentials.md).
- **Local skeleton sources in 3D viewer**
- **Update checker** — notifies you when a newer release is available.
- **Center on selection** — new menu item and keyboard shortcut (`Shift+C`).
- **Documentation site** — getting-started guides, how-to guides, concepts, and
  this reference section.

### Changed

- Smarter label placement, with better collision handling and positioning.
- Reworked and substantially expanded **connectivity widget**.
- Improved point visuals in the scatter view.
- Filtering (the [scoping system](../how-to/filter-a-dataset.md)) can now match
  empty and null values.

### Fixed

- The search widget no longer crashes on a trailing comma.

## [0.1.1](https://github.com/flyconnectome/bigclust2/compare/v0.1.0...v0.1.1) — 2026-07-17

### Added

- Debug menu: new **Traceback** option.

### Changed

- Raised the minimum `nglscenes` version.

## [0.1.0](https://github.com/flyconnectome/bigclust2/releases/tag/v0.1.0) — 2026-07-16

Initial public release on [PyPI](https://pypi.org/project/bigclust2/).

Core capabilities at first release: interactive exploration of clusterings of
high-dimensional connectomic data, embedding and feature comparison, the
connectivity widget, growing and shrinking selections, pushing annotations to
backends, working across multiple views, and Neuroglancer export.
