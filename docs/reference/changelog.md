# Changelog

Notable changes to BigClust, newest first. Versions follow
[semantic versioning](https://semver.org/); the format is loosely based on
[Keep a Changelog](https://keepachangelog.com/).

For the complete commit-level history, see the
[releases on GitHub](https://github.com/flyconnectome/bigclust2/releases).

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
