import logging
import argparse
import sys
from importlib.metadata import PackageNotFoundError, version


def _installed_version() -> str:
    try:
        return version("bigclust2")
    except PackageNotFoundError:
        return "unknown"


def _add_update_command(subparsers) -> None:
    """Register `bigclust2 update` -- a headless meta-data refresh."""
    parser = subparsers.add_parser(
        "update",
        help="Refresh a project's meta data from its declared sources",
        description=(
            "Pull current annotations for every dataset the project declares a "
            "meta source for, merge them into the project's meta table and save "
            "it. The sources are only read from, never written to."
        ),
    )
    parser.add_argument(
        "project",
        metavar="PROJECT",
        help="Path to a local project directory (the one holding the 'info' file)",
    )
    parser.add_argument(
        "--auto-map",
        dest="auto_map_columns",
        action="store_true",
        help=(
            "Match project columns to source columns by name for any column the "
            "project does not already map. Existing mappings always win."
        ),
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        metavar="NAME",
        action="append",
        help=(
            "Only refresh this dataset; repeat for several. Other datasets are "
            "left untouched."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull and merge, but write nothing. Shows what a real run would do.",
    )
    parser.add_argument(
        "--no-save-sources",
        dest="save_sources",
        action="store_false",
        help="Don't store auto-mapped columns back into the project's info file",
    )
    # SUPPRESS so that `bigclust2 --debug update …` is not overridden by this
    # subparser's own default.
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show debug information",
    )
    parser.set_defaults(command="update")


#: How many changed columns to name per dataset before summarizing the rest.
MAX_COLUMNS_SHOWN = 6


def _credentials_hint() -> str:
    """Where to put a token, built from the credential registry."""
    from bigclust2.credentials import SERVICE_SPECS

    env_vars = [var for spec in SERVICE_SPECS.values() for var in spec.env_vars]
    return (
        f"Tip: export the backend's token ({', '.join(env_vars)}), or store it "
        "once via Window → Credentials in the app."
    )


def _run_update(args) -> int:
    """Run the update command. Returns the process exit code."""
    from bigclust2 import update_project_meta

    # Behaving like the app - reusing the tokens it stored - is an entry-point
    # policy, so it happens here rather than inside the library call.
    try:
        from bigclust2.gui.widgets.credentials import apply_saved_credentials

        apply_saved_credentials()
    except Exception as exc:  # noqa: BLE001 - env vars may still carry a token
        logging.getLogger(__name__).debug("No stored credentials applied: %s", exc)

    def progress(done, total, dataset):
        if dataset:
            print(f"  [{done + 1}/{total}] {dataset}…", flush=True)

    try:
        result = update_project_meta(
            args.project,
            auto_map_columns=args.auto_map_columns,
            datasets=args.datasets,
            dry_run=args.dry_run,
            save_sources=args.save_sources,
            progress_callback=progress,
        )
    except Exception as exc:
        if args.debug:
            raise
        print(f"Update failed: {exc}", file=sys.stderr)
        return 1

    for dataset in result.report.datasets:
        if dataset.error:
            print(f"  {dataset.dataset}: failed — {dataset.error}")
            continue
        detail = (
            f"  {dataset.dataset}: {dataset.cells_changed:,} values changed "
            f"in {dataset.rows_changed:,} rows"
        )
        columns = dataset.columns_touched()
        if columns:
            # A wide project can map dozens of columns; the full list belongs in
            # the meta explorer, not in a line of terminal output.
            shown = ", ".join(columns[:MAX_COLUMNS_SHOWN])
            if len(columns) > MAX_COLUMNS_SHOWN:
                shown += f" and {len(columns) - MAX_COLUMNS_SHOWN} more"
            detail += f" ({shown})"
        if dataset.ids_missing:
            detail += f"; {dataset.ids_missing:,} ids not in the source"
        print(detail)

    print(result.summary())

    if result.dry_run:
        print("Re-run without --dry-run to save these values.")

    errors = result.report.errors
    if any(
        "credential" in str(d.error).lower() or "token" in str(d.error).lower()
        for d in errors
    ):
        print(_credentials_hint(), file=sys.stderr)

    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bigclust2",
        description="A Python package for inspecting and visualising large clusterings",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    parser.add_argument(
        "--from",
        dest="dataset",
        metavar="DATASET",
        default=None,
        help="Path or URL of a dataset to load on startup",
    )
    parser.add_argument(
        "--filters",
        dest="filters",
        metavar="EXPR",
        default=None,
        help=(
            "Filter expression applied to the dataset on startup "
            "(e.g. 'dataset == \"hemibrain\"'). Requires --from."
        ),
    )
    parser.add_argument(
        "--embedding",
        dest="embedding",
        metavar="MODE",
        default=None,
        help=(
            "Load-time embedding mode, e.g. 'calculate from distances' or "
            "'calculate from features'. Requires --from."
        ),
    )

    # Without a command we launch the GUI, so this stays optional.
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_update_command(subparsers)

    args = parser.parse_args()

    if args.version:
        print(f"{parser.prog} {_installed_version()}")
        return

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("bigclust2").setLevel(logging.DEBUG)

    if args.command == "update":
        raise SystemExit(_run_update(args))

    # Via the package, not `bigclust2.gui`, so the "Loading bigclust2…" banner
    # covers the (slow) GUI import.
    from bigclust2 import main as run_gui

    run_gui(dataset=args.dataset, filters=args.filters, embedding=args.embedding)
