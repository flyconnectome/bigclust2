"""Project-level shared state for a set of views.

A :class:`Project` is the hub that ties together the multiple *views*
(``MainWidget`` instances) the user opens into one dataset: the main full view,
any selection opened in a new view, and views detached into their own windows.

Today its sole job is to hold *per-neuron visual state* (e.g. which neurons have
had annotations set) and broadcast changes so every view of the project repaints
and newly opened views can pull the current state. The store is deliberately
generic — keyed by ``state_type`` and by the cross-view neuron identity tuple
``(neuron_id, dataset)`` — so other propagated states can be added later without
touching this class.

The ``Project`` knows nothing about views: views connect to its
``neuron_state_changed`` signal themselves (Qt auto-disconnects them on
destruction) and pull the full state via :meth:`neuron_state` when populated.
Isolation between unrelated projects is automatic because each ``Project`` is a
distinct ``QObject`` with its own signal.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class Project(QObject):
    """Shared per-neuron state for all views of one project.

    Parameters of ``neuron_state_changed``: the ``state_type`` (str) and a fresh
    ``changes`` mapping ``{(neuron_id, dataset): value}`` carrying only the
    entries that just changed.
    """

    neuron_state_changed = Signal(str, object)

    def __init__(self):
        super().__init__()
        # state_type -> {(neuron_id, dataset): value}
        self._neuron_states: dict[str, dict[tuple, object]] = {}

    def state_types(self):
        """The state types that currently hold any entries."""
        return list(self._neuron_states)

    def neuron_state(self, state_type):
        """A copy of the full ``{(neuron_id, dataset): value}`` mapping.

        Returns a fresh dict so callers (e.g. a view applying carried-over state)
        cannot mutate the internal store.
        """
        return dict(self._neuron_states.get(state_type, {}))

    def set_neuron_state(self, state_type, keys, value):
        """Record ``value`` for each neuron ``key`` and broadcast the change.

        Parameters
        ----------
        state_type : str
            The state being set, e.g. ``"annotated"``.
        keys : iterable of (neuron_id, dataset)
            Neurons to update. Identity is the cross-view tuple, not a point
            index, so the state survives subsetting/reordering between views.
        value : object
            Value to store (e.g. ``True``). The *visual* meaning of the value is
            owned by each view's handler, not by the project.
        """
        store = self._neuron_states.setdefault(state_type, {})
        changes = {}
        for key in keys:
            store[key] = value
            changes[key] = value
        if changes:
            self.neuron_state_changed.emit(state_type, changes)
