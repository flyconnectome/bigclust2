"""
Important: the default config options represent live datasets. Never use them for
testing or development, as you may accidentally write to the live dataset.
Reading is safe, but writing is not.
"""


from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields as dataclass_fields

try:
    from ...credentials import raise_for_auth_error
except ImportError:  # pragma: no cover - script execution fallback
    from bigclust2.credentials import raise_for_auth_error


@dataclass(frozen=True)
class FlyTableConfig:
    """Required configuration for FlyTable backend writes."""

    table_name: str = field(
        metadata={
            "label": "Table name",
            "placeholder": "e.g. neuron_annotations",
            "tooltip": "Target FlyTable table for writes.",
        }
    )
    base_name: str = field(
        metadata={
            "label": "Base name",
            "placeholder": "e.g. neuron_annotations",
            "tooltip": "Name of the base the table is in. Providing this can speed up finding the table.",
            "required": False,
        }
    )
    id_column: str = field(
        metadata={
            "label": "ID column",
            "placeholder": "e.g. root_id",
            "tooltip": "Column containing neuron IDs in the FlyTable table.",
        }
    )
    server: str = field(
        default="https://flytable.mrc-lmb.cam.ac.uk",
        metadata={
            "label": "Server",
            "placeholder": "Optional. Leave empty to use environment variable.",
            "tooltip": "If left empty, the backend falls back to the configured environment variable.",
            "required": False,
            "show": False,
        },
    )


@dataclass(frozen=True)
class FlyWireFlyTableConfig:
    """FlyTableConfig with defaults set for the FlyWire FlyTable."""

    user_initials: str = field(
        metadata={
            "label": "User initials",
            "placeholder": "e.g. 'PS'",
            "tooltip": "Will be used for e.g. `cell_type_source` when writing to `cell_type`.",
        }
    )
    table_name: str = field(
        default="info",
        metadata={
            "label": "Table name",
            "placeholder": "e.g. neuron_annotations",
            "tooltip": "Target FlyTable table for writes.",
            "show": False,
        },
    )
    base_name: str = field(
        default="main",
        metadata={
            "label": "Base name",
            "placeholder": "e.g. neuron_annotations",
            "tooltip": "Name of the base the table is in. Providing this can speed up finding the table.",
            "required": False,
            "show": False,
        },
    )
    id_column: str = field(
        default="root_783",
        metadata={
            "label": "ID column",
            "placeholder": "e.g. root_id",
            "tooltip": "Column containing neuron IDs in the FlyTable table.",
        },
    )
    server: str = field(
        default="https://flytable.mrc-lmb.cam.ac.uk",
        metadata={
            "label": "Server",
            "placeholder": "Optional. Leave empty to use environment variable.",
            "tooltip": "If left empty, the backend falls back to the configured environment variable.",
            "required": False,
            "show": False,
        },
    )


@dataclass(frozen=True)
class HemibrainFlyTableConfig:
    """FlyTableConfig with defaults set for the Hemibrain FlyTable."""

    user_initials: str = field(
        metadata={
            "label": "User initials",
            "placeholder": "e.g. 'PS'",
            "tooltip": "Will be used for e.g. `cell_type_source` when writing to `cell_type`.",
        }
    )
    table_name: str = field(
        default="hb_info",
        metadata={
            "label": "Table name",
            "placeholder": "e.g. neuron_annotations",
            "tooltip": "Target FlyTable table for writes.",
            "show": False,
        },
    )
    base_name: str = field(
        default="hemibrain",
        metadata={
            "label": "Base name",
            "placeholder": "e.g. neuron_annotations",
            "tooltip": "Name of the base the table is in. Providing this can speed up finding the table.",
            "required": False,
            "show": False,
        },
    )
    id_column: str = field(
        default="bodyId",
        metadata={
            "label": "ID column",
            "placeholder": "e.g. bodyID",
            "tooltip": "Column containing neuron IDs in the FlyTable table.",
        },
    )
    server: str = field(
        default="https://flytable.mrc-lmb.cam.ac.uk",
        metadata={
            "label": "Server",
            "placeholder": "Optional. Leave empty to use environment variable.",
            "tooltip": "If left empty, the backend falls back to the configured environment variable.",
            "required": False,
            "show": False,
        },
    )


@dataclass(frozen=True)
class NeuprintConfig:
    """Required configuration for Neuprint backend writes."""

    dataset: str = field(
        metadata={
            "label": "Dataset",
            "placeholder": "e.g. hemibrain:v1.2.1",
            "tooltip": "Neuprint dataset/collection name to update.",
        }
    )
    server: str = field(
        default="https://neuprint.janelia.org",
        metadata={
            "label": "Server",
            "placeholder": "Neuprint server URL",
        },
    )


@dataclass(frozen=True)
class ClioConfig:
    """Required configuration for Clio backend writes."""

    dataset_name: str = field(
        metadata={
            "label": "Dataset name",
            "placeholder": "e.g. CNS",
            "tooltip": "Clio dataset/collection name to update.",
        }
    )
    auto_fix_instances: bool = field(
        default=True,
        metadata={
            "label": "Auto-fix instances",
            "tooltip": "Automatically fix the `instance` when changing the type. This can lead to unintended consequences, so use with caution.",
            "required": False,
            "type": "bool",
        },
    )


@dataclass(frozen=True)
class CSVConfig:
    """Required configuration for CSV backend writes."""

    filepath: str = field(
        metadata={
            "label": "File path",
            "placeholder": "/path/to/annotations.csv",
            "tooltip": "CSV file to write annotations to.",
        }
    )


class AnnotationBackend(ABC):
    """Abstract base class for annotation backends."""

    BACKEND_NAME = ""
    CONFIG_CLASS = None
    FIELD_SUGGESTIONS = ()

    @classmethod
    def config_fields(cls):
        """Return config field names defined by the backend config dataclass."""
        if cls.CONFIG_CLASS is None:
            return ()
        return tuple(f.name for f in dataclass_fields(cls.CONFIG_CLASS))

    @classmethod
    def config_visible_fields(cls):
        """Return config field names that should be shown in the UI."""
        return tuple(
            field_name
            for field_name in cls.config_fields()
            if cls.config_field_meta(field_name).get("show", True)
        )

    @classmethod
    def config_field_meta(cls, key):
        """Return config field metadata for a specific field name."""
        if cls.CONFIG_CLASS is None:
            return {}
        for dataclass_field in dataclass_fields(cls.CONFIG_CLASS):
            if dataclass_field.name == key:
                return dict(dataclass_field.metadata)
        return {}

    @classmethod
    def config_field_default(cls, key):
        """Return default value for a config field, if one is declared."""
        if cls.CONFIG_CLASS is None:
            return None
        for dataclass_field in dataclass_fields(cls.CONFIG_CLASS):
            if dataclass_field.name != key:
                continue
            if dataclass_field.default is not MISSING:
                return dataclass_field.default
            if dataclass_field.default_factory is not MISSING:
                return dataclass_field.default_factory()
            return None
        return None

    def validate_field_value(self, field, value):
        """Validate that the given value is compatible with the specified field."""
        return True

    @abstractmethod
    def validate(self):
        """Validate the backend configuration."""
        pass

    @abstractmethod
    def validate_field(self, field):
        """Validate that the given field is compatible with the backend."""
        pass

    @abstractmethod
    def _write_annotations(self, ids, value, fields):
        """Write annotations to the backend."""
        pass

    def write_annotations(self, ids, value, fields):
        """Validate and write annotations to the backend."""
        try:
            self._write_annotations(ids, value, fields)
            return {"SUCCESS"}, None
        except Exception as exc:
            return {"ERROR"}, str(exc)

    def read_annotations(self, ids=None):
        """Read annotations from the backend.

        Args:
            ids (list[int], optional): List of neuron IDs to read. If None, read all.

        Returns:
            pd.DataFrame: DataFrame containing the annotations, with (unordered) neuron IDs as the index.
        """
        if not hasattr(self, "_read_annotations"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement reading annotations."
            )

        return self._read_annotations(ids)


class NeuprintBackend(AnnotationBackend):
    """Annotation backend for writing to a neuPrint database."""

    BACKEND_NAME = "neuPrint"
    CONFIG_CLASS = NeuprintConfig
    FIELD_SUGGESTIONS = tuple()  # neuPrint does not support writing annotations

    def __init__(self, config=None):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = self._get_neuprint_module().Client(
                    server=self.config.server, dataset=self.config.dataset
                )
            except Exception as exc:
                # Turns a missing/rejected token into a MissingCredentialsError
                # the GUI can prompt on; anything else propagates unchanged.
                raise_for_auth_error("neuprint", exc)
                raise
        return self._client

    def _get_neuprint_module(self):
        try:
            return importlib.import_module("neuprint")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "neuprint-python (github.com/connectome-neuprint/neuprint-python) is required for NeuprintBackend. "
                "Install it and set your token via Window > Credentials... or the "
                "NEUPRINT_APPLICATION_CREDENTIALS environment variable."
            ) from exc

    def validate(self):
        # Getting the client will validate credentials and dataset access
        _ = self.client

    def validate_field(self, field):
        # neuPrint does not support writing annotations, so no fields are valid
        return False

    def _read_annotations(self, ids=None, fields=None):
        neu = self._get_neuprint_module()
        if ids is not None:
            nc = neu.NeuronCriteria(bodyId=ids)
        else:
            nc = neu.NeuronCriteria()

        if fields is None:
            columns = "all"
        else:
            columns = list(fields)

        return neu.fetch_neurons(
            nc, client=self.client, omit_rois=True, returned_columns=columns
        ).set_index("bodyId")

    def _write_annotations(self, ids, value, fields):
        raise NotImplementedError("neuPrint does not support writing annotations.")


class ClioBackend(AnnotationBackend):
    """Annotation backend for writing to Clio."""

    BACKEND_NAME = "Clio"
    CONFIG_CLASS = ClioConfig
    FIELD_SUGGESTIONS = (
        "type",
        "flywire_type",
        "hemibrain_type",
        "superclass",
        "soma_side",
    )

    disallowed_fields = {
        "bodyid",
    }

    def __init__(self, config: ClioConfig, debug=False):
        self.config = config
        self._client = None
        self.debug = debug

    @property
    def client(self):
        if self._client is None:
            # No explicit token: clio resolves its own token file first and
            # falls back to gcloud, so existing set-ups keep working.
            try:
                self._client = self._get_clio_module().Client(
                    dataset=self.config.dataset_name
                )
            except Exception as exc:
                raise_for_auth_error("clio", exc)
                raise
        return self._client

    def _get_clio_module(self):
        try:
            return importlib.import_module("clio")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "clio-py (github.com/schlegelp/clio-py) is required for ClioBackend. "
                "Install it and set your token via Window > Credentials... "
                "(or configure `gcloud` authentication)."
            ) from exc

    @property
    def available_fields(self):
        return list(self.client.meta["bodyAnnotationSchema"]["collection"].keys())

    def validate(self):
        # Getting the client will validate credentials and dataset access
        _ = self.client

    def validate_field(self, field):
        return field in self.available_fields and field not in self.disallowed_fields

    def _read_annotations(self, ids=None):
        clio = self._get_clio_module()
        return clio.fetch_annotations(bodyid=ids, client=self.client).set_index(
            "bodyid"
        )

    def _write_annotations(self, ids, value, fields):
        ids = np.array(ids).astype(np.int64)

        if self.debug:
            return

        clio = self._get_clio_module()

        to_push = {field: value for field in fields}

        # If we're clearing the type, we should also clear the instance (if it exists)
        if (
            "type" in fields
            and "instance" not in fields
            and value is None
            and self.config.auto_fix_instances
        ):
            to_push["instance"] = None

        clio.set_fields(ids, **to_push, progress=False, client=self.client)

        if (
            "type" in fields
            and "instance" not in fields
            and value is not None
            and self.config.auto_fix_instances
        ):
            # Instance is a combination of "{type}_{soma_side << root_side}"
            # We need to first get the soma/root side to construct the instance value
            ann = clio.fetch_annotations(ids)

            for col in ("soma_side", "root_side"):
                if col not in ann.columns:
                    ann[col] = np.nan

            ann["instance"] = value
            ann["side"] = ann.soma_side.fillna(ann.root_side).astype("string")
            ann.loc[ann.side.notna(), "instance"] = (
                value + "_" + ann.loc[ann.side.notna(), "side"]
            )
            clio.set_annotations(
                ann[["bodyid", "instance"]], progress=False, client=self.client
            )


class FlyTableBackend(AnnotationBackend):
    """Annotation backend for writing to a FlyTable."""

    BACKEND_NAME = "FlyTable"
    CONFIG_CLASS = FlyTableConfig
    FIELD_SUGGESTIONS = (
        "cell_type",
        "hemilineage",
        "super_class",
        "side",
    )

    disallowed_fields = {"root_id", "root_783"}

    def __init__(self, config: FlyTableConfig, debug=False):
        self.config = config
        self.debug = debug
        self._table = None

    @property
    def table(self):
        """Lazily load the FlyTable module and return the target table."""
        if self._table is None:
            ss = self._get_seaserpent_module()
            try:
                self._table = ss.Table(
                    self.config.table_name,
                    base=self.config.base_name,
                    server=self.config.server,
                    read_only=False,
                )
            except Exception as exc:
                raise_for_auth_error("flytable", exc)
                raise
        return self._table

    @property
    def data_columns(self):
        """Return columns in the that are actually useful."""
        return [
            c
            for c in self.table.columns
            if ("_user" not in c)
            and not c.endswith("_source")
            and not c.startswith("_")
        ]

    def _get_seaserpent_module(self):
        try:
            return importlib.import_module("seaserpent")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "seaserpent (github.com/schlegelp/seaserpent) is required for FlyTableBackend. "
                "Install it and set your token via Window > Credentials... or the "
                "SEATABLE_TOKEN environment variable."
            ) from exc

    def validate(self):
        # Accessing the table will validate credentials and table existence
        _ = self.table
        if self.config.id_column not in self.table.columns:
            raise ValueError(
                f"ID column '{self.config.id_column}' does not exist in {self.config.table_name} table."
            )

    def validate_field(self, field):
        return (
            field in self.table.columns
            and field != self.config.id_column
            and field not in self.disallowed_fields
        )

    def _read_annotations(self, ids=None, fields=None):
        """Returns a DataFrame with neuron IDs (unordered) as the index and the requested fields as columns."""
        if fields is None:
            fields = self.data_columns

        if ids is not None:
            ids = np.array(ids).astype(str)
            data = self.table.loc[
                self.table[self.config.id_column].isin(ids), fields
            ].set_index(self.config.id_column)
        else:
            data = self.table[fields].set_index(self.config.id_column)
        data[self.config.id_column] = data.index.astype(
            int
        )  # int64 is stored as str in FlyTable
        return data

    def _write_annotations(self, ids, value, fields):
        ids = np.array(ids).astype(np.int64)
        table_roots = self.table[self.config.id_column].values.astype(np.int64)

        miss = ~np.isin(ids, table_roots)
        if miss.any():
            raise ValueError(
                f"{miss.sum()} of the given root IDs do not exists in {self.config.table_name} table."
            )

        to_update = np.isin(table_roots, ids)

        if self.debug:
            return

        with self._get_seaserpent_module().base.BundleEdits(self.table):
            for f in fields:
                self.table.loc[to_update, f] = value


class FlyWireFlyTableBackend(FlyTableBackend):
    """FlyTableBackend with defaults set for the FlyWire FlyTable."""

    BACKEND_NAME = "FlyWire @ FlyTable"
    CONFIG_CLASS = FlyWireFlyTableConfig
    FIELD_SUGGESTIONS = (
        "cell_type",
        "ito_lee_hemilineage",
        "super_class",
        "side",
    )
    USER_FIELDS = ("cell_type", "hemibrain_type", "malecns_type")
    VALUE_RESTRICTIONS = {
        "side": {"left", "right", "center", None},
        "dimorphism": {
            "female-specific",
            "potentially female-specific",
            "sexually dimorphic",
            "potentially sexually dimorphic",
            None,
        },
    }

    def __init__(self, config: FlyWireFlyTableConfig, debug=False):
        super().__init__(config=config, debug=debug)

    def validate_field_value(self, field, value):
        if field in self.VALUE_RESTRICTIONS:
            return value in self.VALUE_RESTRICTIONS[field]
        return True

    def _write_annotations(self, ids, value, fields):
        # Write the actual fields
        super()._write_annotations(ids, value, fields)

        # Now check if we need to also update the user fields
        user_fields = []
        for f in fields:
            if f in self.USER_FIELDS:
                user_field = f"{f}_source"
                if user_field not in fields:
                    user_fields.append(user_field)

        if user_fields:
            super()._write_annotations(ids, self.config.user_initials, user_fields)


class HemibrainFlyTableBackend(FlyTableBackend):
    """HemibrainFlyTableBackend with defaults set for the Hemibrain FlyTable."""

    BACKEND_NAME = "Hemibrain @ FlyTable"
    CONFIG_CLASS = HemibrainFlyTableConfig
    FIELD_SUGGESTIONS = (
        "type_corrected",
        "instance_corrected",
        "type",
        "ito_lee_hemilineage",
        "super_class",
        "side",
    )
    USER_FIELDS = ("type_corrected",)
    VALUE_RESTRICTIONS = {
        "side": {"left", "right", "center", None},
    }

    def __init__(self, config: HemibrainFlyTableConfig, debug=False):
        super().__init__(config=config, debug=debug)

    def validate_field_value(self, field, value):
        if field in self.VALUE_RESTRICTIONS:
            return value in self.VALUE_RESTRICTIONS[field]
        return True

    def _write_annotations(self, ids, value, fields):
        # Write the actual fields
        super()._write_annotations(ids, value, fields)

        # Now check if we need to also update the user fields
        user_fields = []
        for f in fields:
            if f in self.USER_FIELDS:
                user_field = f"{f}_source"
                if user_field not in fields:
                    user_fields.append(user_field)

        if user_fields:
            super()._write_annotations(ids, self.config.user_initials, user_fields)


class CSVBackend(AnnotationBackend):
    """Annotation backend for writing to a CSV file."""

    BACKEND_NAME = "CSV"
    CONFIG_CLASS = CSVConfig
    FIELD_SUGGESTIONS = ("cell_type", "type", "comment")

    def __init__(self, config: CSVConfig):
        self.config = config
        self.path = Path(config.filepath).expanduser()

    def validate(self):
        # For CSV, we can always write (it will create the file if it doesn't exist)
        pass

    def validate_field(self, field):
        # For CSV, we can accept any field name
        return field != "id"  # 'id' is reserved for the neuron ID column

    @property
    def dataframe(self):
        if self.path.exists():
            return pd.read_csv(self.path, dtype={"id": np.int64})
        else:
            return pd.DataFrame(columns=["id"], dtype=np.int64)

    def _read_annotations(self, ids=None, fields=None):
        df = self.dataframe
        if fields is None:
            fields = df.columns

        if ids is not None:
            ids = np.array(ids).astype(np.int64)
            return df.loc[df.id.isin(ids), fields].set_index("id")
        else:
            return df[fields].set_index("id")

    def _write_annotations(self, ids, value, fields):
        ids = np.array(ids).astype(np.int64)
        df = self.dataframe

        # See if we need to add any IDs that aren't already in the dataframe
        existing_ids = df.id.unique()
        new_ids = np.array(list(set(ids) - set(existing_ids)), dtype=np.int64)
        if len(new_ids):
            # Build a typed int64 ID column without using DataFrame(dtype={...}),
            # which raises on newer pandas versions.
            df = pd.concat(
                [df, pd.DataFrame({"id": pd.Series(new_ids, dtype=np.int64)})],
                ignore_index=True,
            )

        # Update/add the new values for the specified fields and IDs
        for f in fields:
            if f not in df.columns:
                df[f] = value
            else:
                df.loc[df.id.isin(ids), f] = value

        df.to_csv(self.path, index=False)


BACKEND_CLASSES = (
    ClioBackend,
    NeuprintBackend,
    FlyTableBackend,
    FlyWireFlyTableBackend,
    HemibrainFlyTableBackend,
    CSVBackend,
)

BACKEND_REGISTRY = {backend.BACKEND_NAME: backend for backend in BACKEND_CLASSES}


def _coerce_bool(value):
    """Coerce UI/default/saved values to a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def _field_is_bool(backend_cls, key):
    """Whether a config field is declared as a boolean."""
    value_type = backend_cls.config_field_meta(key).get("type", "string")
    return value_type in {bool, "bool", "boolean"}


def _field_is_required(backend_cls, key):
    """Whether a config field must be filled (defaults to True)."""
    return bool(backend_cls.config_field_meta(key).get("required", True))


def _coerce_config(backend_cls, config_dict):
    """Build a normalized kwargs dict for ``backend_cls.CONFIG_CLASS``.

    Values come from `config_dict` where provided, otherwise from the field's
    declared default. Booleans are coerced; everything else is stringified and
    stripped. Required, non-bool fields that resolve to empty raise ``ValueError``
    so callers get a clear message instead of an opaque dataclass error.
    """
    if backend_cls.CONFIG_CLASS is None:
        return {}

    config_dict = config_dict or {}
    values = {}
    for key in backend_cls.config_fields():
        is_bool = _field_is_bool(backend_cls, key)
        if key in config_dict:
            raw = config_dict[key]
        else:
            raw = backend_cls.config_field_default(key)

        if raw is None:
            values[key] = False if is_bool else ""
        elif is_bool:
            values[key] = _coerce_bool(raw)
        else:
            values[key] = str(raw).strip()

    for key, value in values.items():
        if (
            _field_is_required(backend_cls, key)
            and not _field_is_bool(backend_cls, key)
            and not str(value).strip()
        ):
            raise ValueError(
                f"Missing required config field '{key}' for backend "
                f"'{backend_cls.BACKEND_NAME}'."
            )

    return values


def build_backend(name, config_dict, *, debug=False):
    """Instantiate an annotation backend from its registry name and a config dict.

    Shared by the data layer and the configuration dialogs so backend
    construction (default-filling, type coercion, required-field validation)
    lives in one place.

    Parameters
    ----------
    name : str
        A key in ``BACKEND_REGISTRY`` (e.g. "neuPrint", "CSV").
    config_dict : dict
        Flat mapping of ``CONFIG_CLASS`` field names to values. Missing optional
        fields fall back to their declared defaults.
    debug : bool
        Passed to backends that accept it (writes become no-ops); ignored by
        backends whose constructor doesn't take ``debug``.

    Returns
    -------
    AnnotationBackend
    """
    backend_cls = BACKEND_REGISTRY.get(name)
    if backend_cls is None:
        raise ValueError(f"Unknown backend: {name!r}")

    config = backend_cls.CONFIG_CLASS(**_coerce_config(backend_cls, config_dict))

    try:
        return backend_cls(config=config, debug=debug)
    except TypeError:
        # NeuprintBackend / CSVBackend constructors don't accept `debug`.
        return backend_cls(config=config)
