import cmap
import json
import logging
import requests

import numpy as np
import pandas as pd
import polars as pl

from abc import ABC
from pathlib import Path

from .utils import is_url, string_to_polars_filter, Url


logger = logging.getLogger(__name__)


def parse_directory(path):
    """Parse a BigClust dataset directory.

    Parameters
    ----------
    path : str | Path
        Path to the dataset directory (local or remote).

    Returns
    -------
    ParseDirectory
        Parsed directory object.

    """
    if not is_url(path):
        path = Path(path)

        # Convert relative paths to absolute paths
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not (path / "info").exists():
            raise FileNotFoundError(f"No 'info' file found in directory: {path}")

        with open(path / "info", "r") as f:
            info = json.load(f)
    else:
        # Note to self: we could make this compatible with GS:// and S3:// URLs later
        path = Url(path)
        response = requests.get(path / "info")
        response.raise_for_status()
        info = response.json()

    if isinstance(info, list):
        return MultiProjectLoader(path, info)
    elif isinstance(info, dict):
        return SingleProjectLoader(
            name=info.get("name", "unnamed"), path=path, info=info
        )
    else:
        raise ValueError("Invalid info format:", type(info))


class BaseProjectLoader(ABC):
    """Abstract base class for BigClust dataset loaders."""

    @property
    def path(self):
        """Get the path to the dataset."""
        return self._path

    @path.setter
    def path(self, value):
        """Set the path to the dataset."""
        if isinstance(value, Url):
            self._path = value
        elif is_url(value):
            self._path = Url(value)
        else:
            self._path = Path(value)

    @property
    def is_remote(self):
        """Check if the dataset is remote (URL) or local."""
        if isinstance(self._path, Url):
            return True
        return is_url(self._path)

    def load_file(self, file, relative=True):
        """Load a file from the dataset.

        Parameters
        ----------
        file : str
            Name of the file to load. Supported formats:
            - JSON files (including "info")
            - Feather files
            - CSV files
            - Parquet files
        filter_expr : str, optional
            Filter expression to apply when loading the file (if applicable).

        Returns
        -------
        Any
            Loaded file content.

        """
        if isinstance(file, str):
            if self.is_remote:
                file = Url(file)
            else:
                file = Path(file)

        if relative:
            filepath = self.path / file
        else:
            filepath = file

        if file.suffix == ".json" or file.name == "info":
            if self.is_remote:
                response = requests.get(filepath)
                response.raise_for_status()
                return response.json()
            else:
                if not filepath.exists():
                    raise FileNotFoundError(f"File not found: {filepath}")
                with open(filepath, "r") as f:
                    return json.load(f)
        elif file.suffix == ".feather":
            return pd.read_feather(str(filepath))  # need to convert Url to str
        elif file.suffix == ".csv":
            return pd.read_csv(str(filepath))
        elif file.suffix == ".parquet":
            return pd.read_parquet(str(filepath))
        else:
            raise ValueError(f"Unsupported file type: {file}")

    def load_file_lazy(self, file):
        """Load a file from the dataset.

        Parameters
        ----------
        file : str
            Name of the file to load. Supported formats:
            - Feather files
            - CSV files
            - Parquet files

        Returns
        -------
        Any
            Polars LazyFrame of the loaded file content.

        """
        file = str(file)
        if file.endswith(".feather"):
            try:
                lazy = pl.scan_ipc(f"{self.path}/{file}")
                # Some malformed IPC buffers only fail on materialization, not scan creation.
                lazy.limit(1).collect()
                return lazy
            except Exception as e:
                # Some Feather/IPC variants fail in lazy scan but are readable via pandas/pyarrow.
                logger.warning(
                    "Falling back to eager Feather load for '%s' after scan_ipc failed: %s",
                    file,
                    e,
                )
                return pl.from_pandas(self.load_file(file)).lazy()
        elif file.endswith(".csv"):
            return pl.scan_csv(f"{self.path}/{file}")
        elif file.endswith(".parquet"):
            return pl.scan_parquet(f"{self.path}/{file}")
        else:
            raise ValueError(f"Unsupported file type for lazy-loading: {file}")


class MultiProjectLoader(BaseProjectLoader):
    """Loader for multiple BigClust datasets.

    Parameters
    ----------
    path : str | Path | Url
        Path to the directory with multiple BigClust datasets (local or remote).
    info : list
        List of project info dictionaries.

    """

    def __init__(self, path, info):
        self.path = path
        self.info = info

    @property
    def info(self):
        """List of project info dictionaries."""
        return self._info

    @info.setter
    def info(self, value):
        """Set the list of project info dictionaries."""
        # Sanity check the info file format
        if not isinstance(value, list):
            raise ValueError("Info must be a list of project info dictionaries.")

        for project in value:
            if not isinstance(project, str):
                raise ValueError("Each project info must be a dictionary.")

        self._info = value

    @property
    def projects(self):
        """Dictionary of SingleProjectLoader objects for each dataset."""
        if not hasattr(self, "_projects"):
            self.load_project_info()
        return self._projects

    def __repr__(self):
        return f"<MultiProjectLoader with {len(self.info)} projects at {self.path}>"

    def __len__(self):
        return len(self.projects)

    def __getitem__(self, index):
        """Get a SingleProjectLoader for the dataset at the given index.

        Parameters
        ----------
        index : int | str
            Index or name of the dataset.

        Returns
        -------
        SingleProjectLoader
            Loader for the specified dataset.

        """
        if isinstance(index, str):
            for i, project in enumerate(self.projects):
                if project["name"] == index:
                    index = i
                    break
            else:
                raise KeyError(f"Project with name '{index}' not found.")

        return self.projects[index]

    def load_project_info(self):
        """Load or update the project info list."""
        self._projects = []

        for project in self.info:
            try:
                project_path = self.path / project
                info = self.load_file(project_path / "info")
                loader = SingleProjectLoader(info["name"], project_path, info=info)
                self._projects.append(loader)
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    print(
                        f"Warning: Skipping project '{project}' - info file not found."
                    )
                    self._projects.append(None)
                else:
                    raise e


class SingleProjectLoader(BaseProjectLoader):
    """Loader for a single BigClust dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    path : str | Path
        Path to the dataset (local or remote).
    info : dict, optional
        Info JSON as a dictionary.
    filter_expr : str, optional
        Filter expression to apply to the metadata.
        Note that the filter query is only applied
        when calling the `.compile()` method.

    """

    REQ_VALUES = ["meta"]
    REQ_META_COLS = ["id", "dataset", "label"]

    def __init__(self, name, path, info=None, filter_expr=None):
        self.name = name
        self.path = path
        self.filter_expr = filter_expr

        if info is not None:
            self.info = info

    @property
    def info(self):
        """Info JSON as a dictionary."""
        if not hasattr(self, "_info"):
            self.load_info()
        return self._info

    @info.setter
    def info(self, value):
        """Set the info JSON dictionary."""
        # Sanity check the info file format
        for req in self.REQ_VALUES:
            if req not in value:
                raise ValueError(f"Missing required key '{req}' in info: {value}")
        self._info = value

    def __repr__(self):
        return f"<SingleProjectLoader for '{self.name}' at {self.path}>"

    @property
    def meta(self):
        """Metadata as a Pandas DataFrame."""
        if not hasattr(self, "_meta"):
            self.load_meta()
        return self._meta

    @property
    def meta_lazy(self):
        """Metadata as a Polars LazyFrame."""
        if not hasattr(self, "_meta_lazy"):
            self._meta_lazy = self.load_file_lazy(self._get_file_spec("meta"))
        return self._meta_lazy

    @property
    def meta_lazy_schema(self):
        """Meta data schema."""
        if not hasattr(self, "_meta_lazy_schema"):
            self._meta_lazy_schema = self.meta_lazy.collect_schema()
        return self._meta_lazy_schema

    @property
    def meta_columns(self):
        """List of metadata columns."""
        if hasattr(self, "_meta"):
            return list(self._meta.columns)
        else:
            return self.meta_lazy_schema.names()

    @property
    def distances(self):
        """Pairwise distances as a Pandas DataFrame."""
        if not hasattr(self, "_distances"):
            self._distances = self.load_distances()
        return self._distances

    @property
    def has_distances(self):
        """Check if project distances are loaded."""
        return "distances" in self.info and self.info["distances"] is not None

    @property
    def embeddings(self):
        """Low-dimensional embeddings as a Pandas DataFrame."""
        if not hasattr(self, "_embeddings"):
            self._embeddings = self.load_embeddings()
        return self._embeddings

    @property
    def has_embeddings(self):
        """Check if project embeddings are loaded."""
        return "embeddings" in self.info and self.info["embeddings"] is not None

    @property
    def features(self):
        """Low-dimensional embeddings as a Pandas DataFrame."""
        if not hasattr(self, "_features"):
            self._features = self.load_features()
        return self._features

    @property
    def has_features(self):
        """Check if project features are loaded."""
        return "features" in self.info and self.info["features"] is not None

    @property
    def feature_type(self):
        """Low-dimensional embeddings as a Pandas DataFrame."""
        if not self.has_features:
            return None
        return self.info["features"].get("type", None)

    def load_info(self):
        """Load or update the info JSON file."""
        self._info = self.load_file("info")

    def _get_file_spec(self, key):
        """Return the file path configured for a dataset component.

        The info JSON supports either:
          - a plain file path string, or
          - a dictionary with a "file" key.
        """
        value = self.info.get(key)
        return self._spec_to_file(value, key=key)

    @staticmethod
    def _spec_to_file(value, key=None):
        """Resolve a file spec (plain string or ``{"file": ...}`` dict) to a path.

        Returns None if `value` is None.
        """
        if value is None:
            return None
        if isinstance(value, dict):
            file = value.get("file")
            if file is None:
                raise ValueError(
                    f"Missing 'file' entry for '{key or 'spec'}' in info: {value}"
                )
            return file
        return value

    @property
    def normalized_embedding_specs(self):
        """List of file-level embedding specs (no data is loaded).

        Each spec is a dict with keys ``name``, ``emb_columns`` (or None),
        ``emb_file`` (or None), ``feat_spec`` and ``dist_spec`` (the raw
        features/distances sub-spec or None). Supports both the legacy
        single-embedding format and the multi-embedding list format.
        """
        if not hasattr(self, "_embedding_specs"):
            self._embedding_specs = self._normalize_embedding_specs()
        return self._embedding_specs

    def _normalize_embedding_specs(self):
        """Build the canonical list of embedding specs from `self.info`."""
        info = self.info
        emb = info.get("embeddings", None)
        top_feat = info.get("features", None)
        top_dist = info.get("distances", None)

        # Whether the project declares features/distances but no embeddings.
        self._orphan_sources = (emb is None) and (
            top_feat is not None or top_dist is not None
        )

        if emb is None:
            return []

        if isinstance(emb, list):
            # New multi-embedding format: one entry per list item.
            specs = []
            for i, entry in enumerate(emb):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Each embedding entry must be a dict, got {type(entry)}."
                    )
                specs.append(self._make_embedding_spec(entry, i, top_feat, top_dist))
            return specs

        # Legacy single-embedding format (dict or plain path string).
        entry = emb if isinstance(emb, dict) else {"file": emb}
        return [self._make_embedding_spec(entry, 0, top_feat, top_dist)]

    @staticmethod
    def _make_embedding_spec(entry, i, top_feat, top_dist):
        """Turn one embedding entry dict into a normalized spec."""
        emb_columns = entry.get("columns", None)
        emb_file = None if emb_columns is not None else SingleProjectLoader._spec_to_file(
            entry, key="embeddings"
        )
        return {
            "name": str(entry.get("name", f"embedding {i + 1}")),
            "emb_columns": emb_columns,
            "emb_file": emb_file,
            # Per-entry features/distances override the top-level ones.
            "feat_spec": entry.get("features", top_feat),
            "dist_spec": entry.get("distances", top_dist),
        }

    def load_meta(self):
        """Load or update the meta data."""
        file = self._get_file_spec("meta")

        meta = self.load_file(file)
        self._meta = meta

    def load_distances(self):
        """Load or update the pairwise distances."""
        if "distances" in self.info and self.info["distances"] is not None:
            file = self._get_file_spec("distances")
            return self.load_file(file)
        else:
            return None

    def load_features(self):
        """Load or update the feature vectors."""
        if "features" in self.info and self.info["features"] is not None:
            return self.load_file(self._get_file_spec("features"))
        else:
            return None

    def load_embeddings(self):
        """Load or update the low-dimensional embeddings."""
        if "embeddings" in self.info and self.info["embeddings"] is not None:
            # Embeddings can be a separate file or columns in the meta file
            if (
                isinstance(self.info["embeddings"], dict)
                and "columns" in self.info["embeddings"]
            ):
                cols = self.info["embeddings"]["columns"]
                emb = self.meta[cols]
            else:
                emb = self.load_file(self._get_file_spec("embeddings"))

            if isinstance(emb, pd.DataFrame):
                emb = emb.values
            elif not isinstance(emb, np.ndarray):
                raise ValueError(
                    f"Embeddings must be a Pandas DataFrame or NumPy array, got {type(emb)}"
                )

            if emb.shape[1] != 2:
                raise ValueError(
                    f"Embeddings must have exactly 2 dimensions, got {emb.shape[1]}"
                )
            return emb
        else:
            return None

    def _load_embedding_array(self, *, columns, file, filtered_meta, ind, cache):
        """Load a single embedding as an (N, 2) array (filtered if `ind` given)."""
        key = ("emb_cols", tuple(columns)) if columns is not None else ("emb_file", str(file))
        if cache is not None and key in cache:
            return cache[key]

        if columns is not None:
            # `filtered_meta` equals the full meta when no filter is applied.
            emb = filtered_meta[columns].values
        elif ind is None:
            emb = self.load_file(file)
            if isinstance(emb, pd.DataFrame):
                emb = emb.values
        else:
            emb_lazy = self.load_file_lazy(file)
            emb = (
                emb_lazy.with_row_index(name="__index__")
                .filter(pl.col("__index__").is_in(ind))
                .collect()
                .drop("__index__")
                .to_numpy()
            )

        emb = np.asarray(emb)
        if emb.ndim != 2 or emb.shape[1] != 2:
            raise ValueError(
                f"Embeddings must have exactly 2 dimensions, got shape {emb.shape}."
            )

        if cache is not None:
            cache[key] = emb
        return emb

    @staticmethod
    def _detect_id_column(cols, what):
        """Find the id/index column among lazy-frame column names."""
        for candidate in ("id", "index", "__index_level_0__"):
            if candidate in cols:
                return candidate
        raise ValueError(
            f"{what} file must have either an 'id', 'index' or '__index_level_0__' "
            "column to filter on."
        )

    def _load_distances_df(self, *, file, ind, cache):
        """Load a single (square) distance matrix (filtered if `ind` given)."""
        key = ("dist", str(file))
        if cache is not None and key in cache:
            return cache[key]

        if ind is None:
            # No filter: read as-is (file is expected to carry its id index).
            dists = self.load_file(file)
        else:
            dist_lazy = self.load_file_lazy(file)

            # If these are pairwise distances, we need to filter on both rows and columns
            cols = dist_lazy.collect_schema().names()
            id_col = self._detect_id_column(cols, "Distance")

            col_mask = np.zeros(len(cols), dtype=bool)
            if cols.index(id_col) == 0:  # index/id column is the first column
                col_mask[0] = True
                col_mask[ind + 1] = True  # +1 because of the id/index column
            elif cols.index(id_col) == len(cols) - 1:  # id column is the last column
                col_mask[-1] = True
                col_mask[ind] = True
            else:
                raise ValueError(
                    "Distance file must have the 'id', 'index' or '__index_level_0__' column as the first or last column for correct filtering, "
                    f" but got: index {cols.index(id_col)} of {len(cols)} columns"
                )

            # Which columns do we need to load? Keep `__index__` in the lazy
            # selection so projection pushdown doesn't prune the column the
            # filter depends on; drop it in pandas after collecting.
            cols_to_load = list(np.array(cols)[col_mask])
            dists = (
                dist_lazy.with_row_index(name="__index__")
                .filter(pl.col("__index__").is_in(ind))
                .select(["__index__"] + cols_to_load)
                .collect()
                .to_pandas()
                .drop(columns="__index__")
                .set_index(id_col)
            )

            if dists.shape[0] != dists.shape[1]:
                raise ValueError(
                    f"Distance matrix must be square after filtering, but got shape {dists.shape} with {len(ind)} requested IDs."
                )

            # Columns are likely strings whereas indices are likely integers, so we need to make sure they match
            dists.columns = dists.columns.astype(dists.index.dtype)
            if (dists.index != dists.columns).any():
                raise ValueError(
                    "Distance matrix index and columns must match after filtering, but got mismatching indices and columns."
                )

        if cache is not None:
            cache[key] = dists
        return dists

    def _load_features_df(self, *, file, ind, cache):
        """Load a single feature matrix (filtered if `ind` given)."""
        key = ("feat", str(file))
        if cache is not None and key in cache:
            return cache[key]

        if ind is None:
            # No filter: read as-is (file is expected to carry its id index).
            features = self.load_file(file)
        else:
            features_lazy = self.load_file_lazy(file)
            cols = features_lazy.collect_schema().names()
            id_col = self._detect_id_column(cols, "Features")
            features = (
                features_lazy.with_row_index(name="__index__")
                .filter(pl.col("__index__").is_in(ind))
                .drop("__index__")
                .collect()
                .to_pandas()
                .set_index(id_col)
            )

        if cache is not None:
            cache[key] = features
        return features

    def compile(self, progress_callback=None):
        """Load and return all dataset components.

        This is the main method to load the entire dataset at once as it takes care of:
          - loading the data with a given filter query
          - aligning the metadata, distances, embeddings and features
          - sanity checking

        Returns
        -------
        dict
            Dictionary with keys:
            - "meta": Pandas DataFrame of metadata
            - "distances": Pandas DataFrame of pairwise distances of the ACTIVE
              embedding (or None) - kept for backward compatibility
            - "features": Pandas DataFrame of feature vectors of the ACTIVE
              embedding (or None) - kept for backward compatibility
            - "embeddings": NumPy array of the ACTIVE low-dimensional embedding
              (or None) - kept for backward compatibility
            - "embedding_entries": list of all embedding entries, each a dict with
              keys "name", "embedding", "features", "distances"
            - "active_embedding": index of the active embedding (or None)

        """
        specs = self.normalized_embedding_specs

        # Build the (possibly filtered) metadata and the row indices `ind` used to
        # filter every other artifact. `ind is None` means "no filter".
        if self.filter_expr is None:
            report_if_callback(progress_callback, value=0, text="Loading meta data...")
            filtered_meta = self.meta
            ind = None
        else:
            report_if_callback(progress_callback, value=0, text="Loading meta data...")
            meta_lazy = self.load_file_lazy(self._get_file_spec("meta"))
            expr = string_to_polars_filter(self.filter_expr)
            filtered_meta = (
                meta_lazy.with_row_index(name="__index__").filter(expr).collect().to_pandas()
            )
            # We need to be weary of duplicate IDs here - that's why we're using indices
            # instead of IDs for filtering all other data artifacts.
            ind = filtered_meta["__index__"].values
            filtered_meta = filtered_meta.drop(columns="__index__")

        data = {
            "meta": filtered_meta,
            "embeddings": None,
            "distances": None,
            "features": None,
            "embedding_entries": [],
            "active_embedding": None,
        }

        report_if_callback(progress_callback, value=10)

        if not specs:
            if getattr(self, "_orphan_sources", False):
                logger.warning(
                    "Project declares 'features'/'distances' but no 'embeddings'; "
                    "these sources will be ignored."
                )
        else:
            # A per-compile cache so a features/distances file shared by multiple
            # embeddings is loaded (and filtered) only once.
            cache = {}
            entries = []
            n = len(specs)
            for i, spec in enumerate(specs):
                report_if_callback(
                    progress_callback, text=f"Loading embedding '{spec['name']}'..."
                )

                emb = self._load_embedding_array(
                    columns=spec["emb_columns"],
                    file=spec["emb_file"],
                    filtered_meta=filtered_meta,
                    ind=ind,
                    cache=cache,
                )

                feats = None
                if spec["feat_spec"] is not None:
                    feats = self._load_features_df(
                        file=self._spec_to_file(spec["feat_spec"], key="features"),
                        ind=ind,
                        cache=cache,
                    )

                dists = None
                if spec["dist_spec"] is not None:
                    dists = self._load_distances_df(
                        file=self._spec_to_file(spec["dist_spec"], key="distances"),
                        ind=ind,
                        cache=cache,
                    )

                entries.append(
                    {
                        "name": spec["name"],
                        "embedding": emb,
                        "features": feats,
                        "distances": dists,
                    }
                )
                report_if_callback(progress_callback, value=int(10 + (i + 1) * 30 / n))

            # The first embedding is the active one; mirror its artifacts into the
            # top-level keys for backward compatibility.
            data["embedding_entries"] = entries
            data["active_embedding"] = 0
            data["embeddings"] = entries[0]["embedding"]
            data["features"] = entries[0]["features"]
            data["distances"] = entries[0]["distances"]

        if isinstance(self.info["meta"], dict):
            color = self.info["meta"].get("color", None)
        elif "color" in self.meta.columns:
            color = "color"
        else:
            color = None

        if color is not None:
            if isinstance(color, str) and color in data["meta"].columns:
                data["meta"]["_color"] = data["meta"][color]
            elif isinstance(color, dict):
                data["meta"]["_color"] = data["meta"]["dataset"].map(color)
            else:
                try:
                    color = tuple(cmap.Color(color).rgba)
                    data["meta"]["_color"] = [color] * len(data["meta"])
                except BaseException:
                    raise ValueError(f"Invalid color specification: {color}")
        else:
            data["meta"]["_color"] = "white"

        logger.debug(
            (
                "Loaded dataset with: \n"
                f" - {len(data['meta'])} metadata entries\n"
                f" - {data['distances'].shape if data['distances'] is not None else 'no'} distances\n"
                f" - {data['features'].shape if data['features'] is not None else 'no'} features\n"
                f" - {data['embeddings'].shape if data['embeddings'] is not None else 'no'} precomputed embeddings\n"
            )
        )

        # Make doubly sure that metadata index is a RangeIndex (this is important for the scatter plot selection tracking)
        data["meta"].reset_index(drop=True, inplace=True)

        return data


def report_if_callback(progress_callback, text=None, value=None):
    """Report progress if a callback is provided.

    Parameters
    ----------
    progress_callback : callable | None
        Progress callback function.
    text : str, optional
        Text to report.
    value : int, optional
        Progress value to report.

    """
    if progress_callback is None:
        return

    if text is not None:
        progress_callback.setLabelText(text)
    if value is not None:
        progress_callback.setValue(value)
