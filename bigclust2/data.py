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
        return SingleProjectLoader(name=info.get('name', 'unnamed'), path=path, info=info)
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
        return self.info['features'].get('type', None)

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
        if value is None:
            return None
        if isinstance(value, dict):
            file = value.get("file")
            if file is None:
                raise ValueError(f"Missing 'file' entry for '{key}' in info.")
            return file
        return value

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
            if isinstance(self.info["embeddings"], dict) and "columns" in self.info["embeddings"]:
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

    def compile(self, progress_callback=None):
        """Load and return all dataset components.

        This is the main method to load the entire dataset at once as it takes care of:
          - loading the data with a given filter query
          - aligning the metadata, distances, and embeddings
          - sanity checking

        Returns
        -------
        dict
            Dictionary with keys:
            - "meta": Pandas DataFrame of metadata
            - "distances": Pandas DataFrame of pairwise distances (or None)
            - "features": Pandas DataFrame of feature vectors (or None)
            - "embeddings": NumPy array of low-dimensional embeddings (or None)

        """
        # If no filter, we can just return everything
        if self.filter_expr is None:
            data = {}

            for i, attr in enumerate(["meta", "distances", "features", "embeddings"]):
                if attr != "meta" and self.info.get(attr, None) is None:
                    data[attr] = None
                    continue

                report_if_callback(progress_callback, text=f"Loading {attr}...")

                data[attr] = getattr(self, attr)
                report_if_callback(progress_callback, value=int((i + 1) * 40 / 4))
        else:
            report_if_callback(progress_callback, value=0, text="Loading meta data...")

            meta_lazy = self.load_file_lazy(self._get_file_spec("meta"))
            expr = string_to_polars_filter(self.filter_expr)
            filtered_meta = meta_lazy.filter(expr).collect().to_pandas()
            data = {
                "meta": filtered_meta,
                "distances": None,
                "features": None,
                "embeddings": None,
            }

            # IDs to keep
            ids = filtered_meta.id.unique()

            report_if_callback(progress_callback, value=10)

            if self.info.get("embeddings", None) is not None:
                report_if_callback(progress_callback, text="Loading embeddings...")
                # Embeddings can be a separate file or columns in the meta file
                if "columns" in self.info["embeddings"]:
                    cols = self.info["embeddings"]["columns"]
                    data["embeddings"] = filtered_meta[cols].values
                else:
                    emb_lazy = self.load_file_lazy(self._get_file_spec("embeddings"))

                    # We're expecting either an `id` or an `index` column to filter on
                    cols = emb_lazy.collect_schema().names()
                    if "id" in cols:
                        id_col = "id"
                    elif "index" in cols:
                        id_col = "index"
                    else:
                        raise ValueError(
                            "Embeddings must have either an 'id' or 'index' column to filter on."
                        )

                    # Load relevant data, filter and sort by IDs
                    # This will raise if any IDs are missing
                    data["embeddings"] = (
                        (
                            emb_lazy.filter(pl.col(id_col).is_in(ids))
                            .collect()
                            .to_pandas()
                            .set_index(id_col)
                        )
                        .loc[ids]
                        .values
                    )

                    assert (
                        data["embeddings"].shape[1] == 2
                    ), f"Embeddings must have exactly 2 dimensions, got {data['embeddings'].shape[1]}"

                report_if_callback(progress_callback, value=20)

            if self.info.get("distances", None) is not None:
                report_if_callback(progress_callback, text="Loading distances...")
                # Lazy load distances
                dist_lazy = self.load_file_lazy(self._get_file_spec("distances"))

                # If these are pairwise distances, we need to filter on both rows and columns
                cols = dist_lazy.collect_schema().names()

                # We're expecting either an `id` or `index` column for the rows
                if "id" in cols:
                    id_col = "id"
                elif "index" in cols:
                    id_col = "index"
                else:
                    raise ValueError(
                        "Distances must have either an 'id' or 'index' column to filter on."
                    )

                # Next, we need to check that all IDs exists as columns too
                # (note that columns have to be strings)
                missing = ids[~np.isin(ids, cols)]
                if len(missing):
                    raise ValueError(
                        f"The following IDs are missing as distance columns: {missing}"
                    )

                dists = (
                    dist_lazy.filter(pl.col(id_col).is_in(ids))
                    .select(ids.astype(str))
                    .collect()
                    .to_pandas()
                    .set_index(id_col)
                )
                dists.columns = dists.columns.astype(dists.index.dtype)
                dists = dists.loc[ids, ids]
                data["distances"] = dists

                report_if_callback(progress_callback, value=30)

            if self.info.get("features", None) is not None:
                report_if_callback(progress_callback, text="Loading features...")
                # Lazy load features
                features_lazy = self.load_file_lazy(self._get_file_spec("features"))

                # We're expecting either an `id` or `index` column for the rows
                cols = features_lazy.collect_schema().names()
                if "id" in cols:
                    id_col = "id"
                elif "index" in cols:
                    id_col = "index"
                elif "__index_level_0__" in cols:
                    id_col = "__index_level_0__"
                else:
                    raise ValueError(
                        "Features must have either an 'id' or 'index' column to filter on."
                    )

                features = (
                    features_lazy.filter(pl.col(id_col).is_in(ids))
                    .collect()
                    .to_pandas()
                    .set_index(id_col)
                )
                data["features"] = features

                report_if_callback(progress_callback, value=40)

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
