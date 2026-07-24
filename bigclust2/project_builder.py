"""Author BigClust projects from Python.

A BigClust *project* is just a directory with an ``info`` file plus a handful of
parquet tables (see the `data format
<https://flyconnectome.github.io/bigclust2/reference/data-format/>`_).
:class:`ProjectBuilder` writes exactly that layout, so you can build a project
without hand-editing JSON or worrying about the file naming conventions::

    from bigclust2 import ProjectBuilder

    builder = ProjectBuilder("my_project/", name="My clustering")
    builder.set_meta(meta_df, color_column="color")
    builder.add_embedding(columns=["x", "y"], distances=dist_df)
    builder.set_neuroglancer(source="source", neuropil_mesh="brain.ply")
    builder.save()

The same class backs the **File -> Build Project** GUI dialog, so both entry
points produce identical projects. This module is deliberately Qt-free (only
``numpy``/``pandas``) so it can be used in headless data-prep scripts.
"""

import re
import json

import numpy as np
import pandas as pd

from pathlib import Path

__all__ = ["ProjectBuilder", "plan_meta_remap", "apply_meta_remap"]

# Columns a BigClust meta table must provide.
_REQUIRED_META = ("id", "label", "dataset")


def _dedupe_name(name, used):
    """Return ``name`` or the first ``name_2``, ``name_3`` … not in ``used``."""
    if name not in used:
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    return f"{name}_{i}"


def plan_meta_remap(columns, id_col, label_col, dataset_col):
    """Plan mapping arbitrary columns onto the required ``id``/``label``/``dataset``.

    A user's table may name these fields differently (or not have them). This
    resolves a mapping to a set of output columns without ever silently dropping
    data: a reserved-named column (``id``/``label``/``dataset``) that the user did
    *not* map to that field is preserved under a suffixed name so it survives the
    round-trip, and a warning explains the rename.

    Parameters
    ----------
    columns :   list of str
                The source table's column names, in order.
    id_col, label_col, dataset_col : str
                Which source column supplies each required field.

    Returns
    -------
    dict with keys:
        ``assignments`` -- ``{"id": id_col, "label": label_col, "dataset": dataset_col}``
        ``carry``       -- list of ``(source_col, final_name)`` for the columns kept
                           alongside the required three (renamed where necessary).
        ``final_columns`` -- the output column names, required three first.
        ``field_warnings`` -- ``{field: message}`` for each required field whose
                           name clashes with an unmapped column that gets renamed.
    """
    columns = [str(c) for c in columns]
    mapping = {"id": id_col, "label": label_col, "dataset": dataset_col}
    sources = {id_col, label_col, dataset_col}

    used = set(_REQUIRED_META)
    carry = []
    field_warnings = {}
    for col in columns:
        if col in _REQUIRED_META:
            # A reserved-named source column. If it is the column mapped to its own
            # field, or to some other field, its values already live in a required
            # column and it must not be duplicated. Otherwise it clashes with the
            # field we are about to synthesise, so keep it under a safe name.
            if col == mapping[col] or col in sources:
                continue
            new = _dedupe_name(f"{col}_original", used)
            used.add(new)
            carry.append((col, new))
            # Keyed by the field (which shares the clashing column's name) so the
            # UI can show the note next to the offending row.
            field_warnings[col] = (
                f"A column named '{col}' already exists; it will be kept as "
                f"'{new}' since a different column is mapped here."
            )
        else:
            new = _dedupe_name(col, used)
            used.add(new)
            carry.append((col, new))

    final_columns = list(_REQUIRED_META) + [final for _, final in carry]
    return {
        "assignments": mapping,
        "carry": carry,
        "final_columns": final_columns,
        "field_warnings": field_warnings,
    }


def apply_meta_remap(meta, id_col, label_col, dataset_col):
    """Return a copy of ``meta`` with columns remapped per :func:`plan_meta_remap`.

    The result always has ``id``, ``label`` and ``dataset`` columns (sourced from
    the named columns) plus every other column, renamed only where keeping its
    original name would clash with the required three.
    """
    missing = {c for c in (id_col, label_col, dataset_col) if c not in meta.columns}
    if missing:
        raise KeyError(f"Meta table has no column(s): {sorted(missing)}")

    plan = plan_meta_remap(list(meta.columns), id_col, label_col, dataset_col)
    out = pd.DataFrame(index=meta.index)
    out["id"] = meta[id_col].to_numpy()
    out["label"] = meta[label_col].to_numpy()
    out["dataset"] = meta[dataset_col].to_numpy()
    for source, final in plan["carry"]:
        out[final] = meta[source].to_numpy()
    return out


class ProjectBuilder:
    """Helper for building and exporting BigClust projects.

    Parameters
    ----------
    path :          str | pathlib.Path
                    Directory to write the project to (created if necessary).
    name :          str, optional
                    Project name (shown in the GUI). Defaults to the directory
                    name.
    description :   str, optional
                    Short project description.
    dataset :       str, optional
                    Dataset label recorded in the ``info`` file.
    date_created :  str | datetime, optional
                    Creation timestamp. Defaults to now.
    """

    META_FILE = "meta.parquet"
    DISTANCES_FILE = "distances_{i}.parquet"
    FEATURES_FILE = "features_{i}.parquet"
    EMBEDDINGS_FILE = "embeddings_{i}.parquet"
    INFO_FILE = "info"

    def __init__(
        self,
        path,
        name=None,
        description=None,
        dataset=None,
        date_created=None,
    ):
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)

        self.name = name if name is not None else self.path.stem
        self.description = description if description is not None else ""
        self.dataset = dataset if dataset is not None else ""
        self.date_created = (
            pd.Timestamp(date_created).isoformat()
            if date_created is not None
            else pd.Timestamp.now().isoformat()
        )

        self._meta = None
        self._meta_source = None
        self._meta_color = None
        self._meta_last_updated = self.date_created

        # List of embedding entries (see `add_embedding`). Each entry is a dict with
        # keys: name, embeddings, columns, distances, distances_type, distances_metric,
        # features, features_type.
        self._embeddings = []

        self._neuroglancer_source = None
        self._neuroglancer_color = None
        self._neuropil_mesh = None
        self._neuroglancer_transforms = None

    def __repr__(self):
        return (
            f"ProjectBuilder(name={self.name!r}, path={str(self.path)!r}, "
            f"dataset={self.dataset!r})"
        )

    @staticmethod
    def _validate_meta_table(meta):
        if not isinstance(meta, pd.DataFrame):
            raise TypeError(f"Meta table must be a pandas DataFrame, got {type(meta)}")

        missing = {"id", "label", "dataset"} - set(meta.columns)
        if missing:
            raise ValueError(
                f"Meta table must contain columns 'id', 'label' and 'dataset'. Missing: {missing}"
            )

        return meta.copy()

    @staticmethod
    def _validate_embeddings(embeddings):
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError(
                f"Embeddings must be a pandas DataFrame, got {type(embeddings)}"
            )
        if embeddings.shape[1] != 2:
            raise ValueError(
                "Embeddings must contain exactly two columns for x/y coordinates."
            )
        return embeddings.copy()

    @staticmethod
    def _validate_square_matrix(matrix, name):
        if not isinstance(matrix, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame, got {type(matrix)}")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
        if not matrix.index.equals(matrix.columns):
            raise ValueError(f"{name} must have identical index and columns")
        return matrix.copy()

    @staticmethod
    def _validate_knn_matrix(matrix, name):
        for col in matrix.columns:
            # Make sure columns match either `nn_idx_{i}` or `nn_dist_{i}` for some integer i > 0
            if not re.match(r"nn_(idx|dist)_\d+", col):
                raise ValueError(
                    f"{name} must have columns named 'nn_idx_{{i}}' or 'nn_dist_{{i}}' for some integer i > 0"
                )
            # Make sure indices are integers
            if "idx" in col and not pd.api.types.is_integer_dtype(matrix[col]):
                raise ValueError(
                    f"{name} must have integer indices in columns named 'nn_idx_{{i}}'"
                )

        return matrix.copy()

    def set_meta(self, meta, source=None, color_column=None, last_updated=None):
        """Set the project's meta table.

        Parameters
        ----------
        meta :          pandas.DataFrame
                        One row per neuron. Must contain ``id``, ``label`` and
                        ``dataset`` columns.
        source :        str, optional
                        Name of a meta column holding per-row neuroglancer source
                        URLs (used by the 3D viewer).
        color_column :  str, optional
                        Name of a meta column holding per-row colors.
        last_updated :  str | datetime, optional
                        Timestamp recorded for the meta snapshot.
        """
        meta = self._validate_meta_table(meta)
        if color_column is not None and color_column not in meta.columns:
            raise ValueError(
                f"Color column '{color_column}' not found in meta table."
            )

        self._meta = meta
        self._meta_source = source
        self._meta_color = color_column
        if last_updated is not None:
            self._meta_last_updated = pd.Timestamp(last_updated).isoformat()

        return self

    def add_embedding(
        self,
        embeddings=None,
        name=None,
        columns=None,
        distances=None,
        distances_type="connectivity",
        distances_metric=None,
        features=None,
        features_type="connectivity",
    ):
        """Register an embedding (appends to the list of embeddings).

        Call this once per embedding. Each embedding can optionally carry its own
        high-dimensional `distances` and/or `features` sources.

        Parameters
        ----------
        embeddings :    pandas.DataFrame, optional
                        A DataFrame with exactly two columns (x/y coordinates). Provide
                        either this or `columns`.
        name :          str, optional
                        Display name for the embedding (shown in the GUI dropdown).
        columns :       list of str, optional
                        Names of columns in the meta table to use as x/y coordinates
                        instead of a separate embeddings file. Provide either this or
                        `embeddings`.
        distances :     pandas.DataFrame, optional
                        Square pairwise distance matrix paired with this embedding.
        distances_type : str
                        Type of the distances (e.g. "connectivity").
        distances_metric : str, optional
                        Metric used to compute the distances (e.g. "cosine").
        features :      pandas.DataFrame, optional
                        High-dimensional feature matrix paired with this embedding.
        features_type : str
                        Type of the features (e.g. "connectivity").
        """
        if (embeddings is None) == (columns is None):
            raise ValueError(
                "Provide exactly one of `embeddings` (a DataFrame) or `columns` "
                "(a list of meta column names)."
            )

        if embeddings is not None:
            embeddings = self._validate_embeddings(embeddings)
        else:
            if not isinstance(columns, (list, tuple)) or len(columns) != 2:
                raise ValueError(
                    "`columns` must be a list/tuple of exactly two meta column names."
                )
            columns = list(columns)

        # Record the identity of the originally-passed matrices so `save()` can write a
        # matrix shared across embeddings only once (the validation/copy below produces
        # distinct objects per call and would otherwise defeat the dedup).
        distances_id = id(distances) if distances is not None else None
        features_id = id(features) if features is not None else None

        if distances is not None:
            if "knn" not in distances_type:
                distances = self._validate_square_matrix(distances, "distances")
            else:
                distances = self._validate_knn_matrix(distances, "distances")

        if features is not None and not isinstance(features, pd.DataFrame):
            raise TypeError(
                f"Features must be a pandas DataFrame, got {type(features)}"
            )

        self._embeddings.append(
            {
                "name": name,
                "embeddings": embeddings,
                "columns": columns,
                "distances": distances,
                "distances_id": distances_id,
                "distances_type": distances_type if distances is not None else None,
                "distances_metric": distances_metric,
                "features": features.copy() if features is not None else None,
                "features_id": features_id,
                "features_type": features_type if features is not None else None,
            }
        )
        return self

    def clear_embeddings(self):
        """Remove all registered embeddings."""
        self._embeddings = []
        return self

    def _resolve_embedding_index(self, reference):
        """Resolve `reference` (int index or unique embedding name) to an index."""
        if isinstance(reference, str):
            matches = [
                i for i, e in enumerate(self._embeddings) if e["name"] == reference
            ]
            if not matches:
                raise ValueError(f"No embedding named {reference!r}.")
            if len(matches) > 1:
                raise ValueError(f"Embedding name {reference!r} is not unique.")
            return matches[0]

        idx = int(reference)
        if idx < 0:
            idx += len(self._embeddings)
        if not 0 <= idx < len(self._embeddings):
            raise IndexError(f"reference index {reference} out of range.")
        return idx

    def _embedding_coords(self, entry):
        """Return an (n, 2) float array of x/y coordinates for an embedding entry.

        Handles both storage forms: a standalone 2-column DataFrame
        (``entry["embeddings"]``) or two columns referenced in the meta table
        (``entry["columns"]``).
        """
        if entry["embeddings"] is not None:
            return entry["embeddings"].to_numpy(dtype=float)

        if self._meta is None:
            raise ValueError("Meta table must be set to read column-based embeddings.")
        missing = set(entry["columns"]) - set(self._meta.columns)
        if missing:
            raise ValueError(f"Embedding columns {missing} not found in meta table.")
        return self._meta.loc[:, entry["columns"]].to_numpy(dtype=float)

    def _write_embedding_coords(self, entry, coords):
        """Write transformed `coords` (n, 2) back into an embedding entry.

        DataFrame-type entries are overwritten in place (index and column names are
        preserved). Column-type entries are materialized into a private DataFrame
        rather than mutating the shared meta table: this flips the entry to a
        file-based embedding (which ``save`` already supports) so registration never
        corrupts `self._meta`.
        """
        if entry["embeddings"] is not None:
            df = entry["embeddings"]
            entry["embeddings"] = pd.DataFrame(
                coords, index=df.index, columns=df.columns
            )
        else:
            entry["embeddings"] = pd.DataFrame(
                coords, index=self._meta.index, columns=list(entry["columns"])
            )
            entry["columns"] = None

    @staticmethod
    def _fit_apply_affine(P, Q, mask):
        """Fit a full 2D affine (6 DOF) mapping P->Q on masked rows, apply to all P.

        Solves the least-squares homogeneous system ``[x, y, 1] @ M = [x', y']`` for
        the 3x2 matrix M, then transforms every row of P (non-finite rows map to
        non-finite outputs).
        """
        A = np.column_stack([P[mask], np.ones(int(mask.sum()))])  # (m, 3)
        M, *_ = np.linalg.lstsq(A, Q[mask], rcond=None)  # (3, 2)
        A_all = np.column_stack([P, np.ones(len(P))])  # (n, 3)
        return A_all @ M

    @staticmethod
    def _fit_apply_similarity(P, Q, mask):
        """Fit a 2D similarity (rotation + uniform scale + translation) P->Q.

        Uses the Umeyama closed form; the rotation is constrained to a proper
        rotation (determinant +1, i.e. no reflection). Applied to all rows of P.
        """
        X, Y = P[mask], Q[mask]
        m = X.shape[0]
        mu_x, mu_y = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - mu_x, Y - mu_y
        sigma = (Yc.T @ Xc) / m  # (2, 2) cross-covariance
        U, D, Vt = np.linalg.svd(sigma)
        S = np.eye(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # forbid reflection
            S[-1, -1] = -1
        R = U @ S @ Vt  # proper rotation (2, 2)
        scale = (D * np.diag(S)).sum() / ((Xc ** 2).sum() / m)
        t = mu_y - scale * (R @ mu_x)
        return scale * (P @ R.T) + t

    def register_embeddings(self, reference=0, transform="similarity"):
        """Align every other embedding to a reference embedding via landmark registration.

        All embeddings are row-aligned to the meta table (row ``i`` is the same
        neuron in every embedding), so each point is a corresponding landmark. For
        each non-reference embedding, this fits the transform that best maps its x/y
        onto the reference's x/y (least squares over all rows finite in both) and
        applies it to every point. Embeddings are modified in place; the reference is
        left untouched.

        Note that a column-based embedding (added via ``columns=``) is converted to a
        standalone embedding by this method: it stops tracking live edits to those
        meta columns and will be written to its own ``embeddings_{i}.parquet`` on
        ``save``.

        Parameters
        ----------
        reference : int | str
                    Index (supports negative indexing) or unique ``name`` of the
                    embedding to align the others to. Defaults to the first embedding.
        transform : "similarity" | "affine"
                    "similarity" fits a rotation + uniform scale + translation
                    (shape-preserving; no shear or reflection). "affine" fits a full
                    6-DOF affine (adds shear and reflection) for the tightest landmark
                    fit at the cost of possibly distorting the moving layout.
        """
        if transform not in ("affine", "similarity"):
            raise ValueError(
                f"Unknown transform {transform!r}; use 'affine' or 'similarity'."
            )
        if len(self._embeddings) < 2:
            raise ValueError("register_embeddings requires at least two embeddings.")

        ref_idx = self._resolve_embedding_index(reference)
        ref_coords = self._embedding_coords(self._embeddings[ref_idx])
        n = len(ref_coords)

        for i, entry in enumerate(self._embeddings):
            if i == ref_idx:
                continue

            moving = self._embedding_coords(entry)
            if len(moving) != n:
                raise ValueError(
                    f"Embedding {i} has {len(moving)} rows but reference has {n}; "
                    "all embeddings must be row-aligned to the meta table."
                )

            # Fit only on rows that are finite in both the moving and reference sets.
            mask = np.isfinite(moving).all(axis=1) & np.isfinite(ref_coords).all(axis=1)
            if int(mask.sum()) < 3:
                raise ValueError(
                    f"Embedding {i}: need >= 3 finite landmark points to fit a 2D "
                    f"{transform} transform (got {int(mask.sum())})."
                )

            if transform == "affine":
                new_coords = self._fit_apply_affine(moving, ref_coords, mask)
            else:
                new_coords = self._fit_apply_similarity(moving, ref_coords, mask)

            self._write_embedding_coords(entry, new_coords)

        return self

    def set_neuroglancer(
        self,
        source=None,
        color="color",
        neuropil_mesh=None,
        transforms=None,
    ):
        """Configure the neuroglancer (3D) viewer.

        Parameters
        ----------
        source :        str | dict, optional
                        A neuroglancer source URL, the name of a meta column
                        holding per-row source URLs, or a ``{dataset: url}`` map.
        color :         str, optional
                        Name of a meta column holding per-row colors.
        neuropil_mesh : str, optional
                        Path or URL of a neuropil mesh to show as context.
        transforms :    dict, optional
                        Optional per-dataset coordinate transforms.
        """
        self._neuroglancer_source = source
        self._neuroglancer_color = color
        self._neuropil_mesh = neuropil_mesh
        self._neuroglancer_transforms = transforms
        return self

    def _write_parquet(self, df, filename):
        path = self.path / filename
        try:
            df.to_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write {filename} to {path}: {exc}"
            ) from exc
        return path

    def save(
        self,
        include_meta=True,
        include_distances=True,
        include_features=True,
    ):
        """Write the project (``info`` + parquet files) to :attr:`path`.

        Returns the path to the written ``info`` file.
        """
        if include_meta and self._meta is None:
            raise ValueError("Meta table must be set before saving a BigClust project.")

        if not self._embeddings:
            raise ValueError(
                "A BigClust project requires at least one embedding. "
                "Use `add_embedding` before saving."
            )

        info = {
            "name": self.name,
            "description": self.description,
            "dataset": self.dataset,
            "date_created": self.date_created,
            "meta": {
                "file": self.META_FILE,
                "last_updated": self._meta_last_updated,
            },
        }

        if self._meta_source is not None:
            info["meta"]["source"] = self._meta_source
        if self._meta_color is not None:
            info["meta"]["color"] = self._meta_color

        if include_meta:
            self._write_parquet(self._meta, self.META_FILE)

        # Cache of already-written matrices (keyed by object id) so that a
        # distances/features matrix shared by multiple embeddings is only
        # written to disk once and referenced by multiple entries.
        written = {}

        def _write_shared(df, src_id, filename):
            cached = written.get(src_id)
            if cached is not None:
                return cached
            self._write_parquet(df, filename)
            written[src_id] = filename
            return filename

        embeddings_info = []
        for i, entry in enumerate(self._embeddings):
            emb = {}
            if entry["name"] is not None:
                emb["name"] = entry["name"]

            if entry["columns"] is not None:
                if self._meta is not None:
                    missing = set(entry["columns"]) - set(self._meta.columns)
                    if missing:
                        raise ValueError(
                            f"Embedding columns {missing} not found in meta table."
                        )
                emb["columns"] = entry["columns"]
            else:
                filename = self.EMBEDDINGS_FILE.format(i=i)
                self._write_parquet(entry["embeddings"], filename)
                emb["file"] = filename

            if include_distances and entry["distances"] is not None:
                filename = _write_shared(
                    entry["distances"],
                    entry["distances_id"],
                    self.DISTANCES_FILE.format(i=i),
                )
                distances = {
                    "type": entry["distances_type"],
                    "file": filename,
                }
                if entry["distances_metric"] is not None:
                    distances["metric"] = entry["distances_metric"]
                emb["distances"] = distances

            if include_features and entry["features"] is not None:
                filename = _write_shared(
                    entry["features"],
                    entry["features_id"],
                    self.FEATURES_FILE.format(i=i),
                )
                emb["features"] = {
                    "type": entry["features_type"],
                    "file": filename,
                }

            embeddings_info.append(emb)

        info["embeddings"] = embeddings_info

        neuroglancer = {}
        if self._neuroglancer_source is not None:
            neuroglancer["source"] = self._neuroglancer_source
        if self._neuroglancer_color is not None:
            neuroglancer["color"] = self._neuroglancer_color
        if self._neuropil_mesh is not None:
            neuroglancer["neuropil_mesh"] = self._neuropil_mesh
        if self._neuroglancer_transforms is not None:
            neuroglancer["transforms"] = self._neuroglancer_transforms
        if neuroglancer:
            info["neuroglancer"] = neuroglancer

        with open(self.path / self.INFO_FILE, "w") as handle:
            json.dump(info, handle, indent=4)

        return self.path / self.INFO_FILE
