"""Read prepared lung datasets without normalization, splitting, or cell pairing."""

import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


class PreparedLungData:
    """Read a complete prepared bundle and return aligned counts/cells/genes.

    Tahoe batches preserve the order of ``tahoe_cells``. ``common`` selects
    columns in ``common_gene_index`` order; ``full`` preserves the source gene
    axis. Counts remain sparse and unchanged. Metadata indices are reset so
    their positional rows match the returned matrix.
    """

    def __init__(self, project_root=None, bundle_path=None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        if bundle_path is None:
            bundle_path = "08-datasets/LungCancer_A549_Integration/prepared/v1"
        bundle = Path(bundle_path)
        self.bundle_path = (bundle if bundle.is_absolute() else self.project_root / bundle).resolve()
        if not self.bundle_path.is_relative_to(self.project_root):
            raise ValueError("The prepared bundle must be inside project_root")
        with (self.bundle_path / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("status") != "complete":
            raise ValueError("Only a complete prepared bundle can be read")
        tables = self.manifest.get("tables")
        if not isinstance(tables, dict) or not all(isinstance(v, str) for v in tables.values()):
            raise ValueError("manifest.tables must map logical names to relative paths")

    def _path(self, value, base):
        if not isinstance(value, str) or not value or Path(value).is_absolute() or Path(value).drive:
            raise ValueError("Data paths must be nonempty relative paths")
        path = (base / value).resolve()
        if not path.is_relative_to(base) or not path.is_relative_to(self.project_root):
            raise ValueError(f"Data path escapes its allowed directory: {value}")
        return path

    def table(self, name):
        """Read only a logical table explicitly listed in the manifest."""
        tables = self.manifest["tables"]
        if name not in tables:
            raise KeyError(f"Unknown table: {name}")
        return pd.read_parquet(self._path(tables[name], self.bundle_path))

    @staticmethod
    def _unique(frame, column, label):
        if column not in frame or frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"{label}.{column} must contain unique, nonmissing values")

    @staticmethod
    def _integers(frame, column, label):
        if column not in frame or frame[column].isna().any() or not pd.api.types.is_integer_dtype(frame[column]):
            raise ValueError(f"{label}.{column} must contain integer indices")
        indices = frame[column].to_numpy(dtype=np.int64)
        if (indices < 0).any():
            raise ValueError(f"{label}.{column} cannot contain negative indices")
        return indices

    def _gene_view(self, gene_space, dataset):
        if dataset not in ("tahoe", "tidyi"):
            raise ValueError("dataset must be 'tahoe' or 'tidyi'")
        if gene_space not in ("common", "full"):
            raise ValueError("gene_space must be 'common' or 'full'")
        axis = self.table(f"{dataset}_genes")
        self._unique(axis, "ensembl_id", f"{dataset}_genes")
        self._integers(axis, "gene_index", f"{dataset}_genes")
        axis = axis.sort_values("gene_index", kind="stable").reset_index(drop=True)
        if not np.array_equal(axis["gene_index"], np.arange(len(axis))):
            raise ValueError("Full gene indices must cover the source columns exactly once")
        if gene_space == "full":
            return axis, np.arange(len(axis)), len(axis)
        common = self.table("gene_map")
        self._unique(common, "ensembl_id", "gene_map")
        self._integers(common, "common_gene_index", "gene_map")
        common = common.sort_values("common_gene_index", kind="stable").reset_index(drop=True)
        if not np.array_equal(common["common_gene_index"], np.arange(len(common))):
            raise ValueError("Common gene indices must be contiguous and unique")
        columns = self._integers(common, f"{dataset}_column", "gene_map")
        if len(np.unique(columns)) != len(columns) or (columns >= len(axis)).any():
            raise ValueError("Common gene mapping has duplicate or out-of-range source columns")
        if not np.array_equal(axis.iloc[columns]["ensembl_id"], common["ensembl_id"]):
            raise ValueError("Common gene mapping does not match source Ensembl IDs")
        return common.assign(gene_index=columns), columns, len(axis)

    def genes(self, gene_space="common", dataset="tahoe"):
        """Return gene metadata in exactly the same order as count columns."""
        return self._gene_view(gene_space, dataset)[0]

    @staticmethod
    def _counts(path, gene_count):
        matrix = sparse.load_npz(path)
        if not sparse.isspmatrix_csr(matrix):
            raise ValueError(f"Expected a CSR count matrix: {path}")
        if matrix.shape[1] != gene_count:
            raise ValueError("Count matrix width differs from the full gene axis")
        return matrix

    def _conditions(self):
        conditions = self.table("tahoe_conditions")
        self._unique(conditions, "condition_id", "tahoe_conditions")
        return conditions

    def iter_tahoe(self, batch_size=1024, condition_id=None, gene_space="common"):
        """Yield (CSR counts, cells, genes), caching at most one source shard.

        A batch may cross shard boundaries. The matrix row references, not the
        current table row numbers, determine which source counts are returned.
        Unknown conditions and invalid cell references raise instead of being
        silently omitted. Validation occurs when the iterator is consumed.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        genes, columns, gene_count = self._gene_view(gene_space, "tahoe")
        cells = self.table("tahoe_cells")
        self._unique(cells, "cell_id", "tahoe_cells")
        self._integers(cells, "matrix_row", "tahoe_cells")
        if cells["matrix_path"].isna().any() or cells.duplicated(["matrix_path", "matrix_row"]).any():
            raise ValueError("Tahoe cells must reference unique, nonmissing source rows")
        # Check every path before yielding any data, including paths not selected.
        paths = {value: self._path(value, self.project_root) for value in cells["matrix_path"].unique()}
        conditions = self._conditions()
        if not cells["condition_id"].isin(conditions["condition_id"]).all():
            raise ValueError("Tahoe cells reference unknown conditions")
        if condition_id is not None:
            if condition_id not in set(conditions["condition_id"]):
                raise KeyError(f"Unknown Tahoe condition: {condition_id}")
            cells = cells.loc[cells["condition_id"] == condition_id]
            if cells.empty:
                raise ValueError(f"Tahoe condition has no prepared cells: {condition_id}")
        cells = cells.reset_index(drop=True)
        cached_path, cached_matrix = None, None
        for start in range(0, len(cells), int(batch_size)):
            batch = cells.iloc[start:start + int(batch_size)].reset_index(drop=True)
            blocks, positions = [], []
            for matrix_path, group in batch.groupby("matrix_path", sort=False, observed=True):
                if matrix_path != cached_path:
                    cached_matrix = self._counts(paths[matrix_path], gene_count)
                    cached_path = matrix_path
                rows = group["matrix_row"].to_numpy(dtype=np.int64)
                if (rows >= cached_matrix.shape[0]).any():
                    raise ValueError(f"Tahoe matrix_row is out of range: {matrix_path}")
                blocks.append(cached_matrix[rows][:, columns])
                positions.extend(group.index.to_list())
            counts = sparse.vstack(blocks, format="csr")[np.argsort(positions)]
            yield counts, batch, genes.copy()

    def load_tidyi(self, gene_space="common"):
        """Align original TIDYI counts to explicit cell matrix_row references."""
        matrices = self.manifest.get("matrices", {})
        if "tidyi_counts" not in matrices:
            raise KeyError("manifest.matrices is missing tidyi_counts")
        path = self._path(matrices["tidyi_counts"], self.bundle_path)
        genes, columns, gene_count = self._gene_view(gene_space, "tidyi")
        cells = self.table("tidyi_cells").reset_index(drop=True)
        self._unique(cells, "cell_id", "tidyi_cells")
        matrix = self._counts(path, gene_count)
        if matrix.shape[0] != len(cells):
            raise ValueError("TIDYI count rows differ from the cell metadata")
        rows = self._integers(cells, "matrix_row", "tidyi_cells")
        if not np.array_equal(np.sort(rows), np.arange(matrix.shape[0])):
            raise ValueError("TIDYI matrix_row must cover all count rows exactly once")
        return matrix[rows][:, columns], cells, genes

    def control_candidates(self, condition_id):
        """Return same-plate control metadata, never paired cell observations."""
        conditions = self._conditions()
        indexed = conditions.set_index("condition_id")
        if condition_id not in indexed.index:
            raise KeyError(f"Unknown Tahoe condition: {condition_id}")
        links = self.table("tahoe_control_links")
        selected = links.loc[links["condition_id"] == condition_id].copy()
        if selected["control_condition_id"].duplicated().any():
            raise ValueError("Duplicate control candidate links")
        if not selected["control_condition_id"].isin(indexed.index).all():
            raise ValueError("Control links reference unknown conditions")
        if not selected["relationship_type"].eq("same_plate_control_candidate").all():
            raise ValueError("Only same-plate control candidate relationships are supported")
        if not selected["individual_cell_pairing"].eq("not_available").all():
            raise ValueError("Prepared control links cannot claim individual cell pairing")
        controls = indexed.loc[selected["control_condition_id"]]
        if (controls["plate"].isna().any()
                or not controls["plate"].eq(indexed.loc[condition_id, "plate"]).all()
                or not controls["condition_type"].eq("control").all()):
            raise ValueError("Control candidates must be control conditions on the same plate")
        metadata = conditions.rename(columns={col: f"control_{col}" for col in conditions.columns})
        return selected.merge(metadata, on="control_condition_id", how="left", validate="many_to_one")
