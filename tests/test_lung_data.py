"""Semantic reader tests using only small, synthetic sparse count matrices."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse


spec = importlib.util.spec_from_file_location("lung_data", Path(__file__).parents[1] / "src/lung_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PreparedLungData = module.PreparedLungData


class PreparedLungDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "prepared"
        self.bundle.mkdir()
        self.tables = {}
        self.write_table("tahoe_genes", pd.DataFrame({
            "gene_index": [2, 0, 3, 1], "ensembl_id": ["A", "B", "D", "C"],
            "gene_symbol": ["GA", "GB", "GD", "GC"], "token_id": [12, 10, 13, 11],
        }))
        self.write_table("tidyi_genes", pd.DataFrame({
            "gene_index": [3, 1, 0, 2], "ensembl_id": ["B", "A", "C", "E"],
            "gene_symbol": ["GB", "GA", "GC", "GE"],
        }))
        self.write_table("gene_map", pd.DataFrame({
            "common_gene_index": [2, 0, 1], "ensembl_id": ["C", "A", "B"],
            "gene_symbol": ["GC", "GA", "GB"], "tahoe_column": [1, 2, 0],
            "tidyi_column": [0, 1, 3], "tahoe_token_id": [11, 12, 10],
        }))
        self.a = np.array([[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]], dtype=np.int32)
        self.b = np.array([[110, 120, 130, 140], [111, 121, 131, 141]], dtype=np.int32)
        sparse.save_npz(self.root / "a.npz", sparse.csr_matrix(self.a))
        sparse.save_npz(self.root / "b.npz", sparse.csr_matrix(self.b))
        self.tahoe_cells = pd.DataFrame({
            "cell_id": ["tahoe:a2", "tahoe:b0", "tahoe:a0", "tahoe:b1"],
            "source_cell_id": ["a2", "b0", "a0", "b1"],
            "condition_id": ["tahoe:t1", "tahoe:t1", "tahoe:t2", "tahoe:t2"],
            "matrix_path": ["a.npz", "b.npz", "a.npz", "b.npz"],
            "matrix_row": [2, 0, 0, 1], "plate": ["P1"] * 4,
        })
        self.write_table("tahoe_cells", self.tahoe_cells)
        self.conditions = pd.DataFrame({
            "condition_id": ["tahoe:t1", "tahoe:t2", "tahoe:c1", "tahoe:c2"],
            "plate": ["P1"] * 4,
            "condition_type": ["treated", "treated", "control", "control"],
            "drug_id": ["drug1", "drug2", "DMSO", "DMSO"],
        })
        self.write_table("tahoe_conditions", self.conditions)
        self.write_table("tahoe_control_links", pd.DataFrame({
            "condition_id": ["tahoe:t1", "tahoe:t1"],
            "control_condition_id": ["tahoe:c1", "tahoe:c2"],
            "relationship_type": ["same_plate_control_candidate"] * 2,
            "individual_cell_pairing": ["not_available"] * 2,
        }))
        self.tidyi = np.array([[3, 5, 0, 7], [13, 15, 19, 0]], dtype=np.int32)
        sparse.save_npz(self.bundle / "tidyi.npz", sparse.csr_matrix(self.tidyi))
        self.write_table("tidyi_cells", pd.DataFrame({
            "cell_id": ["tidyi:source9", "tidyi:source2"],
            "source_cell_id": ["source9", "source2"],
            "matrix_row": [0, 1],
            "state_label_original": ["PI-/APC+", None], "plate": ["P9", "P2"],
        }))
        self.manifest = {"status": "complete", "tables": self.tables, "matrices": {"tidyi_counts": "tidyi.npz"}}
        self.write_manifest()
        self.reader = PreparedLungData(self.root, self.bundle)

    def write_table(self, name, frame):
        self.tables[name] = f"{name}.parquet"
        frame.to_parquet(self.bundle / self.tables[name], index=False)

    def write_manifest(self):
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_tahoe_cross_shard_batches_preserve_cells_counts_and_common_genes(self):
        expected_full = np.vstack([self.a[2], self.b[0], self.a[0], self.b[1]])
        for batch_size in (1, 2, 3, 9):
            with self.subTest(batch_size=batch_size):
                batches = list(self.reader.iter_tahoe(batch_size=batch_size))
                actual = sparse.vstack([counts for counts, _, _ in batches])
                cells = pd.concat([obs for _, obs, _ in batches], ignore_index=True)
                np.testing.assert_array_equal(actual.toarray(), expected_full[:, [2, 0, 1]])
                self.assertEqual(cells["cell_id"].to_list(), self.tahoe_cells["cell_id"].to_list())
                for counts, obs, genes in batches:
                    self.assertTrue(sparse.isspmatrix_csr(counts))
                    self.assertEqual(counts.dtype, np.dtype("int32"))
                    self.assertEqual(counts.shape, (len(obs), len(genes)))
                    self.assertLessEqual(len(obs), batch_size)
                    self.assertEqual(genes["ensembl_id"].to_list(), ["A", "B", "C"])

    def test_tahoe_full_genes_and_condition_filter(self):
        batches = list(self.reader.iter_tahoe(batch_size=3, condition_id="tahoe:t2", gene_space="full"))
        counts, cells, genes = batches[0]
        np.testing.assert_array_equal(counts.toarray(), np.vstack([self.a[0], self.b[1]]))
        self.assertEqual(cells["cell_id"].to_list(), ["tahoe:a0", "tahoe:b1"])
        self.assertEqual(genes["ensembl_id"].to_list(), ["B", "C", "A", "D"])

    def test_unused_categorical_paths_do_not_create_cells_or_file_reads(self):
        paths = pd.Categorical(self.tahoe_cells["matrix_path"], categories=["a.npz", "b.npz", "unused.npz"])
        self.write_table("tahoe_cells", self.tahoe_cells.assign(matrix_path=paths))
        batches = list(self.reader.iter_tahoe(batch_size=1))
        self.assertEqual(sum(len(cells) for _, cells, _ in batches), len(self.tahoe_cells))

    def test_tidyi_alignment_and_missing_original_labels(self):
        counts, cells, genes = self.reader.load_tidyi()
        np.testing.assert_array_equal(counts.toarray(), self.tidyi[:, [1, 3, 0]])
        self.assertEqual(cells["source_cell_id"].to_list(), ["source9", "source2"])
        self.assertEqual(cells.loc[0, "state_label_original"], "PI-/APC+")
        self.assertTrue(pd.isna(cells.loc[1, "state_label_original"]))
        self.assertEqual(genes["ensembl_id"].to_list(), ["A", "B", "C"])
        full, _, full_genes = self.reader.load_tidyi(gene_space="full")
        np.testing.assert_array_equal(full.toarray(), self.tidyi)
        self.assertEqual(full_genes["ensembl_id"].to_list(), ["C", "A", "E", "B"])

    def test_tidyi_shuffled_metadata_uses_explicit_matrix_rows(self):
        shuffled = self.reader.table("tidyi_cells").iloc[::-1]
        self.write_table("tidyi_cells", shuffled)
        for gene_space, columns in (("common", [1, 3, 0]), ("full", [0, 1, 2, 3])):
            with self.subTest(gene_space=gene_space):
                counts, cells, _ = self.reader.load_tidyi(gene_space=gene_space)
                np.testing.assert_array_equal(counts.toarray(), self.tidyi[[1, 0]][:, columns])
                self.assertEqual(cells["cell_id"].to_list(), shuffled["cell_id"].to_list())
                self.assertEqual(cells["matrix_row"].to_list(), [1, 0])
                self.assertTrue(pd.isna(cells.loc[0, "state_label_original"]))
                self.assertEqual(cells.loc[1, "state_label_original"], "PI-/APC+")

    def test_tidyi_invalid_or_missing_matrix_rows_are_rejected(self):
        original = self.reader.table("tidyi_cells")
        for rows in ([0, 0], [0, 2], [-1, 0], [0.0, 1.0], [0, None]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "matrix_row"):
                self.write_table("tidyi_cells", original.assign(matrix_row=rows))
                self.reader.load_tidyi()
        self.write_table("tidyi_cells", original.drop(columns="matrix_row"))
        with self.assertRaisesRegex(ValueError, "matrix_row"):
            self.reader.load_tidyi()

    def test_control_candidates_are_metadata_and_on_the_same_plate(self):
        candidates = self.reader.control_candidates("tahoe:t1")
        self.assertEqual(candidates["control_condition_id"].to_list(), ["tahoe:c1", "tahoe:c2"])
        self.assertTrue(candidates["control_plate"].eq("P1").all())
        self.assertTrue(candidates["individual_cell_pairing"].eq("not_available").all())
        self.assertNotIn("cell_id", candidates)
        self.assertTrue(self.reader.control_candidates("tahoe:c1").empty)
        self.write_table("tahoe_conditions", self.conditions.assign(plate=["P1", "P1", "P2", "P1"]))
        with self.assertRaisesRegex(ValueError, "same plate"):
            self.reader.control_candidates("tahoe:t1")

    def test_unknown_names_and_invalid_batch_size_are_rejected(self):
        with self.assertRaises(KeyError):
            self.reader.table("../tahoe_cells")
        with self.assertRaises(KeyError):
            list(self.reader.iter_tahoe(condition_id="unknown"))
        with self.assertRaises(KeyError):
            self.reader.control_candidates("unknown")
        with self.assertRaisesRegex(ValueError, "no prepared cells"):
            list(self.reader.iter_tahoe(condition_id="tahoe:c1"))
        for value in (0, -1, True, 1.5, "2"):
            with self.subTest(batch_size=value), self.assertRaises(ValueError):
                list(self.reader.iter_tahoe(batch_size=value))
        with self.assertRaises(ValueError):
            self.reader.genes(gene_space="unsupported")
        with self.assertRaises(ValueError):
            self.reader.genes(dataset="unsupported")

    def test_paths_outside_allowed_directories_are_rejected(self):
        self.write_table("tahoe_cells", self.tahoe_cells.assign(matrix_path=["../outside.npz", "b.npz", "a.npz", "b.npz"]))
        with self.assertRaisesRegex(ValueError, "escapes"):
            list(self.reader.iter_tahoe())
        self.reader.manifest["tables"]["tidyi_cells"] = "../outside.parquet"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.reader.table("tidyi_cells")
        self.reader.manifest["matrices"]["tidyi_counts"] = "../outside.npz"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.reader.load_tidyi()
        with self.assertRaises(ValueError):
            PreparedLungData(self.root, self.root.parent)

    def test_duplicate_cells_and_source_rows_are_rejected(self):
        for frame in (
            self.tahoe_cells.assign(cell_id=["same"] * 4),
            self.tahoe_cells.assign(matrix_path=["a.npz"] * 4, matrix_row=[0] * 4),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                self.write_table("tahoe_cells", frame)
                list(self.reader.iter_tahoe())

    def test_wrong_row_and_gene_mappings_are_rejected(self):
        self.write_table("tahoe_cells", self.tahoe_cells.assign(matrix_row=[99, 0, 0, 1]))
        with self.assertRaisesRegex(ValueError, "out of range"):
            list(self.reader.iter_tahoe())
        mapping = self.reader.table("gene_map")
        self.write_table("gene_map", mapping.assign(tidyi_column=[1, 0, 3]))
        with self.assertRaisesRegex(ValueError, "Ensembl IDs"):
            self.reader.load_tidyi()

    def test_incomplete_bundle_and_tidyi_row_mismatch_are_rejected(self):
        self.manifest["status"] = "building"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "complete"):
            PreparedLungData(self.root, self.bundle)
        self.write_table("tidyi_cells", self.reader.table("tidyi_cells").iloc[:1])
        with self.assertRaisesRegex(ValueError, "count rows"):
            self.reader.load_tidyi()


if __name__ == "__main__":
    unittest.main()
