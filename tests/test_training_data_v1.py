"""Training assets: tiny synthetic fixtures, not biological validation."""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from src.lung.data import TrainingData, prepare_training
from src.lung.data.chemistry import build_chemical_catalog
from src.lung.data.common import ROLES, sha256
from src.lung.data.preprocess import exact_neighbor_graph, log_normalize, weighted_moments
from src.lung.data.reader import SparseRows
from src.lung.data.splits import grouped_roles, tidyi_splits


def tiny_bundle(root):
    root = Path(root)
    bundle = root / "prepared"
    bundle.mkdir()
    (root / "configs").mkdir()
    config = {"bundle": "prepared", "assets": "training-v1", "gene_count": 6,
              "graph_neighbors": 2, "graph_block_size": 2, "stream_batch_size": 16}
    (root / "configs" / "dataset.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    tables = {}

    def table(name, frame):
        tables[name] = f"{name}.parquet"
        frame.to_parquet(bundle / tables[name], index=False)

    gene_ids = [f"ENSG{i:04d}" for i in range(6)]
    axis = pd.DataFrame({"gene_index": range(6), "ensembl_id": gene_ids})
    table("tahoe_genes", axis)
    table("tidyi_genes", axis)
    table("gene_map", pd.DataFrame({"common_gene_index": range(6), "ensembl_id": gene_ids,
                                    "tahoe_column": range(6), "tidyi_column": range(6)}))
    table("drug_catalog", pd.DataFrame([
        {"drug_id": f"drug{i}", "drug_name": f"synthetic{i}", "component_role": "drug",
         "canonical_smiles": "C" * (i + 1)} for i in range(20)] + [
        {"drug_id": "vehicle", "drug_name": "DMSO", "component_role": "vehicle",
         "canonical_smiles": "CS(C)=O"}]))
    conditions = pd.DataFrame([
        {"condition_id": f"condition{i}", "drug_id": f"drug{i}", "condition_type": "treated",
         "plate": "plate1", "source": "Tahoe-100M", "source_sample_id": f"sample{i}",
         "cell_line_id": "CVCL_0023", "dose_value": 1., "dose_unit": "uM", "treatment_hours": 24,
         "is_sequential": False} for i in range(20)] + [
        {"condition_id": "control", "drug_id": "vehicle", "condition_type": "control",
         "plate": "plate1", "source": "Tahoe-100M", "source_sample_id": "control_sample",
         "cell_line_id": "CVCL_0023", "dose_value": np.nan, "dose_unit": "uM",
         "treatment_hours": 24, "is_sequential": False}])
    table("tahoe_conditions", conditions)
    rows = []
    for condition in conditions.itertuples():
        for _ in range(120 if condition.condition_type == "control" else 12):
            i = len(rows)
            rows.append({"cell_id": f"cell{i:04d}", "condition_id": condition.condition_id,
                         "source_sample_id": condition.source_sample_id, "plate": "plate1",
                         "matrix_path": "source.npz", "matrix_row": i})
    table("tahoe_cells", pd.DataFrame(rows))
    rng = np.random.default_rng(12)
    counts = rng.poisson(3., size=(len(rows), 6)).astype(np.int32)
    counts[:, 0] += 1
    sparse.save_npz(root / "source.npz", sparse.csr_matrix(counts))
    table("tahoe_control_links", pd.DataFrame([
        {"condition_id": f"condition{i}", "control_condition_id": "control",
         "relationship_type": "same_plate_control_candidate", "individual_cell_pairing": "not_available"}
        for i in range(20)]))
    states = []
    for plate in (1, 8):
        for col in range(1, 25):
            for row, label in zip("ABC", ("PI-APC-", "PI-APC+", "PI+APC+")):
                states.append({"cell_id": f"state:{plate}:{row}{col}", "well": f"{row}{col}",
                               "plate": plate, "state_label_original": label, "matrix_row": len(states)})
    table("tidyi_cells", pd.DataFrame(states))
    sparse.save_npz(bundle / "tidyi.npz", sparse.csr_matrix(rng.poisson(3., (len(states), 6)) + 1))
    manifest = {"status": "complete", "tables": tables, "matrices": {"tidyi_counts": "tidyi.npz"},
                "outputs": [{"path": value, "sha256": sha256(bundle / value)} for value in tables.values()]}
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return counts


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counts = tiny_bundle(self.root)

    def test_full_assets_are_fit_only_and_source_is_unchanged(self):
        original = sha256(self.root / "source.npz")
        assets = prepare_training(self.root)
        reader = TrainingData(self.root)
        self.assertEqual(reader.manifest["status"], "complete")
        self.assertEqual(reader.neighbors.shape, (6, 3))
        self.assertEqual(reader.bandwidths.shape, (5,))
        cache = json.loads((assets / "fit_cache" / "index.json").read_text(encoding="utf-8"))
        self.assertLessEqual(max(part["rows"] for part in cache["parts"]), 16)
        fit = pd.read_parquet(assets / "train.parquet").query("role == 'fit'")
        y, _ = log_normalize(sparse.csr_matrix(self.counts[fit.matrix_row]))
        size = fit.groupby("condition_id")["cell_id"].transform("size").to_numpy()
        weights = 1. / fit.condition_id.nunique() / size
        expected = np.asarray(y.T @ weights).ravel()
        np.testing.assert_allclose(reader.mean, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(original, sha256(self.root / "source.npz"))
        self.assertEqual(prepare_training(self.root), assets)

    def test_no_cell_or_chemical_group_leakage_and_permissions(self):
        assets = prepare_training(self.root)
        containers = {name: pd.read_parquet(assets / f"{name}.parquet") for name in ("train", "val", "test")}
        sets = {name: set(frame.cell_id) for name, frame in containers.items()}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        merged = pd.concat(containers.values())
        self.assertEqual(merged.groupby("chemical_or_experiment_group_id")["role"].nunique().max(), 1)
        self.assertEqual(set(merged.role), set(ROLES))
        reader = TrainingData(self.root)
        with self.assertRaises(PermissionError):
            reader.condition_ids("final_test")
        with self.assertRaises(PermissionError):
            TrainingData(self.root, allowed_roles=("final_test",))
        testing = TrainingData(self.root, allowed_roles=("final_test",), allow_test=True)
        self.assertTrue(testing.condition_ids("final_test"))
        with self.assertRaises(PermissionError):
            testing.condition_ids("fit")

    def test_independent_response_sampling_and_state_labels(self):
        prepare_training(self.root)
        reader = TrainingData(self.root)
        condition = reader.condition_ids()[0]
        batch = reader.sample_response(condition, control_n=7, target_n=11, rng=np.random.default_rng(1))
        self.assertEqual(batch["control"].shape, (7, 6))
        self.assertEqual(batch["target"].shape, (11, 6))
        self.assertEqual(batch["control"].dtype, np.float32)
        self.assertFalse(set(batch["control_cell_ids"]) & set(batch["target_cell_ids"]))
        self.assertEqual(batch["condition"]["fingerprint"].shape, (2048,))
        self.assertEqual(batch["condition"]["duration_h"], 24.)
        x, labels = reader.state_arrays()
        self.assertEqual(x.shape[1], 6)
        self.assertEqual(set(labels), {0, 1, 2})
        with self.assertRaises(PermissionError):
            reader.sample_response(condition, role="selection")
        with self.assertRaises(ValueError):
            reader.sample_response(condition, control_n=0)

    def test_metadata_only_does_not_claim_preprocessing_ready(self):
        assets = prepare_training(self.root, build_preprocessing=False)
        self.assertFalse((assets / "normalization.npz").exists())
        with self.assertRaisesRegex(ValueError, "PREPROCESSING_REQUIRED"):
            TrainingData(self.root)
        reader = TrainingData(self.root, require_preprocessing=False)
        self.assertEqual(reader.manifest["status"], "metadata_ready")
        self.assertIsNone(reader.mean)

    def test_decision_root_never_reads_treated_response_rows(self):
        prepare_training(self.root)
        reader = TrainingData(self.root, allowed_roles=("final_test",), allow_test=True)
        original_read = reader.rows.read
        observed_roles = []
        def observed_read(rows):
            observed_roles.extend(rows["observation_role"].tolist())
            return original_read(rows)
        with patch.object(reader.rows, "read", side_effect=observed_read):
            sample = reader.sample_controls(reader.condition_ids("final_test")[0], n=8,
                                            rng=np.random.default_rng(3), role="final_test")
        self.assertEqual(set(observed_roles), {"contemporaneous_control"})
        self.assertEqual(sample["control"].shape, (8, 6))
        self.assertFalse(sample["response_outcomes_read"])

    def test_frozen_asset_tampering_is_detected(self):
        assets = prepare_training(self.root, build_preprocessing=False)
        with (assets / "fingerprints.npy").open("ab") as handle:
            handle.write(b"modified")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            TrainingData(self.root, require_preprocessing=False)
        with self.assertRaisesRegex(ValueError, "Frozen training asset"):
            prepare_training(self.root, build_preprocessing=False)

    def test_full_form_chemistry_does_not_desalt(self):
        catalog = pd.DataFrame({"drug_id": list("abcd"), "drug_name": list("ABCD"),
                                "component_role": ["drug"] * 4,
                                "canonical_smiles": ["CCO", "OCC", "CC[NH3+].[Cl-]", "CCN"]})
        chemical, fp, metadata = build_chemical_catalog(catalog)
        identities = chemical.chemical_or_experiment_group_id.to_list()
        self.assertEqual(identities[0], identities[1])
        self.assertNotEqual(identities[2], identities[3])
        self.assertEqual(fp.dtype, np.uint8)
        self.assertIn("no_desalting", metadata["canonicalization"])

    def test_graph_matches_dense_weighted_pearson_and_ties(self):
        values = np.array([[1., 4., 0., 0.], [2., 3., 1., 0.], [4., 2., 3., 0.]])
        weights = np.array([.2, .3, .5])
        matrix = sparse.csr_matrix(values)
        mean, scale, sd, mass = weighted_moments([(matrix, weights)], 4)
        def parts():
            yield matrix[:1], weights[:1]
            yield matrix[1:], weights[1:]
        genes = np.array(["B", "A", "C", "D"])
        actual = exact_neighbor_graph(parts, mean, sd, genes, neighbors=2, block_size=1)
        centered = values - mean
        cov = centered.T @ (weights[:, None] * centered)
        divisor = sd[:, None] * sd[None, :]
        corr = np.divide(cov, divisor, out=np.zeros_like(cov), where=divisor > 0)
        for i in range(4):
            expected = sorted((j for j in range(4) if j != i), key=lambda j: (-abs(corr[i, j]), genes[j]))
            self.assertEqual(actual[i, 0], i)
            self.assertEqual(actual[i, 1:].tolist(), expected[:2])
        self.assertEqual(actual[3, 1:].tolist(), [1, 0])
        self.assertAlmostEqual(mass, 1.)
        self.assertEqual(scale[3], .1)

    def test_state_missing_class_is_recorded_not_reseeded(self):
        original = pd.read_parquet(self.root / "prepared" / "tidyi_cells.parquet")
        subset = original.loc[~((original.plate == 8) & (original.state_label_original == "PI+APC+"))]
        first, report1 = tidyi_splits(subset)
        second, report2 = tidyi_splits(subset.iloc[::-1])
        self.assertEqual(report1, report2)
        self.assertEqual(report1["8_to_1"]["status"], "STATE_SPLIT_INSUFFICIENT")
        self.assertEqual(grouped_roles(range(20)), grouped_roles(reversed(range(20))))

    def test_sparse_row_reader_preserves_order_and_blocks_path_escape(self):
        reader = SparseRows(self.root, [3, 1])
        rows = pd.DataFrame({"matrix_path": ["source.npz"] * 3, "matrix_row": [8, 1, 8]})
        np.testing.assert_array_equal(reader.read(rows).toarray(), self.counts[[8, 1, 8]][:, [3, 1]])
        with self.assertRaises(ValueError):
            reader.read(rows.assign(matrix_path="../outside.npz"))


if __name__ == "__main__":
    unittest.main()
