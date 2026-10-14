"""Stream only v3 fit vehicle cells into source-preserving reference summaries."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration"
ASSETS = BASE / "training-identity-v3"
OUTPUT = BASE / "endpoint-alignment-v1/source_reviews/tahoe_reference_expression_round6"
CELL_COLUMNS = ["cell_id", "matrix_path", "matrix_row", "condition_id", "source_sample_id", "plate",
                "role", "outer_split", "observation_role", "source_id", "background_id"]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen input hash mismatch: {path}")


def reader_snapshot(out):
    paths = ["src/lung/data/reader.py", "src/lung/data/common.py", "src/lung/data/sparse_cache.py",
             "src/lung/data/preprocess.py", "src/lung_data.py"]
    payload = {name: (ROOT / name).read_bytes() for name in paths}
    snapshot = out / "source_snapshots"
    snapshot.mkdir(exist_ok=True)
    records = []
    for name, content in payload.items():
        target = snapshot / name.replace("/", "_")
        target.write_bytes(content)
        records.append({"original_path_at_import": name, "snapshot": target.relative_to(out).as_posix(),
                        "sha256": hashlib.sha256(content).hexdigest()})
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    reader = importlib.import_module("lung.data.reader")
    common = importlib.import_module("lung.data.common")
    cache = importlib.import_module("lung.data.sparse_cache")
    prepared = importlib.import_module("src.lung_data")
    if any((ROOT / name).read_bytes() != content for name, content in payload.items()):
        raise ValueError("Reader source changed during import; retry before reading expression")
    (out / "reader_source_versions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return reader, common, cache, prepared


def validate_control_rows(rows, controls):
    if rows.cell_id.duplicated().any() or rows.duplicated(["matrix_path", "matrix_row"]).any():
        raise ValueError("Duplicate cell identity or physical CSR row")
    if not pd.api.types.is_integer_dtype(rows.matrix_row) or rows.matrix_row.lt(0).any():
        raise ValueError("CSR row must be a nonnegative integer")
    required = {"role": "fit", "outer_split": "train", "observation_role": "contemporaneous_control",
                "source_id": "Tahoe-100M", "background_id": "CVCL_0023"}
    if any(not rows[key].eq(value).all() for key, value in required.items()):
        raise ValueError("Only T/fit contemporaneous A549 controls may be read")
    if not controls.condition_id.is_unique or not controls.is_vehicle.all() or not controls.condition_type.eq("control").all():
        raise ValueError("Control catalog is ambiguous or includes treated conditions")
    table = controls.set_index("condition_id")
    if not rows.condition_id.isin(table.index).all():
        raise ValueError("Cell points to an unapproved control condition")
    for key in ["source_sample_id", "plate"]:
        if not np.array_equal(rows[key].to_numpy(), table.loc[rows.condition_id, key].to_numpy()):
            raise ValueError(f"Control metadata identity mismatch: {key}")


def verify_obs_identity(rows, obs):
    if rows.matrix_row.ge(len(obs)).any():
        raise ValueError("CSR row exceeds source observation table")
    original = obs.iloc[rows.matrix_row.to_numpy()].reset_index(drop=True)
    if (not np.array_equal(rows.cell_id.to_numpy(), ("tahoe:" + original.BARCODE_SUB_LIB_ID).to_numpy())
            or not np.array_equal(rows.source_sample_id.to_numpy(), original["sample"].to_numpy())
            or not np.array_equal(rows.plate.to_numpy(), original.plate.to_numpy())
            or not original.drug.eq("DMSO_TF").all() or not original.cell_line_id.eq("CVCL_0023").all()):
        raise ValueError("Actual source barcode/sample/plate/vehicle identity mismatch")


def accumulate(block, sample_indices, count_sum, detected, n_cells):
    # SparseRows gives independent selected-row CSR arrays; cleanup never mutates source mappings.
    block = block.copy()
    block.sum_duplicates()
    block.eliminate_zeros()
    if not np.isfinite(block.data).all() or (block.data < 0).any() or not np.equal(block.data, np.floor(block.data)).all():
        raise ValueError("Reference expression must contain finite nonnegative integer raw counts")
    if block.shape[0] != len(sample_indices) or block.shape[1] != count_sum.shape[1]:
        raise ValueError("Selected expression shape disagrees with identity/axis")
    for index in np.unique(sample_indices):
        selected = block[sample_indices == index]
        count_sum[index] += np.asarray(selected.sum(axis=0, dtype=np.int64)).ravel()
        detected[index] += selected.getnnz(axis=0)
        n_cells[index] += selected.shape[0]


def averages(count_sum, detected, n_cells):
    means = np.full(count_sum.shape, np.nan, dtype=np.float64)
    rates = np.full(count_sum.shape, np.nan, dtype=np.float64)
    present = n_cells > 0
    means[present] = count_sum[present] / n_cells[present, None]
    rates[present] = detected[present] / n_cells[present, None]
    return means, rates, present


def difference(left, right, means, rates, counts, names, comparison):
    result = {"comparison": comparison, "left": names[left], "right": names[right],
              "left_n_fit_cells": int(counts[left]), "right_n_fit_cells": int(counts[right]),
              "status": "descriptive_only_not_state_or_significance"}
    if counts[left] == 0 or counts[right] == 0:
        return {**result, "status": "missing_fit_reference", "raw_mean_mae": None, "raw_mean_rmse": None,
                "detection_rate_mae": None, "raw_mean_vectors_identical": None}
    delta = means[left] - means[right]
    return {**result, "raw_mean_mae": float(np.mean(np.abs(delta))), "raw_mean_rmse": float(np.sqrt(np.mean(delta ** 2))),
            "detection_rate_mae": float(np.mean(np.abs(rates[left] - rates[right]))),
            "raw_mean_vectors_identical": bool(np.array_equal(means[left], means[right]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-rows", type=int, default=256)
    args = parser.parse_args()
    if args.batch_rows < 1:
        raise ValueError("Batch rows must be positive")
    out = args.output.resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError("Reference version already exists; choose a new output")
    out.mkdir(parents=True, exist_ok=True)
    reader_module, common, cache_module, prepared_module = reader_snapshot(out)
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    input_paths = [ASSETS / "manifest.json"]
    for name in ["conditions.parquet", "control_links.parquet", "genes.parquet", "train.parquet"]:
        path = ASSETS / name
        verify(path, manifest["files"][name]["sha256"])
        input_paths.append(path)
    conditions = pd.read_parquet(ASSETS / "conditions.parquet")
    controls = conditions.loc[conditions.condition_type.eq("control")].copy()
    controls = controls.assign(plate_order=controls.plate.str.extract(r"(\d+)", expand=False).astype(int)).sort_values(["plate_order", "source_sample_id"]).reset_index(drop=True)
    links = pd.read_parquet(ASSETS / "control_links.parquet")
    genes = pd.read_parquet(ASSETS / "genes.parquet")
    if len(controls) != 28 or controls.plate.nunique() != 14 or not controls.groupby("plate").size().eq(2).all() or len(genes) != 32461:
        raise ValueError("Expected v3 28 controls, fourteen plates and 32,461 common genes")
    rows = pd.read_parquet(ASSETS / "train.parquet", columns=CELL_COLUMNS,
                           filters=[("role", "==", "fit"), ("condition_id", "in", controls.condition_id.tolist())])
    validate_control_rows(rows, controls)
    prepared = prepared_module.PreparedLungData(ROOT, BASE / "prepared/v1")
    input_paths.append(prepared.bundle_path / "manifest.json")
    recorded = {entry["path"]: entry for entry in prepared.manifest["outputs"]}
    for key in ["gene_map", "tahoe_genes", "tahoe_shards"]:
        name = prepared.manifest["tables"][key]
        path = prepared.bundle_path / name
        verify(path, recorded[name]["sha256"])
        input_paths.append(path)
    axis = prepared.genes("common", "tahoe")
    for key in ["common_gene_index", "tahoe_column", "ensembl_id"]:
        if not np.array_equal(axis[key].to_numpy(), genes[key].to_numpy()):
            raise ValueError("Training/prepared/source gene axis identity differs")
    registry = common.source_shard_registry(prepared)

    class ReadOnlyCSRCache(cache_module.CSRCache):
        def _build(self, *unused):
            raise PermissionError("This audit cannot create or rebuild shared CSR caches")

    reader = reader_module.SparseRows(ROOT, genes.tahoe_column, registry=registry, max_cache_bytes=0, max_open_mappings=1)
    reader.disk_cache = ReadOnlyCSRCache(ROOT, max_open_mappings=1)
    sample_index = {key: index for index, key in enumerate(controls.condition_id)}
    count_sum = np.zeros((len(controls), len(genes)), dtype=np.int64)
    detected = np.zeros_like(count_sum)
    counts = np.zeros(len(controls), dtype=np.int64)
    source_log = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    for number, (name, part) in enumerate(rows.groupby("matrix_path", sort=True), 1):
        record = registry[name]
        obs_path = ROOT / record["obs_path"]
        verify(obs_path, record["obs_sha256_from_checkpoint"])
        obs = pd.read_parquet(obs_path, columns=["BARCODE_SUB_LIB_ID", "sample", "plate", "drug", "cell_line_id"])
        verify_obs_identity(part, obs)
        for start in range(0, len(part), args.batch_rows):
            selected = part.iloc[start:start + args.batch_rows]
            block = reader.read(selected)
            indices = selected.condition_id.map(sample_index).to_numpy(dtype=int)
            accumulate(block, indices, count_sum, detected, counts)
            peak_rss = max(peak_rss, process.memory_info().rss)
        digest = reader.disk_cache._sources[(ROOT / name).resolve()][1]
        cache_dir = reader.disk_cache.directory / common.digest_json({"schema": 1, "source_sha256": digest})
        cache_manifest = json.loads((cache_dir / "manifest.json").read_text())
        source_log.append({"matrix_path": name, "matrix_sha256": digest, "obs_path": record["obs_path"],
                           "obs_sha256": record["obs_sha256_from_checkpoint"], "fit_vehicle_rows_read": len(part),
                           "cache_manifest": str(cache_dir.relative_to(ROOT) / "manifest.json"), "cache_manifest_sha256": sha(cache_dir / "manifest.json"),
                           **{f"cache_{field}_sha256": cache_manifest["arrays"][field]["sha256"] for field in ["data", "indices", "indptr"]}})
        if number % 100 == 0:
            print(f"reference_shards={number}, fit_vehicle_cells={counts.sum()}, rss_MiB={process.memory_info().rss / 2**20:.1f}", flush=True)
    if counts.sum() != len(rows) or not np.array_equal(counts, rows.condition_id.value_counts().reindex(controls.condition_id, fill_value=0).to_numpy()):
        raise ValueError("Cell totals do not match the exact fit-control selection")
    means, rates, present = averages(count_sum, detected, counts)
    plates = controls.plate.drop_duplicates().tolist()
    plate_sums, plate_detected, plate_counts = [], [], []
    within = []
    for plate in plates:
        mask = controls.plate.eq(plate).to_numpy()
        plate_sums.append(count_sum[mask].sum(axis=0)); plate_detected.append(detected[mask].sum(axis=0)); plate_counts.append(counts[mask].sum())
        left, right = np.flatnonzero(mask)
        within.append(difference(left, right, means, rates, counts, controls.source_sample_id.tolist(), "two_controls_same_plate"))
    plate_sums, plate_detected, plate_counts = np.asarray(plate_sums), np.asarray(plate_detected), np.asarray(plate_counts)
    plate_means, plate_rates, plate_present = averages(plate_sums, plate_detected, plate_counts)
    between = [difference(left, right, plate_means, plate_rates, plate_counts, plates, "cell_weighted_references_different_plates") for left, right in combinations(range(len(plates)), 2)]
    common_arrays = {"ensembl_ids": genes.ensembl_id.to_numpy(dtype=str), "common_gene_index": genes.common_gene_index.to_numpy(), "tahoe_column": genes.tahoe_column.to_numpy(), "scale": "raw_counts_no_added_normalization"}
    for name, identifiers, sums, detections, n, mean, rate, available in [
        ("sample_references.npz", controls.source_sample_id.to_numpy(dtype=str), count_sum, detected, counts, means, rates, present),
        ("plate_references.npz", np.asarray(plates), plate_sums, plate_detected, plate_counts, plate_means, plate_rates, plate_present)]:
        np.savez_compressed(out / name, ids=identifiers, raw_count_sum=sums, detected_cell_count=detections,
                            n_fit_cells=n, raw_count_mean=mean, detection_rate=rate, reference_available=available, **common_arrays)
        with np.load(out / name, allow_pickle=False) as saved:
            for key, expected in [("raw_count_sum", sums), ("detected_cell_count", detections), ("n_fit_cells", n), ("raw_count_mean", mean), ("detection_rate", rate), ("ensembl_ids", common_arrays["ensembl_ids"])]:
                np.testing.assert_array_equal(saved[key], expected)
    controls.assign(n_fit_cells=counts, reference_available=present, average_common_axis_library_size=means.sum(axis=1)).to_csv(out / "reference_samples.csv", index=False)
    rows.to_parquet(out / "fit_control_cell_identity.parquet", index=False)
    genes.to_parquet(out / "gene_axis.parquet", index=False)
    pd.DataFrame(source_log).to_csv(out / "source_reads.csv", index=False)
    pd.DataFrame(within + between).to_csv(out / "reference_differences.csv", index=False)
    old = BASE / "endpoint-alignment-v1/source_reviews/tahoe_state_design_round5/repeated_treatment_groups.csv"
    repeated = pd.read_csv(old)
    input_paths.append(old)
    reference_links = []
    condition_table = conditions.set_index("condition_id")
    for group in repeated.to_dict("records"):
        for condition_id in group["condition_ids"].split("|"):
            target = condition_table.loc[condition_id]
            if any(target[key] != group[key] for key in ["chemical_or_experiment_group_id", "dose_um", "duration_h"]):
                raise ValueError("Previously audited exact treatment group has changed")
            selected = links.loc[links.condition_id.eq(condition_id)]
            if len(selected) != 2 or not selected.plate.eq(target.plate).all():
                raise ValueError("Treatment lacks its two same-plate reference links")
            for link in selected.to_dict("records"):
                index = sample_index[link["control_condition_id"]]
                if controls.iloc[index].plate != target.plate:
                    raise ValueError("Reference source is on a different plate")
                reference_links.append({**link, "target_role_metadata_only": target.role, "reference_role": "fit",
                                        "reference_n_cells": int(counts[index]), "reference_available": bool(present[index]),
                                        "response_expression_values_read": False})
    pd.DataFrame(reference_links).to_csv(out / "repeated_treatment_reference_links.csv", index=False)
    validation = {"status": "fit_reference_expression_read_and_summarized_not_predictive_or_biological_validation",
                  "samples": len(controls), "plates": len(plates), "fit_vehicle_cells": len(rows), "csr_shards": len(source_log),
                  "common_genes": len(genes), "samples_with_no_fit_cells": controls.loc[~present, "source_sample_id"].tolist(),
                  "same_plate_control_pairs": len(within), "different_plate_reference_pairs": len(between),
                  "repeated_treatment_groups": len(repeated), "repeated_treatment_conditions": len({r["condition_id"] for r in reference_links}),
                  "all_source_cell_identities_checked": True, "summary_full_array_roundtrip_equal": True,
                  "roles_exposed_as_expression": ["fit"], "observation_roles_exposed_as_expression": ["contemporaneous_control"],
                  "val_test_tables_opened": False, "treated_expression_labels_analyzed": False,
                  "opaque_shared_source_and_cache_bytes_hashed": True, "shared_cache_created_or_modified": False,
                  "normalization_fitted": False, "training_or_prediction_run": False, "scientific_endpoint_values_accessed": False,
                  "measured_peak_rss_MiB": peak_rss / 2**20, "batch_rows": args.batch_rows,
                  "reference_difference_meaning": "raw_expression_input_differences_only; confounded_by_library_size_batch_and_quality; not_biological_states_or_significance"}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (out / "audit_tahoe_reference_expression_snapshot.py").write_bytes(Path(__file__).read_bytes())
    output_hashes = {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "training_signature": manifest["signature"],
              "inputs": {str(path.relative_to(ROOT)): sha(path) for path in input_paths},
              "outputs": output_hashes, "validation": validation}
    (out / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
