"""Build a versioned, aligned raw-count bundle from frozen selection-v1."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy
from scipy import sparse

from prepared_aux_tables import build_aux_tables

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lung_data import PreparedLungData

INTEGRATION = Path("08-datasets/LungCancer_A549_Integration")
SELECTION = INTEGRATION / "selection-v1"
TAHOE = Path("08-datasets/Tahoe-100M/2dc57900b7981cfcf5e211527169a0b006546a95")
TIDYI = Path("08-datasets/TIDYI/E-MTAB-16354/audit_sources")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def local_path(relative):
    require(not Path(relative).is_absolute() and not Path(relative).drive, "Expected a relative path")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "Source path escapes project")
    return path


def verify_selection():
    manifest = json_read(ROOT / SELECTION / "output_manifest.json")
    for item in manifest["outputs"]:
        path = local_path(item["path"])
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"],
                f"Frozen selection output changed: {item['path']}")
    audit = json_read(ROOT / TAHOE / "full_expression/coverage_audit.json")
    status = json_read(ROOT / TAHOE / "full_expression/download_status.json")
    require(status["state"] == "complete", "Tahoe download is not complete")
    require(audit["processed_shards"] == audit["expected_shards"] == 3388 and
            audit["missing_filter_counts"]["full"] == 0 and
            audit["barcode_sample_plate_checks"] == "passed" and
            audit["source_hashes"] == "fresh_all_processed_source_and_derived_hashes_verified",
            "Prior strict Tahoe coverage audit is not complete")
    return audit


class Bundle:
    def __init__(self, path):
        self.path = path
        self.tables = {}
        self.inputs = {SELECTION / "output_manifest.json",
                       TAHOE / "full_expression/coverage_audit.json",
                       TAHOE / "full_expression/download_status.json"}

    def read_csv(self, relative):
        self.inputs.add(relative)
        return pd.read_csv(ROOT / relative)

    def read_parquet(self, relative):
        self.inputs.add(relative)
        return pd.read_parquet(ROOT / relative)

    def write(self, name, relative, frame):
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        self.tables[name] = relative
        return frame


def prepare_genes_and_tidyi(bundle):
    tg = bundle.read_parquet(TAHOE / "full_expression/var.parquet")
    tg.insert(0, "gene_index", np.arange(len(tg)))
    ag = bundle.read_csv(TIDYI / "A549_author_gene_annotations.csv")
    source = TIDYI / "A549_author_collection_counts.npz"
    bundle.inputs.add(source)
    with np.load(ROOT / source, allow_pickle=False) as data:
        obs = bundle.read_csv(SELECTION / "selected_tidyi_cells.csv")
        rows = obs["matrix_row"].to_numpy()
        require(np.array_equal(data["cell_ids"][rows], obs["cell_id"]), "TIDYI cell ID alignment")
        require(np.array_equal(data["states"][rows], obs["cell_state"]), "TIDYI label alignment")
        require(np.array_equal(data["plates"][rows], obs["Plate"]), "TIDYI plate alignment")
        require(ag["ensembl_id"].is_unique, "Duplicate TIDYI gene IDs")
        ag = ag.set_index("ensembl_id").loc[data["gene_ids"]].reset_index()
        expression = data["expression"][rows]
        require(expression.shape == (743, 32858) and np.isfinite(expression).all() and
                (expression >= 0).all() and (expression <= np.iinfo(np.int32).max).all() and
                np.array_equal(expression, expression.astype(np.int32)), "Invalid TIDYI raw counts")
        counts = sparse.csr_matrix(expression.astype(np.int32))
    ag.insert(0, "gene_index", np.arange(len(ag)))
    require(len(tg) == 62710 and tg["ensembl_id"].is_unique, "Tahoe gene axis changed")
    common = tg.rename(columns={"gene_index": "tahoe_column", "token_id": "tahoe_token_id"}).merge(
        ag[["ensembl_id", "gene_index"]].rename(columns={"gene_index": "tidyi_column"}),
        on="ensembl_id", validate="one_to_one", sort=False)
    common.insert(0, "common_gene_index", np.arange(len(common)))
    frozen = bundle.read_csv(INTEGRATION / "2026-09-18/shared_gene_vocabulary.csv")
    require(len(common) == 32461 and np.array_equal(common["ensembl_id"], frozen["ensembl_id"]),
            "Common gene identities/order differ from frozen vocabulary")
    bundle.write("tahoe_genes", "gene_axes/tahoe.parquet", tg)
    bundle.write("tidyi_genes", "gene_axes/tidyi.parquet", ag)
    bundle.write("gene_map", "gene_map.parquet", common)
    obs = obs.rename(columns={"cell_id": "source_cell_id", "Cell_line": "cell_line",
                              "cell_state": "state_label_original", "Plate": "plate", "Well": "well",
                              "matrix_path": "source_matrix_path", "matrix_row": "source_matrix_row",
                              "sample": "source_sample_id"})
    obs.insert(0, "cell_id", "tidyi:" + obs["source_cell_id"])
    obs["matrix_row"] = np.arange(len(obs))
    obs["count_scale"] = "raw_counts"
    obs["label_scope"] = "original_sorted_state_not_natural_apoptosis_rate"
    bundle.write("tidyi_cells", "tidyi/cells.parquet", obs)
    sparse.save_npz(bundle.path / "tidyi/counts.npz", counts)
    return len(common)


def prepare_tahoe(bundle):
    conditions = bundle.read_csv(SELECTION / "selected_tahoe_conditions.csv")
    catalog = bundle.read_parquet(TAHOE / "metadata/drug_metadata.parquet")
    catalog = catalog.rename(columns={"drug": "drug_name_original"})
    catalog["drug_name"] = catalog["drug_name_original"].str.strip()
    require(catalog["drug_name"].is_unique and "DMSO_TF" not in set(catalog["drug_name"]),
            "Drug source labels need explicit duplicate review")
    catalog["component_role"] = "drug"
    catalog = pd.concat([catalog, pd.DataFrame([{
        "drug_name_original": "DMSO_TF", "drug_name": "DMSO_TF", "component_role": "vehicle_control"
    }])], ignore_index=True)
    catalog["drug_id"] = catalog["drug_name"].map(
        lambda label: "tahoe_drug:" + hashlib.sha256(label.encode()).hexdigest()[:16])
    catalog["id_basis"] = "trimmed_source_label_not_canonical_chemical_identity"
    catalog["annotation_scope"] = "source_metadata_including_author_assisted_annotations_not_measured_outcome"
    conditions["drug_name"] = conditions["drug_key"].str.strip()
    conditions = conditions.merge(catalog[["drug_name", "drug_id"]], on="drug_name", validate="many_to_one")
    require(len(conditions) == 1336 and conditions["drug_id"].notna().all(), "Condition drug join incomplete")
    conditions = conditions.rename(columns={"sample": "source_sample_id"})
    conditions.insert(0, "condition_id", "tahoe:" + conditions["source_sample_id"])
    conditions["cell_line"] = "A549"
    conditions["condition_type"] = np.where(conditions["selection_status"].eq("keep_control"), "control", "treated")
    # Source control placeholder 0 uM is not a measured solvent concentration.
    conditions["source_dose_value"] = conditions["dose_value"]
    conditions["source_dose_unit"] = conditions["dose_unit"]
    conditions.loc[conditions["condition_type"].eq("control"), ["dose_value", "dose_unit"]] = [np.nan, None]
    conditions["count_scale"] = "raw_counts"
    conditions["is_sequential"] = False
    catalog["used_in_selected_conditions"] = catalog["drug_id"].isin(conditions["drug_id"])
    cells = bundle.read_parquet(SELECTION / "selected_tahoe_cells.parquet").rename(columns={
        "BARCODE_SUB_LIB_ID": "source_cell_id", "sample": "source_sample_id",
        "cell_line": "cell_line_id", "csr_shard": "matrix_path"})
    cells.insert(0, "cell_id", "tahoe:" + cells["source_cell_id"])
    cells["condition_id"] = "tahoe:" + cells["source_sample_id"]
    cells["cell_line"] = "A549"
    cells = cells.sort_values(["matrix_path", "matrix_row"], kind="stable").reset_index(drop=True)
    require(len(cells) == 2558270 and cells["cell_id"].is_unique and
            not cells.duplicated(["matrix_path", "matrix_row"]).any() and
            cells["condition_id"].isin(conditions["condition_id"]).all(), "Tahoe selected cell identity error")
    counts = cells.groupby("condition_id").size()
    conditions["n_cells"] = conditions["condition_id"].map(counts)
    require(conditions["n_cells"].eq(conditions["expression_a549_cells"]).all(), "Condition counts differ")
    links = bundle.read_csv(SELECTION / "selected_tahoe_control_links.csv")
    links["condition_id"] = "tahoe:" + links["sample"]
    links["control_condition_id"] = "tahoe:" + links["control_sample"]
    indexed = conditions.set_index("condition_id")
    require(len(links) == 2616 and links["condition_id"].value_counts().eq(2).all(), "Control link count error")
    for role in ["condition_id", "control_condition_id"]:
        require(links[role].isin(indexed.index).all() and
                np.array_equal(indexed.loc[links[role], "plate"], links["plate"]), "Cross-plate control link")
    require(indexed.loc[links["control_condition_id"], "condition_type"].eq("control").all(), "Invalid controls")
    bundle.write("drug_catalog", "drug_catalog.parquet", catalog)
    bundle.write("tahoe_cells", "tahoe/cells.parquet", cells)
    bundle.write("tahoe_conditions", "tahoe/conditions.parquet", conditions)
    bundle.write("tahoe_control_links", "tahoe/control_links.parquet", links)
    return cells, conditions, catalog


def verify_shards(bundle, cells):
    """Reuse prior strict hashes; check current sizes/mtimes and all selected obs rows."""
    records = []
    for number, (relative, group) in enumerate(cells.groupby("matrix_path", sort=True), 1):
        matrix = local_path(relative)
        checkpoint_path = ROOT / TAHOE / "full_expression/checkpoints" / (matrix.stem + ".json")
        checkpoint = json_read(checkpoint_path)
        members = {entry["path"]: entry for entry in checkpoint["files"]}
        obs_relative = TAHOE / "full_expression/obs" / (matrix.stem + ".parquet")
        record = {"matrix_path": relative, "selected_rows": len(group), "source_rows": checkpoint["a549_rows"],
                  "obs_path": obs_relative.as_posix(), "checkpoint_sha256": digest(checkpoint_path),
                  "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix()}
        for role, path in [("csr", matrix), ("obs", ROOT / obs_relative)]:
            prior = members[path.relative_to(ROOT / TAHOE).as_posix()]
            stat = path.stat()
            require(stat.st_size == prior["bytes"] and stat.st_mtime_ns == prior["mtime_ns"],
                    f"Source changed after strict audit: {path.name}")
            record.update({f"{role}_bytes": stat.st_size, f"{role}_mtime_ns": stat.st_mtime_ns,
                           f"{role}_sha256_from_checkpoint": prior["sha256"]})
        obs = pd.read_parquet(ROOT / obs_relative)
        require(len(obs) == checkpoint["a549_rows"], "Shard obs length differs from checkpoint")
        rows = group["matrix_row"].to_numpy()
        require((rows >= 0).all() and (rows < len(obs)).all(), "Tahoe matrix row out of bounds")
        selected = obs.iloc[rows]
        for left, right in [("source_cell_id", "BARCODE_SUB_LIB_ID"), ("source_sample_id", "sample"),
                            ("plate", "plate"), ("cell_line_id", "cell_line_id")]:
            require(np.array_equal(group[left].astype(str), selected[right].astype(str)), f"Shard {left} alignment")
        records.append(record)
        if number % 500 == 0:
            print(f"Checked selected row references in {number} shards", flush=True)
    require(len(records) == 3388, "Unexpected selected shard coverage")
    return bundle.write("tahoe_shards", "tahoe/shards.parquet", pd.DataFrame(records))


def condition_candidates(bundle, conditions):
    endpoints = pd.read_parquet(bundle.path / bundle.tables["endpoints"])
    components = pd.read_parquet(bundle.path / bundle.tables["endpoint_components"])
    rna = pd.read_parquet(bundle.path / bundle.tables["macaya_rna_samples"])
    singles = components.groupby("endpoint_condition_id").filter(lambda x: len(x) == 1)
    singles = singles.loc[singles["component_role"].eq("drug")]
    endpoint_info = endpoints.drop_duplicates("endpoint_condition_id").set_index("endpoint_condition_id")
    rows = []
    for _, component in singles.iterrows():
        ep = endpoint_info.loc[component.endpoint_condition_id]
        matched = conditions.loc[conditions["cell_line"].eq(ep.cell_line) &
                                 conditions["drug_name"].eq(component.drug_name) &
                                 conditions["dose_unit"].eq(component.dose_unit) &
                                 conditions["dose_value"].eq(component.dose_value)]
        for _, treatment in matched.iterrows():
            rows.append({"expression_source": "tahoe", "expression_condition_id": treatment.condition_id,
                         "expression_time_h": treatment.treatment_hours,
                         "endpoint_condition_id": component.endpoint_condition_id, "endpoint_time_h": ep.time_h,
                         "cell_line": ep.cell_line, "drug_name": component.drug_name,
                         "dose_value": component.dose_value, "dose_unit": component.dose_unit})
        dose_column = component["source_dose_column"]
        if dose_column in rna:
            candidates = rna.loc[rna["cell_line"].eq(ep.cell_line) &
                                 rna["condition_original"].eq(component.drug_name) &
                                 rna[dose_column].eq(component.dose_value)].drop_duplicates("bulk_condition_id")
            for _, sample in candidates.iterrows():
                rows.append({"expression_source": "macaya_rna", "expression_condition_id": sample.bulk_condition_id,
                             "expression_time_h": sample.time_h, "endpoint_condition_id": component.endpoint_condition_id,
                             "endpoint_time_h": ep.time_h, "cell_line": ep.cell_line,
                             "drug_name": component.drug_name, "dose_value": component.dose_value,
                             "dose_unit": component.dose_unit})
    table = pd.DataFrame(rows)
    table["time_match"] = table["expression_time_h"].eq(table["endpoint_time_h"])
    table["link_level"] = "condition_candidate_only"
    table["sample_pairing"] = "not_established"
    table["chemical_form_equivalence"] = "source_name_match_only_not_independently_confirmed"
    table["culture_and_sampling_equivalence"] = "not_established"
    table["control_equivalence"] = "not_established"
    table["use_as_paired_training_target"] = False
    return bundle.write("cross_source_candidates", "cross_source_candidates.parquet", table)


def main():
    final = (ROOT / INTEGRATION / "prepared/v1").resolve()
    stage = final.with_name(".v1.building")
    require(stage.is_relative_to(ROOT / INTEGRATION) and final.is_relative_to(ROOT / INTEGRATION), "Unsafe build path")
    require(not final.exists() and not stage.exists(), "Bundle or staging already exists; refusing overwrite")
    audit = verify_selection()
    stage.mkdir(parents=True)
    bundle = Bundle(stage)
    print("Preparing gene axes and raw TIDYI counts", flush=True)
    common_count = prepare_genes_and_tidyi(bundle)
    print("Preparing Tahoe condition and cell indexes", flush=True)
    cells, conditions, catalog = prepare_tahoe(bundle)
    shards = verify_shards(bundle, cells)
    auxiliary = build_aux_tables(ROOT, stage)
    auxiliary["tables"]["endpoints"] = auxiliary["tables"].pop("endpoint_measurements")
    bundle.tables.update(auxiliary["tables"])
    bundle.inputs.update(Path(p) for p in auxiliary["inputs"])
    candidates = condition_candidates(bundle, conditions)
    backgrounds = bundle.read_csv(SELECTION / "lung_backgrounds_to_preserve.csv")
    bundle.write("lung_backgrounds_to_preserve", "auxiliary/lung_backgrounds_to_preserve.parquet", backgrounds)
    counts = {"tahoe_cells": len(cells), "tahoe_conditions": len(conditions),
              "tahoe_treated_conditions": int(conditions["condition_type"].eq("treated").sum()),
              "tahoe_control_conditions": int(conditions["condition_type"].eq("control").sum()),
              "tahoe_control_cells": int(conditions.loc[conditions["condition_type"].eq("control"), "n_cells"].sum()),
              "tahoe_shards": len(shards), "drug_catalog_entries": len(catalog),
              "selected_treated_drug_labels": int(conditions.loc[conditions["condition_type"].eq("treated"), "drug_id"].nunique()),
              "tidyi_cells": 743, "common_genes": common_count, "tahoe_full_genes": 62710,
              "tidyi_full_genes": 32858, "cross_source_candidates": len(candidates),
              "lung_background_metadata_entries": len(backgrounds), **auxiliary["counts"]}
    manifest = {"schema_version": 1, "version": "v1", "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(), "counts": counts,
                "tables": bundle.tables, "matrices": {"tidyi_counts": "tidyi/counts.npz"},
                "path_conventions": {"tables_and_matrices": "relative_to_bundle", "source_paths": "relative_to_project"},
                "scope": "raw_counts_and_metadata_only_no_model_training_or_split",
                "normalization": None, "feature_selection": "exact_Ensembl_intersection_only",
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "pandas": pd.__version__, "scipy": scipy.__version__, "pyarrow": pyarrow.__version__},
                "prior_tahoe_strict_audit_utc": audit["checked_utc"]}
    json_write(stage / "manifest.json", manifest)
    reader = PreparedLungData(ROOT, stage)
    x, obs, genes = reader.load_tidyi()
    require(x.shape == (743, 32461) and len(obs) == x.shape[0] and len(genes) == x.shape[1], "TIDYI read check")
    batches = []
    for condition in ["tahoe:smp_1625", "tahoe:smp_2523"]:
        count = 0
        for x, obs, genes in reader.iter_tahoe(batch_size=512, condition_id=condition):
            require(x.shape == (len(obs), common_count) and x.data.min(initial=0) >= 0, "Tahoe read check")
            count += len(obs)
        expected = int(conditions.set_index("condition_id").loc[condition, "n_cells"])
        require(count == expected and len(reader.control_candidates(condition)) == 2, "Condition read count")
        batches.append({"condition_id": condition, "cells_read": count, "gene_columns": common_count})
    checks = {"status": "passed", "purpose": "data_integrity_and_reader_checks_not_model_validation",
              "selected_tahoe_row_references_checked": len(cells), "shards_checked": len(shards),
              "selection_output_hashes": "verified_this_run",
              "tahoe_csr_hashes": "reused_prior_strict_audit_with_current_size_and_mtime_checks",
              "tahoe_all_selected_obs_identity_alignment": "passed", "tidyi_full_raw_counts_and_labels": "passed",
              "tidyi_label_counts": reader.table("tidyi_cells")["state_label_original"].value_counts().to_dict(),
              "real_tahoe_condition_reads": batches, "common_gene_exact_ID_alignment": "passed"}
    json_write(stage / "data_checks.json", checks)
    json_write(stage / "summary.json", counts)
    for source, target in [(Path("docs/DATA_PREPARED_V1.md"), "README.md"),
                           (Path("docs/DATA_FIELDS_V1.md"), "FIELD_DICTIONARY.md")]:
        (stage / target).write_text((ROOT / source).read_text(encoding="utf-8"), encoding="utf-8")
        bundle.inputs.add(source)
    # Include source hashes and generator versions, without duplicating large source data.
    bundle.inputs.update({Path("scripts/prepare_lung_dataset_v1.py"), Path("scripts/prepared_aux_tables.py"),
                          Path("src/lung_data.py")})
    manifest["inputs"] = [fingerprint(ROOT / p) for p in sorted(bundle.inputs)]
    manifest["source_assets"] = [{"path": p, "bytes": local_path(p).stat().st_size,
                                  "verification": "existence_and_size_only_this_build"}
                                 for p in auxiliary["source_assets"]]
    manifest["outputs"] = [{"path": p.relative_to(stage).as_posix(), "bytes": p.stat().st_size,
                            "sha256": digest(p)} for p in sorted(stage.rglob("*"))
                           if p.is_file() and p.name != "manifest.json"]
    json_write(stage / "manifest.json", manifest)
    # Both resolved paths were checked above; never overwrite an existing version.
    stage.rename(final)
    print(json.dumps({"bundle": str(final), "counts": counts, "checks": "passed"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
