"""Create explicit data selections without fitting models or deleting sources."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "08-datasets"
TAHOE = DATA / "Tahoe-100M/2dc57900b7981cfcf5e211527169a0b006546a95"
TIDYI = DATA / "TIDYI/E-MTAB-16354"
INTEGRATION = DATA / "LungCancer_A549_Integration"
OUT = INTEGRATION / "selection-v1"


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def select_tahoe():
    audit_path = TAHOE / "full_expression/coverage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    status = json.loads((TAHOE / "full_expression/download_status.json").read_text())
    assert status["state"] == "complete"
    assert audit["scope"] == "complete_published_expression"
    samples = pd.read_parquet(TAHOE / "metadata/sample_metadata.parquet")
    drugs = pd.read_parquet(TAHOE / "metadata/drug_metadata.parquet")
    observed = pd.read_parquet(TAHOE / "full_expression/a549_expression_coverage.parquet")
    assert observed["BARCODE_SUB_LIB_ID"].is_unique
    counts = observed.groupby(["sample", "plate"], observed=True).agg(
        observed_a549_cells=("BARCODE_SUB_LIB_ID", "size"),
        expression_a549_cells=("expression_downloaded", "sum"),
    ).reset_index()
    conditions = samples.merge(counts, on=["sample", "plate"], how="left", validate="one_to_one")
    for col in ["observed_a549_cells", "expression_a549_cells"]:
        conditions[col] = conditions[col].fillna(0).astype(int)
    # Missing observed cells means no matching observation in the complete reference.
    conditions["drug_key"] = conditions["drug"].str.strip()
    drugs = drugs.assign(drug_key=drugs["drug"].str.strip())
    conditions = conditions.merge(
        drugs[["drug_key", "canonical_smiles", "pubchem_cid"]],
        on="drug_key", how="left", validate="many_to_one",
    )
    parsed = conditions["drugname_drugconc"].map(ast.literal_eval)
    conditions["recorded_component_count"] = parsed.map(len)
    conditions["dose_value"] = parsed.map(lambda v: v[0][1] if len(v) == 1 else np.nan)
    conditions["dose_unit"] = parsed.map(lambda v: v[0][2] if len(v) == 1 else None)
    conditions["treatment_hours"] = 24
    conditions["time_source"] = "Tahoe preprint v1 study-level treatment protocol"
    conditions["source"] = "Tahoe-100M"
    conditions["cell_line_id"] = "CVCL_0023"
    conditions["source_sample_qc_scope"] = "mixed_cell_line_sample_not_A549_recomputed"
    has_expression = conditions["expression_a549_cells"].gt(0)
    is_control = conditions["drug"].eq("DMSO_TF")
    has_structure = conditions["canonical_smiles"].fillna("").astype(str).str.strip().ne("")
    valid_dose = pd.to_numeric(conditions["dose_value"], errors="coerce").map(np.isfinite)
    valid_dose &= conditions["dose_unit"].notna() & conditions["recorded_component_count"].eq(1)
    conditions["selection_status"] = "exclude_no_A549_expression"
    conditions.loc[has_expression, "selection_status"] = "hold_missing_dose_metadata"
    conditions.loc[has_expression & valid_dose, "selection_status"] = "hold_missing_structure"
    conditions.loc[has_expression & valid_dose & has_structure, "selection_status"] = "keep_response"
    conditions.loc[has_expression & is_control, "selection_status"] = "keep_control"
    conditions["selected"] = conditions["selection_status"].str.startswith("keep_")
    csv(conditions, "tahoe_condition_selection.csv")
    csv(conditions.loc[conditions["selected"]], "selected_tahoe_conditions.csv")

    controls = conditions.loc[conditions["selection_status"].eq("keep_control"), ["sample", "plate"]]
    treatments = conditions.loc[conditions["selection_status"].eq("keep_response"), ["sample", "plate"]]
    links = treatments.merge(controls.rename(columns={"sample": "control_sample"}), on="plate", how="left")
    assert links["control_sample"].notna().all()
    assert links["sample"].nunique() == len(treatments)
    links["cell_line_id"] = "CVCL_0023"
    links["relationship_type"] = "same_plate_control_candidate"
    links["individual_cell_pairing"] = "not_available"
    csv(links, "selected_tahoe_control_links.csv")

    observed = observed.merge(
        conditions[["sample", "plate", "selection_status", "selected"]],
        on=["sample", "plate"], how="left", validate="many_to_one",
    )
    assert observed["selection_status"].notna().all()
    observed.loc[~observed["expression_downloaded"], "selection_status"] = "exclude_expression_not_published"
    observed["selected"] &= observed["expression_downloaded"]
    frames = []
    for path in sorted((TAHOE / "full_expression/obs").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["BARCODE_SUB_LIB_ID"])
        frame["csr_shard"] = relative(TAHOE / "full_expression/csr" / (path.stem + ".npz"))
        frame["matrix_row"] = np.arange(len(frame), dtype=np.int32)
        frames.append(frame)
    matrix_index = pd.concat(frames, ignore_index=True)
    assert len(matrix_index) == audit["a549_expression_rows"]
    assert matrix_index["BARCODE_SUB_LIB_ID"].is_unique
    selected = observed.loc[observed["selected"]].drop(columns=["source_row_group"])
    selected = selected.merge(matrix_index, on="BARCODE_SUB_LIB_ID", how="left", validate="one_to_one")
    assert selected["matrix_row"].notna().all()
    selected["matrix_row"] = selected["matrix_row"].astype(np.int32)
    selected.to_parquet(OUT / "selected_tahoe_cells.parquet", index=False)
    observed.loc[~observed["selected"]].to_parquet(OUT / "held_or_excluded_tahoe_cells.parquet", index=False)
    lung = pd.read_parquet(TAHOE / "metadata/cell_line_metadata.parquet")
    lung = lung.loc[lung["Organ"].astype(str).str.casefold().eq("lung"),
                    ["cell_name", "Cell_ID_Cellosaur", "Organ"]].drop_duplicates()
    lung["selection_status"] = "retain_source_for_lung_expansion"
    csv(lung, "lung_backgrounds_to_preserve.csv")
    return {
        "source_sample_count": len(conditions), "selected_conditions": int(conditions["selected"].sum()),
        "condition_status_counts": conditions["selection_status"].value_counts().to_dict(),
        "published_a549_cells": audit["a549_expression_rows"], "selected_a549_cells": len(selected),
        "cell_status_counts": observed["selection_status"].value_counts().to_dict(),
        "preserved_lung_metadata_backgrounds": len(lung), "control_links": len(links),
    }


def select_tidyi():
    author_path = TIDYI / "audit_sources/A549_author_collection_counts.npz"
    meta = pd.read_csv(TIDYI / "audit_sources/human_ss3_metadata_selected.csv", index_col=0)
    records = meta.reset_index().rename(columns={meta.index.name or "index": "cell_id"})
    records["source_record"] = "human_ss3.h5ad"
    records["selection_status"] = "exclude_non_lung_from_current_samples"
    records.loc[records["Cell_line"].isna(), "selection_status"] = "hold_unconfirmed_cell_identity"
    records.loc[records["Cell_line"].eq("A549"), "selection_status"] = "keep_A549_state"
    old_meta = pd.read_csv(TIDYI / "SS3_Human_A549_metadata.csv", index_col=0)
    with np.load(author_path, allow_pickle=False) as author, np.load(TIDYI / "A549_expression_audited.npz", allow_pickle=False) as old:
        ids = author["cell_ids"]
        assert len(ids) == len(np.unique(ids)) == 743
        old_ids = old["cell_ids"][old["eligible"]]
        row_index = pd.Index(ids).get_indexer(old_ids)
        gene_index = pd.Index(author["gene_ids"]).get_indexer(old["gene_ids"])
        assert (row_index >= 0).all() and (gene_index >= 0).all()
        assert np.array_equal(old["expression"][old["eligible"]], author["expression"][np.ix_(row_index, gene_index)])
        assert np.array_equal(old_meta.loc[old_ids, "cell_state"].to_numpy(), author["states"][row_index])
        selected = meta.loc[ids].copy()
        assert selected["Cell_line"].eq("A549").all()
        selected.insert(0, "cell_id", ids)
        selected["matrix_path"] = relative(author_path)
        selected["matrix_row"] = np.arange(len(ids))
        selected["selection_status"] = "keep_A549_state"
        csv(selected, "selected_tidyi_cells.csv")
        legacy = old_meta.reset_index().rename(columns={old_meta.index.name or "index": "cell_id"})
        legacy["source_record"] = "official_A549_CSV"
        legacy["selection_status"] = "hold_unconfirmed_cell_identity"
        legacy.loc[legacy["cell_id"].isin(old_ids), "selection_status"] = "exclude_duplicate_cell_record"
        cols = ["cell_id", "Cell_line", "cell_state", "Plate", "Well", "source_record", "selection_status"]
        ledger = pd.concat([records[cols], legacy[cols]], ignore_index=True)
        csv(ledger, "tidyi_record_selection.csv")
    return {"selected_cells": len(selected), "original_gene_count": len(gene_index),
            "original_label_counts": selected["cell_state"].value_counts().to_dict(),
            "record_status_counts": ledger["selection_status"].value_counts().to_dict(),
            "duplicate_count_values_and_labels_checked": len(old_ids)}


def select_endpoints():
    base = DATA / "GSE161218/GSE161218"
    first = pd.read_csv(base / "annexin_v_fig_s2c_long.csv").assign(source_block="fig_s2c_lestaurtinib")
    second = pd.read_csv(base / "annexin_v_fig_s3de_midostaurin_long.csv")
    records = pd.concat([first, second], ignore_index=True)
    records["selection_status"] = np.where(records["annexin_v_positive_percent"].notna(),
                                            "keep_measured_lung_endpoint", "hold_missing_endpoint_value")
    records["use_scope"] = np.where(records["cell_line"].eq("A549"), "A549", "other_lung_extension")
    records["label_unit"] = "percent"
    records["expression_pairing_status"] = "not_established"
    csv(records, "endpoint_selection.csv")
    csv(records.loc[records["selection_status"].str.startswith("keep_")], "selected_endpoints.csv")
    return {"source_positions": len(records), "retained_numeric_values": int(records["annexin_v_positive_percent"].notna().sum()),
            "missing_values_preserved": int(records["annexin_v_positive_percent"].isna().sum()),
            "lung_backgrounds": sorted(records["cell_line"].unique().tolist())}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    print("Selecting Tahoe conditions and indexed cells", flush=True)
    tahoe = select_tahoe()
    print("Selecting TIDYI unique A549 records", flush=True)
    tidyi = select_tidyi()
    endpoints = select_endpoints()
    sources = pd.DataFrame([
        ["Tahoe source shards", "keep", "Preserve other lung backgrounds and existing complete audit dependencies"],
        ["Tahoe raw_a549/csr/obs", "keep", "Paired representations required by current checkpoints"],
        ["TIDYI classifier_data.tar.gz", "keep", "Complete recovery source and additional human-platform data"],
        ["TIDYI human_ss3.h5ad extracted copy", "cleanup_candidate", "Redundant recoverable extraction; rehash archive/member before removal"],
        ["TIDYI mouse archive members", "exclude_from_human_lung_inputs", "Remain inside intact recovery archive"],
        ["TIDYI human_10x and parameter table", "hold_for_metadata_review", "Not yet curated; parameters are not assumed to be Annexin labels"],
        ["Macaya H1792 RNA and lung endpoints", "keep", "Relevant lung expression and measured endpoints; no fabricated pairing"],
        ["Hu2025 source documents", "keep_reference_only", "Relevant sequential design; not a numeric training matrix"],
    ], columns=["object", "disposition", "reason"])
    csv(sources, "source_file_policy.csv")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "version": "selection-v1",
               "scope": "A549 ready subset; preserve other lung expansion sources",
               "not_performed": ["model_training", "algorithm_evaluation", "train_test_split", "normalization", "gene_feature_selection"],
               "tahoe": tahoe, "tidyi": tidyi, "endpoints": endpoints,
               "negative_or_weak_drug_response_is_not_an_exclusion_reason": True,
               "no_arbitrary_low_rna_or_low_condition_cell_count_filter": True,
               "physical_cleanup": "Recorded separately in file_cleanup_report.json"}
    (OUT / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_records = [{"path": relative(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
                      for p in sorted(OUT.iterdir()) if p.is_file()]
    (OUT / "output_manifest.json").write_text(json.dumps(output_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
