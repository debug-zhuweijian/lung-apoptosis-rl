"""Append SJB published means and reviewed material inputs; never generate RNA."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from build_endpoint_alignment_v1 import EndpointBuilder, ROOT, write_alignment
from build_endpoint_condition_inputs import main as build_inputs, reviewed_additions
from lung.data.drug_identity import describe_structure
from lung.data.endpoint_alignment import aggregate_units, file_hash, load_alignment

BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1"
SJB = BASE / "source_reviews/state_function_round9/sjb"
OLD_REVIEW = BASE / "source_reviews/chemical_round3/revision-2"
SOURCE_ID = "Wu2025_SJB2_043"
COMPOUND_ID = SOURCE_ID + ":SJB2-043"
PROTOCOL_ID = SOURCE_ID + ":Fig4A_Annexin_parent_gate"
MEANS_FILE = "Fig4A_digitized_apoptosis_means.csv"


def relative(path):
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require_empty(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Use an empty new output directory: " + str(path))


def frozen_snapshot(previous_alignment, previous_inputs):
    return {relative(path): file_hash(path)
            for directory in [previous_alignment, previous_inputs, OLD_REVIEW.parent]
            for path in sorted(directory.rglob("*")) if path.is_file()}


def validate_mean_rows(frame):
    """Reject changed units/timing, duplicate panels and fabricated repeats."""
    if len(frame) != 4 or frame.condition_id.duplicated().any():
        raise ValueError("SJB intake requires four unique published means")
    rows = frame.sort_values("dose_uM", kind="stable").reset_index(drop=True)
    if rows.dose_uM.tolist() != [0, .625, 2.5, 10]:
        raise ValueError("SJB dose set differs from the primary figure")
    checks = {
        "cell_line": "A549", "duration_h": 24, "unit": "percent",
        "duration_evidence": "Fig4_caption_direct_not_borrowed_from_RNA",
        "value_origin": "digitized_published_bar_mean", "author_reported_n": 9,
        "replicate_kind": "not_specified", "uncertainty_type": "pixel_readout_range_not_SD_or_CI",
        "endpoint": "author_reported_Annexin_PI_apoptosis_rate",
        "denominator": "events_in_displayed_Annexin_PI_parent_gate;upstream_gate_not_reported",
    }
    for field, expected in checks.items():
        if not rows[field].eq(expected).all():
            raise ValueError("SJB source field differs: " + field)
    numbers = rows[["value", "readout_lower", "readout_upper"]].to_numpy(dtype=float)
    if (not np.isfinite(numbers).all() or (numbers < 0).any() or (numbers > 100).any()
            or not rows.readout_lower.le(rows.value).all() or not rows.value.le(rows.readout_upper).all()):
        raise ValueError("SJB percentage or pixel-readout range is invalid")
    return rows


def read_sjb_means():
    inventory_path = SJB / "local_file_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    evidence = [inventory_path]
    for entry in inventory:
        path = SJB / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("SJB reviewed source changed: " + entry["file"])
        evidence.append(path)
    if MEANS_FILE not in {r["file"] for r in inventory}:
        raise ValueError("SJB means are not bound to the reviewed inventory")
    return validate_mean_rows(pd.read_csv(SJB / MEANS_FILE)), evidence


def add_sjb(builder, rows, evidence):
    if PROTOCOL_ID in builder.protocols or any(
            r["source_id"] == SOURCE_ID for r in builder.conditions.values()):
        raise ValueError("SJB source already present; do not duplicate published means")
    for path in evidence:
        builder.source(path)
    builder.source(Path(__file__).resolve())
    builder.protocols[PROTOCOL_ID] = {
        "source": "https://doi.org/10.3390/cimb47030155",
        "endpoint": "Annexin_positive_fraction",
        "denominator": "author analyzed Annexin/PI parent-gate events; not seeded-cell count",
        "sampling_gate": "Fig4A Annexin horizontal and PI vertical; upstream gates and event counts not reported",
        "method_evidence": "Methods2.8 and Fig4 caption independently establish doses and 24h",
        "numeric_basis": "digitized published bar mean, not displayed quadrant value or raw replicate",
        "replicate_type": "published_mean_not_raw_replicate",
        "author_reported_n": 9, "author_replicate_type": "unknown",
        "author_error_bar": "SD; numeric SD not recovered",
        "readout_interval": "pixel readout range; not SD, SE or statistical CI",
        "culture": "ATCC A549; RPMI1640;10%FBS;1%penicillin-streptomycin;37C;5%CO2",
        "floating_cell_recovery": "not_reported", "vehicle": "Annexin0uM; solvent not explicit",
        "rna_expression_samples_read": 0, "same_sample_RNA_endpoint_pairing": False,
        "material_scope": "paper names MCE; current HY-15757 nominal structure; historical catalog/lot unknown",
        "method_complete_for_intake": True,
    }
    for row_number, row in enumerate(rows.to_dict("records"), 2):
        dose = float(row["dose_uM"])
        condition = builder.condition(
            SOURCE_ID, PROTOCOL_ID, "Fig4A_24h_published_mean", "A549",
            "A549_parental_unmeasured_reference", [("SJB2-043", dose)] if dose else [],
            treatment_hours=24.0, feature_status="not_generated",
            chemical_review_note="MCE current nominal structure; paper historical catalog/lot unknown",
            response_domain="assay_defined_Annexin_parent_gate",
            rna_expression_samples_read=0,
        )
        builder.observation(
            condition, "Annexin_positive_fraction", float(row["value"]) / 100,
            "published_mean", SJB / MEANS_FILE,
            f"csv_row={row_number};{row['condition_id']};Fig4A_published_bar_mean",
            replicate_kind="published_mean_not_raw_replicate", author_reported_n=9,
            author_replicate_type="unknown", numeric_basis="digitized_published_mean",
            original_value_percent=float(row["value"]),
            pixel_readout_lower_fraction=float(row["readout_lower"]) / 100,
            pixel_readout_upper_fraction=float(row["readout_upper"]) / 100,
            readout_interval_kind="pixel_readout_range_not_SD_or_CI",
            author_reported_error_type="SD", author_reported_sd_fraction=None,
            primary_figure=relative(SJB / row["source_image"]),
        )


def builder_from_previous(previous):
    builder = EndpointBuilder()
    builder.conditions = {r["condition_id"]: r for r in previous["conditions"].to_dict("records")}
    builder.interventions = previous["interventions"].to_dict("records")
    builder.observations = previous["observations"].to_dict("records")
    builder.protocols = previous["protocols"].copy()
    builder.sources = {r["file"]: r for r in previous["manifest"]["source_files"]}
    return builder


def build_chemical_review(output, previous_review=OLD_REVIEW):
    require_empty(output)
    reviewed_additions(previous_review)  # verifies the complete inherited hash chain
    old = pd.read_csv(previous_review / "chemical_candidates.csv")
    nominal = json.loads((SJB / "material_smiles_validation.json").read_text(encoding="utf-8"))
    structure = describe_structure(nominal["source_smiles"])
    molecule = Chem.MolFromSmiles(structure["canonical_full_smiles"])
    if (structure["rdkit_inchikey"] != "CMYQQADDUUDCCA-UHFFFAOYSA-N"
            or rdMolDescriptors.CalcMolFormula(molecule) != "C17H9NO3"
            or structure["fragment_count"] != 1 or structure["formal_charge"] != 0
            or structure["unspecified_stereo_count"] != 0):
        raise ValueError("SJB official nominal material identity differs from reviewed evidence")
    if COMPOUND_ID in set(old.source_id):
        raise ValueError("SJB material already included")
    row = {
        "source_id": COMPOUND_ID, "source_compound_name": "SJB2-043",
        "molecule_chembl_id": None,
        "status": "article_MCE_plus_current_official_HY15757_nominal_identity;historical_catalog_lot_unknown",
        "primary_connectivity_verified": True, "primary_reported_representation_verified": True,
        "primary_structure_identity_verified": True, "eligible_for_structural_features": True,
        "connectivity_only_eligible": True, "formal_training_ready": False,
        **{k: structure[k] for k in ["canonical_full_smiles", "identity_id", "parent_id",
                                     "rdkit_inchikey", "unspecified_stereo_count", "defined_stereo_count"]},
        "molecular_formula": "C17H9NO3",
    }
    merged = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "chemical_candidates.csv", index=False)
    review = {
        "new_source_id": COMPOUND_ID, "inherited_candidates": len(old), "candidates": len(merged),
        "inherited_held_rows_retained": True, "manufacturer": "MedChemExpress",
        "current_official_catalog": "HY-15757", "current_official_CAS": "63388-44-3",
        "article_catalog_number": None, "article_lot": None,
        "historical_material_lot_verified": False, "formal_training_ready": False,
        "nominal_identity_evidence": "paper2.1 names MCE; current official product/data sheet and SMILES",
        "source_scope": "chemical features only; no observed initial RNA or paired RNA endpoint",
    }
    write_json(output / "review.json", review)
    (output / "README.md").write_text(
        "# SJB additional chemical review v1\n\n"
        "完整继承 condition-inputs-v5 使用的 revision-2 三个候选（含两个暂缓），新增一个 SJB 官方名义结构。\n\n"
        "论文明确 MCE 厂家，当前产品 HY-15757/CAS63388-44-3；原论文历史货号、lot、纯度未知。\n\n"
        "旧审查 manifest 递归绑定，未改旧候选状态；只提供化学输入，不生成表达，不表示正式训练就绪。\n",
        encoding="utf-8",
    )
    inputs = [previous_review / "manifest.json", SJB / "PMC11941171.xml",
              SJB / "cimb-47-00155.pdf", SJB / "SJB2-043-DataSheet-MedChemExpress.pdf",
              SJB / "MCE_web_tool_evidence.md", SJB / "material_smiles_validation.json",
              SJB / "material_identity_readback.md", SJB / "nominal_material_condition_links.json",
              ROOT / "src/lung/data/drug_identity.py", Path(__file__).resolve()]
    outputs = [output / n for n in ["chemical_candidates.csv", "review.json", "README.md"]]
    write_json(output / "manifest.json", {
        "schema_version": 1, "version": "round9-sjb-chemical-review-v1",
        "inputs_sha256": {relative(p): file_hash(p) for p in inputs},
        "outputs_sha256": {p.name: file_hash(p) for p in outputs},
        "counts": {"records": len(merged), "strict_feature_eligible": int(merged.eligible_for_structural_features.sum())},
    })
    reviewed_additions(output)


def strict_annexin_count(requests, observations):
    available = observations[observations.endpoint.eq("Annexin_positive_fraction")
                             & observations.measurement_status.eq("observed")
                             & observations.exclusion_reason.fillna("").eq("")]
    selected = requests[requests.condition_id.isin(available.condition_id)
                        & requests.cell_line.eq("A549") & requests.component_count.eq(1)
                        & requests.treatment_hours.eq(24) & requests.chemical_inputs_available
                        & requests.input_contract_blockers.fillna("").eq("")]
    return len(selected)


def validate_outputs(alignment, inputs, previous, previous_inputs, chemical_review):
    actual = load_alignment(alignment, ROOT)
    old_requests = pd.read_parquet(previous_inputs / "condition_requests.parquet")
    requests = pd.read_parquet(inputs / "condition_requests.parquet")
    for name, keys in [("conditions", ["condition_id"]), ("interventions", ["condition_id", "compound_id"]),
                       ("observations", ["observation_id"])]:
        old = previous[name]
        retained = actual[name][actual[name].condition_id.isin(previous["conditions"].condition_id)]
        pd.testing.assert_frame_equal(old.sort_values(keys).reset_index(drop=True),
                                      retained[old.columns].sort_values(keys).reset_index(drop=True), check_dtype=False)
    for key, protocol in previous["protocols"].items():
        assert actual["protocols"][key] == protocol
    new_conditions = actual["conditions"][actual["conditions"].source_id.eq(SOURCE_ID)]
    new_observations = actual["observations"][actual["observations"].condition_id.isin(new_conditions.condition_id)]
    means, _ = read_sjb_means()
    assert len(new_conditions) == len(new_observations) == 4
    np.testing.assert_allclose(sorted(new_observations.value), sorted(means.value / 100), rtol=0, atol=1e-15)
    assert new_observations.replicate_kind.eq("published_mean_not_raw_replicate").all()
    assert new_observations.author_replicate_type.eq("unknown").all()
    assert new_observations.author_reported_n.eq(9).all()
    assert new_conditions.initial_expression_sample_id.isna().all()
    assert not new_conditions.same_sample_expression_endpoint.any()
    units = aggregate_units(new_conditions, new_observations)
    assert units.measured_replicates.eq(0).all() and units.available_numeric_records.eq(1).all()
    assert units.sd_fraction.isna().all()
    new_parts = actual["interventions"][actual["interventions"].condition_id.isin(new_conditions.condition_id)]
    assert sorted(new_parts.dose_value) == [.625, 2.5, 10] and new_parts.dose_unit.eq("uM").all()
    retained_requests = requests[requests.condition_id.isin(old_requests.condition_id)]
    pd.testing.assert_frame_equal(old_requests.sort_values("condition_id").reset_index(drop=True),
                                  retained_requests[old_requests.columns].sort_values("condition_id").reset_index(drop=True),
                                  check_dtype=False)
    assert requests.expression_feature_status.eq("not_generated").all()
    assert requests.generator_checkpoint.isna().all() and not requests.formal_training_ready.any()
    compounds = pd.read_parquet(inputs / "compounds.parquet")
    old_compounds = pd.read_parquet(previous_inputs / "compounds.parquet")
    retained = compounds[compounds.compound_id.isin(old_compounds.compound_id)]
    columns = [c for c in old_compounds.columns if c != "fingerprint_row"]
    pd.testing.assert_frame_equal(old_compounds[columns].sort_values("compound_id").reset_index(drop=True),
                                  retained[columns].sort_values("compound_id").reset_index(drop=True), check_dtype=False)
    fingerprints = np.load(inputs / "fingerprints.npy", allow_pickle=False)
    old_fingerprints = np.load(previous_inputs / "fingerprints.npy", allow_pickle=False)
    assert fingerprints.shape == (len(compounds), 2048) and fingerprints.dtype == np.uint8
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    for row in compounds.to_dict("records"):
        expected = np.zeros(2048, dtype=np.uint8)
        if row["structure_eligible"]:
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(row["canonical_full_smiles"])), expected)
        np.testing.assert_array_equal(fingerprints[row["fingerprint_row"]], expected)
    indices = compounds.set_index("compound_id").fingerprint_row
    for row in old_compounds.to_dict("records"):
        np.testing.assert_array_equal(old_fingerprints[row["fingerprint_row"]], fingerprints[indices[row["compound_id"]]])
    registry, inherited_paths = reviewed_additions(chemical_review)
    assert len(registry) == 4 and registry[COMPOUND_ID]["eligible"]
    assert not registry["Huang2019_6k:6k"]["eligible"] and not registry["C9MD00494G_8h:8h"]["eligible"]
    assert (OLD_REVIEW / "manifest.json").resolve() in inherited_paths
    assert (OLD_REVIEW.parent / "manifest.json").resolve() in inherited_paths
    before = strict_annexin_count(old_requests, previous["observations"])
    after = strict_annexin_count(requests, actual["observations"])
    assert after - before == 3
    return {"conditions": len(actual["conditions"]), "observations": len(actual["observations"]),
            "added_conditions": 4, "added_mean_records": 4, "added_nonzero_interventions": 3,
            "compounds": len(compounds), "structure_resolved_compounds": int(compounds.structure_eligible.sum()),
            "conditions_with_chemical_inputs": int(requests.chemical_inputs_available.sum()),
            "strict_A549_single_24h_Annexin_before": before, "strict_A549_single_24h_Annexin_after": after,
            "all_previous_records_and_protocols_preserved": True,
            "all_fingerprints_recomputed_bitwise": True, "previous_fingerprints_preserved_by_compound": True,
            "inherited_chemical_reviews_recursively_verified": True,
            "new_source_independent_replicates_recovered": 0, "RNA_samples_or_features_generated": 0,
            "formal_training_ready": False}


def run(alignment, inputs, chemical_review, previous_alignment=BASE / "curated-20261010-r4",
        previous_inputs=BASE / "condition-inputs-v5"):
    for output in [alignment, inputs, chemical_review]:
        require_empty(output)
    before_hashes = frozen_snapshot(previous_alignment, previous_inputs)
    previous = load_alignment(previous_alignment, ROOT)
    old_meta = json.loads((previous_inputs / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in old_meta["files"].items():
        if file_hash(previous_inputs / name) != digest:
            raise ValueError("Previous condition inputs changed: " + name)
    for name, digest in old_meta["inputs_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Previous condition input evidence changed: " + name)
    assert old_meta["inputs_sha256"][relative(previous_alignment / "manifest.json")] == file_hash(previous_alignment / "manifest.json")
    assert old_meta["inputs_sha256"][relative(OLD_REVIEW / "manifest.json")] == file_hash(OLD_REVIEW / "manifest.json")
    means, evidence = read_sjb_means()
    builder = builder_from_previous(previous)
    builder.source(previous_alignment / "manifest.json")
    add_sjb(builder, means, evidence)
    build_chemical_review(chemical_review)
    write_alignment(builder, alignment)
    build_inputs(inputs, BASE / "chemical-links-v2", alignment, chemical_review)
    result = validate_outputs(alignment, inputs, previous, previous_inputs, chemical_review)
    after_hashes = frozen_snapshot(previous_alignment, previous_inputs)
    assert before_hashes == after_hashes, "Previous frozen package files changed during extension"
    write_json(inputs / "prior_packages_before_after.json", {
        "all_unchanged": True, "before_sha256": before_hashes, "after_sha256": after_hashes})
    result["prior_frozen_files_hash_verified_unchanged"] = len(before_hashes)
    write_json(inputs / "round9_validation.json", result)
    for folder in [alignment, inputs]:
        (folder / "README.md").write_text(
            "# SJB round9 additive input package\n\n"
            "r4/v5 保持不变。新增 SJB 四个发表均值条件，其中三个非零单药24h；无原始重复、无RNA。\n\n"
            "Annexin 分母保持作者分析母门、上游gate未知；像素范围不是SD/CI，n9重复类型未知。\n\n"
            "材料为论文MCE与当前官方HY-15757名义结构，历史lot/catalog未知。精确验证见 condition-inputs-v6/round9_validation.json。\n",
            encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-output", type=Path, default=BASE / "curated-20261010-r5")
    parser.add_argument("--inputs-output", type=Path, default=BASE / "condition-inputs-v6")
    parser.add_argument("--chemical-review-output", type=Path, default=SJB / "chemical-review-v1")
    args = parser.parse_args()
    run(args.alignment_output.resolve(), args.inputs_output.resolve(), args.chemical_review_output.resolve())
