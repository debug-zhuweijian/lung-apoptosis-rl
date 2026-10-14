"""Source-preserving bulk reference / caspase challenge metadata interface.

This does not implement or qualify a P/G/Q model. Metadata serving never parses
functional outcomes. RNA and published means remain different observation units.
"""
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

SCHEMA = "martin-parallel-reference-caspase-bridge-v1"
BACKGROUND_ORDER = ("HCC2935", "HCC827", "NCI-H1975", "PC9")
RNA_BACKGROUND_NAMES = {"NCI-H1975": "H1975"}
DOMAIN_IDS = ("Martin2022_S2F", "Martin2022_S2M")
SCALE = "published_relative_caspase_3_7_AUC_72h_plot_scale"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def build(source, output, *, project_root):
    """Explicit data curation operation; it reads published means to separate Y.

    Roles depend only on the four background names, never on numerical outcomes.
    They are retrospective source-condition holds, not independent experiments.
    """
    source, output, root = Path(source).resolve(), Path(output).resolve(), Path(project_root).resolve()
    if output.exists():
        raise FileExistsError("IMMUTABLE_BRIDGE_VERSION_EXISTS")
    rna, functional = _read(source / "rna_manifest.json"), _read(source / "function/manifest.json")
    chemistry = _read(source / "chemistry/manifest.json")
    chemical_items = _read(source / "chemistry" / chemistry["items_file"])
    chemicals = {c["source_drug_name"]: c for c in chemical_items}
    if (len(chemicals) != 4 or chemistry["source_function_sha256"] != sha(source / "function/published_means.csv")
            or chemistry["experimental_stock_form_certified"] is not False):
        raise ValueError("NOMINAL_CHEMISTRY_BINDING_CHANGED")
    if (rna["rna_sample_count"] != 60 or rna["baseline_samples"] != 12
            or functional["counts"]["single_active_bars"] != 20):
        raise ValueError("SOURCE_COUNTS_CHANGED")
    with (source / "rna/rna_samples.csv").open(encoding="utf-8", newline="") as stream:
        samples = list(csv.DictReader(stream))
    with (source / "function/published_means.csv").open(encoding="utf-8", newline="") as stream:
        bars = list(csv.DictReader(stream))
    references = []
    for background in BACKGROUND_ORDER:
        source_name = RNA_BACKGROUND_NAMES.get(background, background)
        selected = [s for s in samples if s["cell_line"] == source_name and s["state"] == "parental_DMSO"]
        if (len(selected) != 3 or {s["biological_rna_replicate"] for s in selected} != {"1", "2", "3"}
                or any(s["same_well_function_pairing"] != "not_declared" for s in selected)):
            raise ValueError("PARENTAL_PARALLEL_RNA_REFERENCE_REQUIRED")
        references.append({"state_reference_id": "GSE193258:" + background + ":parental_DMSO",
            "cell_background": background, "reference_state": "parental_DMSO_parallel_reference",
            "RNA_source_cell_line": source_name,
            "background_name_binding": "GEO_GSM_H1975_titles_and_treatment_protocol_NCI-H1975;_paper_Cell_Lines" if source_name != background else "exact_source_name",
            "matrix_columns": [s["matrix_column"] for s in selected],
            "column_indices": [int(s["column_index"]) for s in selected], "gsm_ids": [s["gsm"] for s in selected],
            "measurement_level": "bulk_RNA_biological_triplicates_not_single_cells",
            "parental_drug_free_culture_days": 21, "vehicle_hours": 24,
            "bridge_status": "same_source_parallel_background_explicit_assumption_unvalidated",
            "same_well_function_pairing": False, "same_batch_function_pairing_verified": False,
            "functional_initial_state_equality_verified": False, "functional_Y_per_RNA_replicate": False})
    by_background = {r["cell_background"]: r for r in references}
    candidates, observations = [], []
    for bar in bars:
        if bar["candidate_type"] != "single":
            continue
        components = json.loads(bar["components_json"])
        if (len(components) != 1 or components[0]["unit"] != "nM"
                or float(bar["exposure_h"]) != 72 or float(bar["observation_window_h"]) != 72):
            raise ValueError("COMPLETE_SINGLE_CANDIDATE_REQUIRED")
        cid = bar["observation_id"]
        ref = by_background[bar["cell_background"]]
        chemical = chemicals[components[0]["drug"]]
        dose_uM = float(components[0]["concentration"]) / 1000
        candidates.append({"candidate_id": cid, "observation_id": cid,
            "measurement_domain_id": bar["measurement_domain_id"], "decision_block_id": bar["decision_block_id"],
            "state_reference_id": ref["state_reference_id"], "cell_background": bar["cell_background"],
            "functional_state": "author_parental_background", "components": components,
            "chemical": deepcopy(chemical), "nominal_dose_uM": dose_uM,
            "dose_basis": chemistry["dose_basis"],
            "actual_protocol": {"reported_exposure_h": 72, "observation_window_h": 72,
                "withdrawal_before_endpoint": "not_described", "prior_treatment": "parental",
                "nominal_seeded_cells_per_well": 4000, "plate_format": "96_well",
                "attachment_before_challenge": "overnight_duration_not_numeric"},
            "response_request": {"state_reference_id": ref["state_reference_id"], "components": components,
                "chemical_id": chemical["chemical_id"], "nominal_dose_uM": dose_uM,
                "target_observation_h": 72, "reported_exposure_h": 72, "withdrawal_before_endpoint": "not_described",
                "gene_axis": "source_gene_symbols", "RNA_units": "log2(TPM+0.01)", "measurement_level": "bulk",
                "preprocessing_model_identity": None, "producer_status": "pending_native_bulk_LUAD_state_protocol_response",
                "standard_Tahoe_24h_descriptor_eligible": False}})
        values = [float(bar[k]) for k in ("pixel_reading_lower", "published_mean_digitized", "pixel_reading_upper")]
        if not np.isfinite(values).all() or not 0 < values[0] < values[1] < values[2]:
            raise ValueError("INVALID_DIGITIZED_FUNCTION_VALUE")
        observations.append({"observation_id": cid, "candidate_id": cid, "value": values[1],
            "reading_lower": values[0], "reading_upper": values[2], "value_scale": SCALE,
            "label_kind": "digitized_published_group_mean_not_raw_replicate", "weight": 1,
            "uncertainty_type": "finite_pixel_reading_bound_not_CI_not_SD",
            "biological_function_replicate_ids": [], "source_panel": bar["panel"]})
    if len(candidates) != 20 or len({c["candidate_id"] for c in candidates}) != 20:
        raise ValueError("CORE_OBSERVATION_IDENTITIES_CHANGED")
    domains = [{"measurement_domain_id": domain, "value_scale": SCALE,
        "allowed_state_references": [r["state_reference_id"] for r in references],
        "endpoint_definition": {"signal": "distinct_green_caspase_3_7_spots/confluence",
            "normalization": functional["normalization"], "summary": "source_reported_72h_AUC",
            "integration_algorithm": "unknown", "full_AUC_units": "not_fully_specified", "sampling_interval_h": 4},
        "absolute_Annexin_fraction_eligible": False, "separate_measurement_head_required": True}
        for domain in DOMAIN_IDS]
    blocks = []
    for block in functional["blocks"]:
        ids = block["core_single_observation_ids"]
        rows = [c for c in candidates if c["candidate_id"] in ids]
        if len(rows) != len(ids):
            raise ValueError("CORE_BLOCK_INCOMPLETE")
        blocks.append({"decision_block_id": block["decision_block_id"],
            "measurement_domain_id": block["measurement_domain_id"], "state_reference_id": rows[0]["state_reference_id"],
            "cell_background": block["cell_background"], "observation_window_h": 72, "value_scale": SCALE,
            "candidate_ids": ids, "candidate_scope": "all_reported_active_monotherapies_in_this_panel",
            "source_vehicle_and_combination_ids_retained_as_auxiliary": block["vehicle_observation_ids"] + block["combination_auxiliary_observation_ids"]})
    folds = [{"fold_id": "leave_background_out:" + bg, "held_out_background": bg,
              "fit_backgrounds": [x for x in BACKGROUND_ORDER if x != bg],
              "fit_candidate_ids": [c["candidate_id"] for c in candidates if c["cell_background"] != bg],
              "final_test_candidate_ids": [c["candidate_id"] for c in candidates if c["cell_background"] == bg],
              "selection_candidate_ids": [], "reward_qualification_status": "pending_no_selection_role"}
             for bg in BACKGROUND_ORDER]
    metadata = {"schema": SCHEMA, "references": references, "candidates": candidates,
                "measurement_domains": domains, "decision_blocks": blocks, "folds": folds,
                "status": "metadata_and_real_source_values_connected_models_not_executed",
                "evaluation_is_retrospective": True, "source_observation_counts": functional["counts"],
                "primary_scope": "cross_LUAD_parental_background_monotherapy_caspase_extension_not_A549_validation",
                "G_Q_actual_execution": False, "old_fraction_G_compatible": False}
    source_paths = [source / "rna_manifest.json", source / "function/manifest.json", source / "function/published_means.csv"]
    source_paths += [source / "chemistry/manifest.json"]
    for name, item in chemistry["files"].items():
        path = source / "chemistry" / name
        if sha(path) != item["sha256"]:
            raise ValueError("CHEMICAL_ASSET_CHANGED")
        source_paths.append(path)
    source_paths += [source / name for name in rna["files"]]
    for item in rna["source_files"].values():
        path = source / "raw" / item["url"].rsplit("/", 1)[1]
        if sha(path) != item["sha256"]:
            raise ValueError("OFFICIAL_RNA_SOURCE_CHANGED")
        source_paths.append(path)
    evidence = root / "experiments/20261011-data-alignment-round16/state_function_new_sources"
    source_paths += [evidence / name for name in functional["source_hashes"]]
    source_paths += [root / name for name in ("src/lung/data/martin_state_function.py",
        "scripts/build_martin_state_function.py", "scripts/materialize_martin2022_rna.py",
        "scripts/digitize_martin2022_function.py", "scripts/build_martin2022_chemistry.py")]
    source_bindings = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in source_paths}
    # Verify source-declared hashes before publishing the separate bridge version.
    for name, item in rna["files"].items():
        if sha(source / name) != item["sha256"]:
            raise ValueError("RNA_ASSET_CHANGED")
    for name, expected in functional["source_hashes"].items():
        if sha(evidence / name) != expected:
            raise ValueError("FUNCTION_SOURCE_CHANGED")
    output.mkdir(parents=True)
    _save(output / "metadata.json", metadata)
    _save(output / "observations.json", observations)
    manifest = {"schema": SCHEMA, "source_bindings": source_bindings,
                "files": {name: sha(output / name) for name in ("metadata.json", "observations.json")},
                "RNA_source_directory": str(source.relative_to(root)).replace("\\", "/"),
                "chemistry_directory": str((source / "chemistry").relative_to(root)).replace("\\", "/"),
                "counts": {"references": 4, "RNA_reference_samples": 12, "core_observations": 20,
                           "measurement_domains": 2, "decision_blocks": 8, "folds": 4}}
    _save(output / "manifest.json", manifest)
    return manifest


class MartinStateFunctionInputs:
    def __init__(self, directory, *, project_root):
        self.directory, self.root = Path(directory).resolve(), Path(project_root).resolve()
        self.manifest = _read(self.directory / "manifest.json")
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("UNSUPPORTED_BRIDGE_SCHEMA")
        for base, items in ((self.root, self.manifest["source_bindings"]), (self.directory, self.manifest["files"])):
            for name, expected in items.items():
                path = (base / name).resolve()
                if not path.is_relative_to(base) or sha(path) != expected:
                    raise ValueError("SOURCE_OR_BRIDGE_CONTENT_CHANGED")
        self.metadata = _read(self.directory / "metadata.json")
        self.candidates = {c["candidate_id"]: c for c in self.metadata["candidates"]}
        self.references = {r["state_reference_id"]: r for r in self.metadata["references"]}
        self.domains = {d["measurement_domain_id"]: d for d in self.metadata["measurement_domains"]}
        self.blocks = {b["decision_block_id"]: b for b in self.metadata["decision_blocks"]}
        self.folds = {f["fold_id"]: f for f in self.metadata["folds"]}
        if len(self.candidates) != 20 or len(self.references) != 4 or len(self.domains) != 2 or len(self.blocks) != 8:
            raise ValueError("BRIDGE_METADATA_COUNTS_INVALID")
        chemistry = self.root / self.manifest["chemistry_directory"]
        self.fingerprints = np.load(chemistry / "morgan2048_chiral.npy", allow_pickle=False)
        if (self.fingerprints.shape != (4, 2048) or self.fingerprints.dtype != np.uint8
                or not np.isin(self.fingerprints, (0, 1)).all()):
            raise ValueError("INVALID_NOMINAL_CHEMICAL_FINGERPRINTS")
        for c in self.candidates.values():
            chemical = c["chemical"]
            fp = self.fingerprints[chemical["fingerprint_row"]]
            if hashlib.sha256(fp.tobytes()).hexdigest() != chemical["fingerprint_sha256"]:
                raise ValueError("FINGERPRINT_ROW_IDENTITY_CHANGED")
        self.fingerprints.setflags(write=False)

    def model_input(self, candidate_id):
        packet = deepcopy(self.candidates[candidate_id])
        fingerprint = self.fingerprints[packet["chemical"]["fingerprint_row"]].copy()
        fingerprint.setflags(write=False)
        return {**packet, "chemical_fingerprint": fingerprint}

    def training_domain(self, fold_id, measurement_domain_id):
        fold, domain = self.folds[fold_id], self.domains[measurement_domain_id]
        rows = [c for c in self.candidates.values() if c["candidate_id"] in fold["fit_candidate_ids"]
                and c["measurement_domain_id"] == measurement_domain_id]
        return {**deepcopy(domain), "fit_state_references": sorted({c["state_reference_id"] for c in rows}),
                "fit_candidate_ids": [c["candidate_id"] for c in rows],
                "held_out_state_status": "predeclared_background_extrapolation_not_validated"}

    def decision_inputs(self, candidate_ids):
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("EMPTY_OR_DUPLICATE_CANDIDATES")
        rows = [self.candidates[cid] for cid in candidate_ids]
        block_ids = {r["decision_block_id"] for r in rows}
        if len(block_ids) != 1:
            raise ValueError("MIXED_MEASUREMENT_STATE_OR_TIME_DECISION_BLOCK")
        block = self.blocks[next(iter(block_ids))]
        if set(candidate_ids) != set(block["candidate_ids"]):
            raise ValueError("COMPLETE_PREDECLARED_CORE_BLOCK_REQUIRED")
        domain = self.domains[block["measurement_domain_id"]]
        if any(r["state_reference_id"] != block["state_reference_id"]
               or r["state_reference_id"] not in domain["allowed_state_references"] for r in rows):
            raise ValueError("STATE_OUTSIDE_DECLARED_DOMAIN")
        return {"decision_block": deepcopy(block), "measurement_domain": deepcopy(domain),
                "candidates": [deepcopy(r) for r in rows], "response_features_status": "pending_native_response_pipeline"}

    def reference_expression(self, state_reference_id):
        """Real source-axis bulk values; three columns are RNA repeats, not cells."""
        reference = self.references[state_reference_id]
        source = self.root / self.manifest["RNA_source_directory"]
        matrix = np.load(source / "rna/log2TPM.npy", allow_pickle=False, mmap_mode="r")
        values = np.array(matrix[:, reference["column_indices"]], copy=True)
        if values.shape != (19712, 3) or not np.isfinite(values).all():
            raise ValueError("INVALID_SOURCE_BULK_REFERENCE")
        values.setflags(write=False)
        return {"values": values, "RNA_units": "log2(TPM+0.01)", "measurement_level": "bulk",
                "state_reference": deepcopy(reference), "gene_axis_path": str(source / "rna/rna_genes.csv"),
                "common_axis_status": "comparison_only_unique_symbol_coverage_not_preprocessing_compatibility"}

    def observation_units(self, fold_id, role, measurement_domain_id):
        """Explicit label API. Final Y must only be read after predictions freeze."""
        if role not in ("fit", "final_test"):
            raise ValueError("NO_SUCH_MEASURED_ROLE")
        fold = self.folds[fold_id]
        ids = fold[role + "_candidate_ids"]
        allowed = {cid for cid in ids if self.candidates[cid]["measurement_domain_id"] == measurement_domain_id}
        if measurement_domain_id not in self.domains:
            raise ValueError("UNKNOWN_MEASUREMENT_DOMAIN")
        labels = _read(self.directory / "observations.json")
        selected = [{**r, "role": role} for r in labels if r["candidate_id"] in allowed]
        if len(selected) != len(allowed) or len({r["observation_id"] for r in selected}) != len(selected):
            raise ValueError("LABEL_IDENTITY_MISMATCH")
        return selected
