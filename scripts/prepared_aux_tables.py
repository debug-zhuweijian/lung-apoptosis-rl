"""Prepare measured endpoints and auxiliary source tables without modelling."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd


SELECTION = Path("08-datasets/LungCancer_A549_Integration/selection-v1")
MACAYA = Path("08-datasets/GSE161218/GSE161218")
TIDYI = Path("08-datasets/TIDYI/E-MTAB-16354/audit_sources")
HU = Path("08-datasets/Hu2025_Sequential")
DRUG_COLUMNS = {
    "trametinib_uM": "Trametinib",
    "lestaurtinib_uM": "Lestaurtinib",
    "midostaurin_uM": "Midostaurin",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_source(root: Path, source_dir: Path, name: str) -> Path:
    """Resolve a relative source name without escaping its directory or project."""
    root = root.resolve()
    name_path = Path(name)
    _require(not (name_path.is_absolute() or name_path.drive or name_path.root),
             "Source file name must be relative and must not contain a drive")
    base = (root / source_dir).resolve()
    target = (base / name_path).resolve()
    _require(base.is_relative_to(root) and target.is_relative_to(base),
             "Source file path escapes its allowed directory")
    return target.relative_to(root)


def _endpoint_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    source = SELECTION / "endpoint_selection.csv"
    df = pd.read_csv(root / source)
    _require(len(df) == 144, "Frozen selection-v1 requires 144 endpoint positions")
    _require(df["source_block"].nunique() == 2, "Endpoint source blocks changed")
    _require(df["time_h"].eq(24).all(), "Unexpected endpoint time")
    _require(df["annexin_v_positive_percent"].notna().sum() == 142,
             "Frozen selection-v1 requires 142 measured and two missing endpoints")
    values = df["annexin_v_positive_percent"].dropna()
    _require(values.between(0, 100).all(), "Endpoint percent outside [0, 100]")
    key_columns = ["source_block", "cell_line", "condition"]
    _require(df[key_columns].notna().all().all(), "Missing endpoint condition identity")
    _require(not df[key_columns].astype(str).apply(lambda c: c.str.contains("|", regex=False)).any().any(),
             "Endpoint identity contains reserved delimiter")
    workbook = MACAYA / "sources/41467_2023_41828_MOESM7_ESM.xlsx"
    _require((root / workbook).is_file(), "Endpoint source workbook is missing")
    condition_ids = "macaya|" + df[key_columns].astype(str).agg("|".join, axis=1)
    repeats = df["experiment_number_within_endpoint"]
    _require(repeats.isin([1, 2, 3]).all(), "Unexpected endpoint replicate number")
    measurements = df.assign(
        endpoint_condition_id=condition_ids,
        endpoint_id=condition_ids + "|exp_" + repeats.astype(int).astype(str),
        condition_original=df["condition"],
        annexin_fraction=df["annexin_v_positive_percent"] / 100,
        has_measured_endpoint=df["annexin_v_positive_percent"].notna(),
        replicate_id=repeats.astype(int),
        source_file=workbook.as_posix(), source_table=source.as_posix(),
        pairing_status="not_established", is_sequential=False,
    )
    _require(measurements["endpoint_id"].is_unique, "Duplicate source-scoped endpoint ID")
    conditions = measurements.drop_duplicates("endpoint_condition_id")
    _require(len(conditions) == 48, "Expected 48 source-scoped endpoint conditions")
    component_rows = []
    for _, row in conditions.iterrows():
        group = measurements.loc[measurements["endpoint_condition_id"].eq(row.endpoint_condition_id)]
        _require(group[list(DRUG_COLUMNS)].nunique(dropna=False).le(1).all(),
                 "Drug concentrations differ within an endpoint condition")
        active = [(column, name, row[column]) for column, name in DRUG_COLUMNS.items()
                  if pd.notna(row[column]) and row[column] > 0]
        _require(all(pd.isna(row[c]) or row[c] >= 0 for c in DRUG_COLUMNS), "Negative drug dose")
        is_control = row["condition"] == "Control"
        expected = 0 if is_control else (2 if row["condition"] == "Simultaneous combination" else 1)
        _require(len(active) == expected, "Condition has missing or unexpected drug components")
        if not is_control and expected == 1:
            _require(active[0][1] == row["condition"], "Condition and positive-dose drug disagree")
        for index, (column, drug, dose) in enumerate(active or [(None, "untreated", None)]):
            component_rows.append({
                "endpoint_condition_id": row.endpoint_condition_id,
                "component_index": index,
                "component_role": "untreated_control" if is_control else "drug",
                "source_block": row.source_block, "cell_line": row.cell_line,
                "condition_original": row["condition"], "drug_name": drug,
                "dose_value": dose, "dose_unit": "uM" if column else None,
                "source_dose_column": column, "is_control": is_control,
                "source_sheet": row.source_sheet, "source_file": workbook.as_posix(),
                "source_table": source.as_posix(), "is_sequential": False,
            })
    components = pd.DataFrame(component_rows)
    _require(len(components) == 60, "Expected 48 active components and 12 untreated rows")
    return measurements, components, [source.as_posix()], [workbook.as_posix()]


def _rna_samples(root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    source = MACAYA / "rna_sample_conditions.csv"
    validation_source = MACAYA / "counts_read_validation.csv"
    source_df = pd.read_csv(root / source)
    validations = pd.read_csv(root / validation_source)
    _require(len(source_df) == len(validations) == 9, "Expected nine bulk RNA samples")
    _require(source_df["cell_line"].eq("H1792").all() and source_df["time_h"].eq(16).all(),
             "RNA cell line or treatment time changed")
    _require(source_df["GSM"].is_unique and validations["GSM"].is_unique, "Duplicate RNA accession")
    validation_columns = {c: "validation_" + c for c in validations if c != "GSM"}
    df = source_df.merge(validations.rename(columns=validation_columns), on="GSM",
                         how="left", validate="one_to_one")
    _require(df["validation_file"].eq(df["expression_file"]).all(), "RNA validation file mismatch")
    _require(df["validation_title"].eq(df["title"]).all(), "RNA validation title mismatch")
    paths = [_safe_source(root, MACAYA / "sources", name) for name in df["expression_file"]]
    _require(all((root / p).is_file() for p in paths), "Missing original RNA counts gzip")
    _require(df["endpoint_replicate_link"].eq("not established").all(), "Unexpected RNA pairing")
    df = df.assign(
        bulk_sample_id="macaya_rna|" + df["GSM"],
        bulk_condition_id="macaya_rna|H1792|16h|" + df["condition"],
        condition_original=df["condition"],
        replicate_id=df["replicate_number_within_rna"],
        expression_path=[p.as_posix() for p in paths],
        source_table=source.as_posix(), validation_source=validation_source.as_posix(),
        pairing_status="not_established", is_sequential=False,
    )
    _require(df["bulk_condition_id"].nunique() == 3, "Expected three RNA treatment conditions")
    return df, [source.as_posix(), validation_source.as_posix()], [p.as_posix() for p in paths]


def _source_references(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    inputs = [HU / "source_manifest.json", HU / "README.md",
              TIDYI / "human_ss3_download_manifest.json", SELECTION / "file_cleanup_report.json"]
    hu_manifest = json.loads((root / inputs[0]).read_text(encoding="utf-8"))
    (root / inputs[1]).read_text(encoding="utf-8")
    archive_manifest = json.loads((root / inputs[2]).read_text(encoding="utf-8"))
    cleanup = json.loads((root / inputs[3]).read_text(encoding="utf-8-sig"))
    archive = TIDYI / "classifier_data.tar.gz"
    _require((root / archive).is_file(), "Preserved TIDYI archive is missing")
    _require((root / archive).stat().st_size == archive_manifest["expected_bytes"],
             "TIDYI archive size no longer matches its verified manifest")
    _require(archive_manifest["archive_complete"] and
             archive_manifest["archive_sha256"] == cleanup["recovery_archive_sha256"],
             "TIDYI archive verification records disagree")
    hu_paths = {name: _safe_source(root, HU, name) for name in hu_manifest}
    hu_assets = list(hu_paths.values())
    _require(all((root / p).is_file() for p in hu_assets), "A recorded Hu source document is missing")
    _require(all((root / hu_paths[name]).stat().st_size == record["bytes"]
                 for name, record in hu_manifest.items()), "Hu source document size changed")
    supplement = HU / "pharmaceutics-17-01044-s001.zip"
    with zipfile.ZipFile(root / supplement) as zipped:
        members = zipped.namelist()
    _require(members == ["pharmaceutics-3727392-supplementary.pdf"],
             "Hu supplementary archive content changed; review raw-data availability")
    inputs.append(supplement)
    sequence = pd.DataFrame([{
        "source_id": "hu2025_pemetrexed_osimertinib",
        "doi": "10.3390/pharmaceutics17081044", "pmcid": "PMC12389513",
        "source_kind": "published_sequence_design_reference",
        "cell_lines": "HCC827;NCI-H1975", "is_sequential": True,
        "source_file": (HU / "PMC12389513.xml").as_posix(),
        "source_url": hu_manifest["PMC12389513.xml"]["source"],
        "supplement_archive": supplement.as_posix(), "supplement_member": members[0],
        "supplement_pdf": _safe_source(root, HU, members[0]).as_posix(),
        "figure_file": (HU / "pharmaceutics-17-01044-g003.jpg").as_posix(),
        "availability_evidence": (HU / "README.md").as_posix(),
        "availability_checked_date": "2026-09-18", "has_paired_rna": False,
        "has_replicate_level_endpoint_values": False,
        "use_scope": "reference_only_not_numeric_training_data",
    }])
    # Member identities were read from the retained tar on 2026-09-19.
    # Reuse its verified manifest; do not decompress multi-GB matrices per build.
    pending_specs = [
        ("tidyi_human_10x", "human_10x.h5ad", "hold_for_metadata_review",
         "Human platform data retained in archive; not yet curated for lung inputs"),
        ("tidyi_mouse_10x", "mouse_10x.h5ad", "excluded_from_human_lung_inputs",
         "Mouse archive member retained; outside current human lung input scope"),
        ("tidyi_mouse_ss3", "mouse_ss3.h5ad", "excluded_from_human_lung_inputs",
         "Mouse archive member retained; outside current human lung input scope"),
        ("tidyi_rates", "rates_FinalExp_week2.csv", "hold_for_parameter_interpretation",
         "Rate parameter table; must not be used as measured Annexin endpoints"),
    ]
    pending_rows = [{
        "resource_id": resource_id, "source_id": "tidyi_author_classifier_archive",
        "status": status, "source_file": archive.as_posix(), "archive_member": member,
        "archive_present": True, "member_verification": "prior_verified_archive_member",
        "member_inventory_checked_date": "2026-09-19",
        "archive_sha256_from_manifest": archive_manifest["archive_sha256"],
        "source_evidence": (TIDYI / "human_ss3_download_manifest.json").as_posix(),
        "usable_as_current_training_input": False, "reason": reason,
    } for resource_id, member, status, reason in pending_specs]
    for resource_id, reason in [
        ("hu2025_paired_rna", "Paired RNA expression data not acquired"),
        ("hu2025_replicate_endpoints", "Replicate-level Annexin numeric values not acquired"),
    ]:
        pending_rows.append({
            "resource_id": resource_id, "source_id": "hu2025_pemetrexed_osimertinib",
            "status": "not_acquired", "source_file": None, "archive_member": None,
            "archive_present": None, "member_verification": None,
            "member_inventory_checked_date": None, "archive_sha256_from_manifest": None,
            "source_evidence": (HU / "README.md").as_posix(),
            "usable_as_current_training_input": False, "reason": reason,
        })
    pending = pd.DataFrame(pending_rows)
    return sequence, pending, [p.as_posix() for p in inputs], [archive.as_posix()] + [p.as_posix() for p in hu_assets]


def build_aux_tables(project_root: Path, output_root: Path) -> dict:
    """Write five small tables into a new staging root created by the caller.

    Returned ``inputs`` names files read by this module; ``source_assets`` names
    underlying local data/documents checked for existence but not reprocessed.
    No RNA counts or matrix are joined, transformed, copied, or normalized.
    """
    root, output = Path(project_root).resolve(), Path(output_root).resolve()
    _require(output != root and output.is_relative_to(root),
             "Auxiliary staging output must be a directory within project_root")
    if not output.is_dir():
        raise FileNotFoundError("Caller must create output_root before building auxiliary tables")
    names = {
        "endpoint_measurements": "endpoints/measurements.parquet",
        "endpoint_components": "endpoints/components.parquet",
        "macaya_rna_samples": "auxiliary/macaya_rna_samples.parquet",
        "sequence_sources": "auxiliary/sequence_sources.parquet",
        "pending_data": "auxiliary/pending_data.parquet",
    }
    targets = [root / _safe_source(root, output.relative_to(root), path) for path in names.values()]
    if any(path.exists() for path in targets):
        raise FileExistsError("Auxiliary table output already exists; refusing overwrite")
    measurements, components, ep_inputs, ep_assets = _endpoint_tables(root)
    rna, rna_inputs, rna_assets = _rna_samples(root)
    sequences, pending, ref_inputs, ref_assets = _source_references(root)
    frames = [measurements, components, rna, sequences, pending]
    for target, frame in zip(targets, frames):
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    return {
        "counts": {
            "endpoint_positions": len(measurements),
            "measured_endpoints": int(measurements["has_measured_endpoint"].sum()),
            "missing_endpoints": int((~measurements["has_measured_endpoint"]).sum()),
            "endpoint_conditions": measurements["endpoint_condition_id"].nunique(),
            "endpoint_components": len(components), "macaya_rna_samples": len(rna),
            "sequence_sources": len(sequences), "pending_data": len(pending),
        },
        "tables": names,
        "inputs": sorted(set(ep_inputs + rna_inputs + ref_inputs)),
        "source_assets": sorted(set(ep_assets + rna_assets + ref_assets)),
    }
