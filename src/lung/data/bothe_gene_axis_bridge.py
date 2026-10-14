"""Metadata-only exact NCBI dbXrefs bridge; no count or model qualification."""
import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

FROZEN_HASHES = {
    "annotation_manifest": "c50c69db266c8de1b5ce91f8f0aa364cf2cffbb5a7e185625314037f6200e9c2",
    "axis": "4d38a758179d9123fff1ffda1f1eae85fe5f57faa31a3a6df01fcdefbb5314fc",
    "axis_manifest": "513c9650acec6613d61c6f32be26d68ec477d7fb9c3c38f43742829443cbba05",
    "official": "2f9f947bea41dc2dc7bc3826bee304f770e2afa98a1478d79466c95cf6c3ca8a",
    "official_readme": "303cce0aa6012ff94ce34d88ff999a24131d838e4acac76a4abc1880b59d5e2e",
    "official_head": "6fa10dafc83208e611af7d99d07465847f4f7c3f4ebbea4fa86e1b84ea31ea2e",
    "gene_table": "fb6edae7f2458f33ea45ff7765be2e8d2ce0861435eea501732aa591e6231a1f",
    "scope_ledger": "a24ad3e640aa47757fc0f00a8d41387c884d92059c2fd90c27def40ab990c7d2",
}
ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "08-datasets/Bothe2021/E-MTAB-9923/reference"
REGISTERED_PATHS = {
    "annotation_manifest": REFERENCE / "unique-introns-v1/manifest.json",
    "gene_table": REFERENCE / "unique-introns-v1/gene_table.tsv",
    "scope_ledger": REFERENCE / "unique-introns-v1/transcript_scope_ledger.tsv",
    **{key: REFERENCE / "gene-axis-bridge-v1/official" / filename for key, filename in
       {"official": "Homo_sapiens.gene_info.gz", "official_readme": "README.txt", "official_head": "head.json"}.items()},
    **{key: ROOT / "08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1" / filename
       for key, filename in {"axis": "genes.parquet", "axis_manifest": "manifest.json"}.items()},
}
OUTPUT_NAMES = {"source_status_ledger.tsv", "joint_axis.tsv", "scope_status_ledger.tsv", "official_tax_excluded.tsv"}
QUALIFICATIONS = {key: False for key in ("raw", "counts", "source_decoder", "normalization_equivalence", "model", "function")}
SAMPLE_IDS = tuple(f"RNA-seq_{history}_rep{rep}" for history in
                   ("dex_dex", "dex_etoh", "etoh_dex", "etoh_etoh") for rep in (1, 2, 3))


@dataclass(frozen=True)
class JointBulkCountProjection:
    version: str
    sample_ids: tuple
    ensembl_ids: tuple
    ncbi_gene_ids: tuple
    values: tuple
    observed_mask: tuple
    observed_samples: tuple
    counts_complete: bool
    domain: str = field(default="bulk_total_RNA_unique_intron_fragment_counts_on_joint32461", init=False)
    model_eligible: bool = field(default=False, init=False)
    function_eligible: bool = field(default=False, init=False)


def project_production_counts(version):
    """Consume source-bound production only; return gene-major exact integer tuples.

    Missing/invalid production raises unchanged; this API never creates a pending
    placeholder. Fixture tests monkeypatch internal loaders and grant no evidence.
    """
    from .bothe_count_production import load_production
    counts = load_production(version)
    ledger, axis = load_bridge(REFERENCE / "gene-axis-bridge-v1")
    genes = tuple(ledger.ncbi_gene_id)
    statuses = tuple(ledger.annotation_status)
    if (counts.version != version or counts.gene_ids != genes or counts.annotation_status != statuses
            or len(genes) != 29634 or counts.sample_ids != SAMPLE_IDS
            or len(axis) != 32461 or axis.common_gene_index.tolist() != list(range(32461))
            or axis.ensembl_id.duplicated().any()
            or counts.domain != "bulk_total_RNA_unique_intron_fragment_counts"
            or counts.model_eligible is not False or counts.function_eligible is not False):
        raise ValueError("Production source gene/sample axis or observation domain mismatch")
    available = counts.observed_samples
    if (not isinstance(available, tuple) or not available or len(set(available)) != len(available)
            or not set(available).issubset(SAMPLE_IDS)
            or type(counts.counts_complete) is not bool
            or counts.counts_complete is not (len(available) == 12)
            or not isinstance(counts.values, tuple) or len(counts.values) != len(genes)):
        raise ValueError("Invalid production completeness or count shape")
    for status, row in zip(statuses, counts.values):
        if not isinstance(row, tuple) or len(row) != 12:
            raise ValueError("Invalid production count row shape")
        for sample, value in zip(SAMPLE_IDS, row):
            observed = status == "retained" and sample in available
            if (observed and (type(value) is not int or not 0 <= value <= 2**63 - 1)) or (not observed and value is not None):
                raise ValueError("Count observation must be exact integer or unobserved None")
    mapped = ledger.loc[ledger.id_bridge_mask]
    if (len(mapped) != 17107 or mapped.ncbi_gene_id.duplicated().any()
            or mapped.ensembl_id.duplicated().any() or mapped.common_gene_index.duplicated().any()):
        raise ValueError("ID bridge must remain one-to-one")
    source_index = {gene: index for index, gene in enumerate(genes)}
    empty = (None,) * 12
    rows, identities = [], []
    for row in axis.itertuples(index=False):
        if row.id_bridge_mask:
            source_row = ledger.iloc[source_index[row.ncbi_gene_id]]
            if (source_row.mapping_status != "mapped" or source_row.annotation_status != "retained"
                    or source_row.ensembl_id != row.ensembl_id or source_row.common_gene_index != row.common_gene_index):
                raise ValueError("Joint/source ID bridge correspondence mismatch")
            rows.append(counts.values[source_index[row.ncbi_gene_id]])
            identities.append(row.ncbi_gene_id)
        else:
            rows.append(empty)
            identities.append(None)
    masks = tuple(tuple(value is not None for value in row) for row in rows)
    return JointBulkCountProjection(version, SAMPLE_IDS, tuple(axis.ensembl_id), tuple(identities),
                                    tuple(rows), masks, available, counts.counts_complete)


def supplemental_tables(scope, raw_official):
    scope = scope.copy()
    scope["common_gene_index"] = pd.NA
    scope["id_bridge_mask"] = False
    scope["mapping_status"] = scope.reason + "_transcript_record_no_independent_mapping"
    excluded = raw_official.loc[raw_official["#tax_id"].ne("9606"), ["#tax_id", "GeneID", "dbXrefs"]].copy()
    excluded["mapping_status"] = "excluded_non_9606"
    return scope, excluded


def summarize(source, full, ledger, official, scope, excluded):
    return {"official_human_rows": len(official), "annotation_genes": len(source),
            "joint_axis_genes": len(full), "mapped_measurable_genes": int(full.id_bridge_mask.sum()),
            "source_statuses": ledger.mapping_status.value_counts().to_dict(),
            "official_non_9606_excluded": len(excluded), "scope_transcript_records": len(scope),
            "scope_reasons": scope.reason.value_counts().to_dict()}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gene_info(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle, sep="\t", dtype="string")
    required = {"#tax_id", "GeneID", "dbXrefs"}
    if not required.issubset(frame.columns):
        raise ValueError("Expected human gene_info schema")
    frame = frame.loc[frame["#tax_id"].eq("9606")].copy()
    if frame.empty:
        raise ValueError("No tax_id 9606 records")
    if (frame.GeneID.isna().any() or frame.dbXrefs.isna().any() or frame.GeneID.duplicated().any()
            or not frame.GeneID.str.fullmatch(r"[1-9][0-9]*").all()):
        raise ValueError("GeneID must be unique positive integers")
    return frame


def build_tables(source, axis, official):
    """Return all source rows and all target rows, preserving unmapped NA."""
    if source.ncbi_gene_id.duplicated().any():
        raise ValueError("Duplicate annotation GeneID")
    if source.ncbi_gene_id.isna().any() or not source.ncbi_gene_id.astype("string").str.fullmatch(r"[1-9][0-9]*").all():
        raise ValueError("Invalid annotation GeneID")
    if official.GeneID.duplicated().any():
        raise ValueError("Duplicate official GeneID")
    if official.GeneID.isna().any() or official.dbXrefs.isna().any():
        raise ValueError("Missing official identity field")
    if axis.ensembl_id.duplicated().any() or axis.ensembl_id.isna().any():
        raise ValueError("Target Ensembl identifiers must be unique")
    if axis.common_gene_index.tolist() != list(range(len(axis))):
        raise ValueError("Target axis order must be contiguous")
    if not axis.ensembl_id.str.fullmatch(r"ENSG[0-9]{11}").all():
        raise ValueError("Target must contain exact unversioned Ensembl gene IDs")
    relations = {}
    reverse = {}
    for row in official.itertuples(index=False):
        ids = tuple(sorted(set(token.split(":", 1)[1] for token in row.dbXrefs.split("|")
                               if token.startswith("Ensembl:"))))
        if any(not re.fullmatch(r"ENSG[0-9]{11}", key) for key in ids):
            raise ValueError("Unexpected Ensembl dbXref identifier")
        relations[row.GeneID] = ids
        for key in ids:
            reverse.setdefault(key, set()).add(row.GeneID)
    target = dict(zip(axis.ensembl_id, axis.common_gene_index))
    rows = []
    for row in source.itertuples(index=False):
        key = str(row.ncbi_gene_id)
        candidates = relations.get(key, ())
        status = "mapped"
        if key not in relations:
            status = "absent_current_ncbi_deprecation_unconfirmed"
        elif not candidates:
            status = "no_ensembl_dbxref"
        elif len(candidates) != 1:
            status = "ambiguous_one_to_many"
        elif len(reverse[candidates[0]]) != 1:
            status = "ambiguous_many_to_one"
        elif candidates[0] not in target:
            status = "not_in_joint_axis"
        elif row.status != "retained":
            status = "annotation_unmeasurable"
        matched = status == "mapped"
        rows.append({"ncbi_gene_id": key, "annotation_status": row.status,
                     "official_ensembl_candidates": "|".join(candidates),
                     "ensembl_id": candidates[0] if matched else None,
                     "common_gene_index": target[candidates[0]] if matched else None,
                     "mapping_status": status, "id_bridge_mask": matched})
    ledger = pd.DataFrame(rows)
    ledger["common_gene_index"] = ledger.common_gene_index.astype("Int64")
    full = axis[["common_gene_index", "ensembl_id"]].copy()
    mapped = ledger[ledger.id_bridge_mask][["common_gene_index", "ncbi_gene_id"]]
    full = full.merge(mapped, on="common_gene_index", how="left", validate="one_to_one", sort=False)
    full["id_bridge_mask"] = full.ncbi_gene_id.notna()
    full["mapping_status"] = full.id_bridge_mask.map({True: "mapped", False: "no_eligible_source"})
    return ledger, full


def load_bridge(directory):
    """Rehash registered small metadata and reconstruct tables from official IDs."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "bothe-gene-axis-bridge-v1":
        raise ValueError("Unknown bridge schema")
    if (manifest.get("validation_revision") != 2 or manifest.get("status") != "ready_id_bridge_only"
            or type(manifest.get("counts_recovered")) is not int or manifest["counts_recovered"] != 0
            or manifest.get("qualifications") != QUALIFICATIONS):
        raise ValueError("Metadata bridge cannot claim raw/count/function/model qualification")
    if any(value is not False for value in manifest["qualifications"].values()):
        raise ValueError("Bridge qualifications must be boolean false")
    if manifest.get("tax_id") != 9606 or manifest.get("source_domain") != "Bothe bulk unique intronic RNA counts; distinct from joint single-cell expression":
        raise ValueError("Bridge observation domain mismatch")
    if set(manifest["inputs"]) != set(REGISTERED_PATHS) or set(manifest["outputs"]) != OUTPUT_NAMES:
        raise ValueError("Bridge input/output schema set mismatch")
    for name, expected_hash in FROZEN_HASHES.items():
        entry = manifest["inputs"][name]
        if (entry["sha256"] != expected_hash or Path(entry["path"]).resolve() != REGISTERED_PATHS[name].resolve()):
            raise ValueError("Frozen bridge axis/annotation hash mismatch")
    for entry in manifest["inputs"].values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Bridge input hash mismatch")
        if Path(entry["path"]).stat().st_size != entry["bytes"]:
            raise ValueError("Bridge input size mismatch")
    annotation = json.loads(REGISTERED_PATHS["annotation_manifest"].read_text(encoding="utf-8"))
    for key, filename in (("gene_table", "gene_table.tsv"), ("scope_ledger", "transcript_scope_ledger.tsv")):
        if annotation["output_files"][filename]["sha256"] != manifest["inputs"][key]["sha256"]:
            raise ValueError("Annotation output identity mismatch")
    for name, expected_hash in manifest["outputs"].items():
        if sha256(directory / name) != expected_hash:
            raise ValueError("Bridge output hash mismatch")
    inputs = manifest["inputs"]
    source = pd.read_csv(inputs["gene_table"]["path"], sep="\t", dtype="string")
    official = read_gene_info(inputs["official"]["path"])
    result = build_tables(source, pd.read_parquet(inputs["axis"]["path"]), official)
    scope, excluded = supplemental_tables(pd.read_csv(inputs["scope_ledger"]["path"], sep="\t", dtype="string"),
                                          pd.read_csv(inputs["official"]["path"], sep="\t", dtype="string"))
    if manifest["summary"] != summarize(source, result[1], result[0], official, scope, excluded):
        raise ValueError("Bridge summary does not match actual source")
    for name, frame in zip(("source_status_ledger.tsv", "joint_axis.tsv", "scope_status_ledger.tsv", "official_tax_excluded.tsv"), (*result, scope, excluded)):
        if sha256(directory / name) != manifest["outputs"][name]:
            raise ValueError("Bridge output hash mismatch")
        saved = pd.read_csv(directory / name, sep="\t", dtype="string")
        expected = pd.read_csv(__import__("io").StringIO(frame.to_csv(sep="\t", index=False)), sep="\t", dtype="string")
        pd.testing.assert_frame_equal(saved, expected)
    return result
