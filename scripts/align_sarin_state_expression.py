"""Map Sarin microarray probes to the common gene axis without rescaling/averaging."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "08-datasets/LungCancer_A549_Integration"
SOURCES = BASE / "endpoint-alignment-v1/source_reviews/state_function_round2"
DEFAULT_OUTPUT = BASE / "endpoint-alignment-v1/source_reviews/state_gene_round4"
SCALE = "deposited_background_corrected_quantile_normalized_microarray_VALUE"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def strings(path, **kwargs):
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def transcript_ids(row):
    """Use only explicit ENST identifiers, never gene-name inference."""
    return sorted(set(re.findall(r"ENST\d+(?:\.\d+)?", row["ENSEMBL_ID"] + "|" + row["ACCESSION_STRING"])))


def read_transcripts(path, wanted):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]*)";', fields[8]))
            transcript = attrs.get("transcript_id", "")
            if transcript in wanted:
                rows.append({"transcript_id": transcript, "ensembl_gene_id": attrs["gene_id"],
                             "annotation_gene_name": attrs.get("gene_name", ""),
                             "chromosome": fields[0], "start_1based": int(fields[3]),
                             "end_1based": int(fields[4]), "strand": fields[6],
                             "annotation_line_1based": line_number})
    if not rows:
        raise ValueError("No requested ENST identifiers found in transcript annotation")
    return pd.DataFrame(rows)


def mapping_ledger(annotation, axis, hgnc, transcripts):
    """A primary Entrez bridge and an explicit GRCh37 ENST cross-check."""
    if not axis.ensembl_id.is_unique or not axis.common_gene_index.is_unique:
        raise ValueError("Common axis must have unique ENSG identifiers and coordinates")
    if not np.array_equal(axis.common_gene_index.to_numpy(), np.arange(len(axis))):
        raise ValueError("Common axis order is not its contiguous coordinate order")
    if not annotation.probe_id.is_unique:
        raise ValueError("Probe IDs must be unique")
    entrez_genes, entrez_hgnc = defaultdict(set), defaultdict(set)
    for row in hgnc.to_dict("records"):
        if row["status"] == "Approved" and row["entrez_id"]:
            genes = set(re.findall(r"ENSG\d+", row["ensembl_gene_id"]))
            entrez_genes[row["entrez_id"]].update(genes)
            entrez_hgnc[row["entrez_id"]].add(row["hgnc_id"])
    transcript_genes = defaultdict(set)
    for row in transcripts.to_dict("records"):
        transcript_genes[row["transcript_id"]].add(row["ensembl_gene_id"])
    coordinates = axis.set_index("ensembl_id").common_gene_index.to_dict()
    symbols = axis.set_index("ensembl_id").gene_symbol.to_dict()
    rows = []
    for index, row in enumerate(annotation.to_dict("records")):
        ids = transcript_ids(row)
        # The source uses unversioned ENST IDs. Explicitly reject unseen versions.
        unknown = [item for item in ids if item not in transcript_genes]
        ent = entrez_genes.get(row["LOCUSLINK_ID"], set())
        tx = set().union(*(transcript_genes.get(item, set()) for item in ids)) if ids else set()
        assigned, basis = "", ""
        if len(ent) > 1 or len(tx) > 1:
            status = "ambiguous_gene_annotation"
        elif ent and tx and ent != tx:
            status = "entrez_transcript_annotation_conflict"
        elif ent:
            assigned = next(iter(ent))
            basis = "platform_Entrez_to_HGNC_ENSG"
            status = "mapped_one_gene"
        elif tx and not unknown:
            assigned = next(iter(tx))
            basis = "all_platform_ENST_to_Ensembl75_ENSG"
            status = "mapped_one_gene"
        elif tx:
            status = "incomplete_transcript_mapping_without_gene_bridge"
        else:
            status = "no_verified_gene_bridge"
        if assigned and assigned not in coordinates:
            status = "resolved_gene_outside_common_axis"
        rows.append({"source_probe_row": index, "probe_id": row["probe_id"],
                     "source_refseq": row["REFSEQ"], "source_entrez_id": row["LOCUSLINK_ID"],
                     "source_gene_symbol": row["GENE_SYMBOL"], "source_primary_ensembl_field": row["ENSEMBL_ID"],
                     "source_all_transcript_ids": "|".join(ids), "unresolved_transcript_ids": "|".join(unknown),
                     "hgnc_ids": "|".join(sorted(entrez_hgnc[row["LOCUSLINK_ID"]])),
                     "hgnc_gene_candidates": "|".join(sorted(ent)),
                     "grch37_transcript_gene_candidates": "|".join(sorted(tx)),
                     "resolved_ensembl_gene_id": assigned, "common_gene_index": coordinates.get(assigned, -1),
                     "common_gene_symbol": symbols.get(assigned, ""), "mapping_basis": basis,
                     "mapping_status": status, "transcript_crosscheck_complete": bool(ids) and not unknown,
                     "source_symbol_equals_common_symbol": bool(assigned) and row["GENE_SYMBOL"] == symbols.get(assigned),
                     "symbol_used_to_choose_gene": False})
    ledger = pd.DataFrame(rows)
    eligible = ledger.mapping_status.eq("mapped_one_gene")
    counts = ledger.loc[eligible, "resolved_ensembl_gene_id"].value_counts()
    ledger["resolved_probes_for_gene"] = ledger.resolved_ensembl_gene_id.map(counts).fillna(0).astype(int)
    ledger["gene_matrix_status"] = np.where(
        eligible & ledger.resolved_probes_for_gene.eq(1), "unique_resolved_probe",
        np.where(eligible, "multiple_resolved_probes_preserved_not_aggregated", "excluded_unresolved_or_outside_axis"))
    return ledger


def aligned_matrix(probe_values, ledger, gene_count):
    values = np.asarray(probe_values, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(ledger) or not np.isfinite(values).all():
        raise ValueError("Probe values must be a finite probe-by-sample matrix")
    result = np.full((values.shape[1], gene_count), np.nan, dtype=np.float64)
    selected = ledger.gene_matrix_status.eq("unique_resolved_probe")
    indices = ledger.loc[selected, "common_gene_index"].to_numpy(dtype=int)
    if len(set(indices)) != len(indices) or (indices < 0).any() or (indices >= gene_count).any():
        raise ValueError("Selected gene coordinates must be unique and in range")
    result[:, indices] = values[selected.to_numpy()].T
    return result, np.isfinite(result)


def verify_source():
    """Read all deposited values again, verify old extraction and sample identities."""
    with gzip.open(SOURCES / "GSE108214_family.soft.gz", "rt", encoding="utf-8") as handle:
        text = handle.read()
    platform = strings(StringIO(text.split("!platform_table_begin\n", 1)[1].split("!platform_table_end", 1)[0]), sep="\t")
    expression = strings(SOURCES / "GSE108214_probe_expression.csv.gz")
    annotation = strings(SOURCES / "GSE108214_probe_annotation.csv.gz")
    samples = strings(SOURCES / "GSE108214_samples.csv")
    expected_annotation = platform.set_index("ID").loc[expression.ID_REF].reset_index().rename(columns={"ID": "probe_id", "ID_REF": "probe_id"})
    pd.testing.assert_frame_equal(annotation, expected_annotation)
    samples_raw = []
    for block in re.split(r"(?=\^SAMPLE = )", text)[1:]:
        accession = block.splitlines()[0].split(" = ", 1)[1]
        table = strings(StringIO(block.split("!sample_table_begin\n", 1)[1].split("!sample_table_end", 1)[0]), sep="\t")
        if table.ID_REF.tolist() != expression.ID_REF.tolist() or table.VALUE.tolist() != expression[accession].tolist():
            raise ValueError(f"Source expression mismatch: {accession}")
        metadata = defaultdict(list)
        for line in block.splitlines():
            if line.startswith("!Sample_") and " = " in line:
                key, value = line.split(" = ", 1)
                metadata[key[8:]].append(value)
        chars = dict(value.split(": ", 1) for value in metadata["characteristics_ch1"])
        cached = samples.loc[samples.geo_accession.eq(accession)].iloc[0]
        state = "A549rCDDP2000" if chars["sensitivity"] == "resistant" else "A549"
        if cached.sample_title != metadata["title"][0] or cached.state_id != state or float(cached.dose_uM) != float(chars["treatment conc"]):
            raise ValueError(f"Source sample metadata mismatch: {accession}")
        if cached.treatment_h != "24" or cached.observed_initial_state != "False" or cached.paired_to_function_replicate != "False":
            raise ValueError("Reference time or pairing boundary changed")
        samples_raw.append(accession)
    if expression.shape != (29729, 23) or len(samples_raw) != 22 or samples.geo_accession.tolist() != samples_raw:
        raise ValueError("Unexpected source dimensions or sample order")
    return expression, annotation, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError("Frozen output already exists; use a new output directory")
    download_manifest = out / "annotation_download_manifest.json"
    downloads = json.loads(download_manifest.read_text(encoding="utf-8"))
    for record in downloads:
        if sha256(out / record["file"]) != record["sha256"]:
            raise ValueError(f"Annotation checksum mismatch: {record['file']}")
    inventory = json.loads((SOURCES / "source_inventory.json").read_text(encoding="utf-8"))
    source_names = ["GSE108214_family.soft.gz", "GSE108214_probe_expression.csv.gz", "GSE108214_probe_annotation.csv.gz",
                    "GSE108214_samples.csv", "GSE108214_protocol_metadata.json", "PMC5877628.xml", "PMC5528889.xml",
                    "Sarin_condition_correspondence.csv", "Sarin2017_Annexin_source_values.csv"]
    source_records = {item["file"]: item for item in inventory["files"]}
    for name in source_names:
        if sha256(SOURCES / name) != source_records[name]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    expression, annotation, samples = verify_source()
    axis_path = BASE / "prepared/v1/gene_map.parquet"
    axis = pd.read_parquet(axis_path)
    wanted = set().union(*(set(transcript_ids(row)) for row in annotation.to_dict("records")))
    transcripts = read_transcripts(out / "Homo_sapiens.GRCh37.75.gtf.gz", wanted)
    hgnc = strings(out / "hgnc_complete_set_2026-10-10.txt", sep="\t")
    ledger = mapping_ledger(annotation, axis, hgnc, transcripts)
    values = expression.iloc[:, 1:].to_numpy(dtype=np.float64)
    aligned, mask = aligned_matrix(values, ledger, len(axis))
    outputs = []
    def table(name, data):
        data.to_csv(out / name, index=False, encoding="utf-8", compression="infer")
        outputs.append(name)
    table("probe_gene_mapping.csv.gz", ledger)
    table("transcript_gene_annotation_subset.csv.gz", transcripts)
    table("GSE108214_aligned_samples.csv", samples)
    table("common_gene_axis.csv.gz", axis)
    selected = ledger.gene_matrix_status.eq("unique_resolved_probe")
    table("gene_matrix_columns.csv.gz", ledger.loc[selected, ["probe_id", "common_gene_index", "resolved_ensembl_gene_id", "mapping_basis"]])
    np.savez_compressed(out / "GSE108214_original_probe_values.npz", expression=values,
                        probe_ids=expression.ID_REF.to_numpy(dtype=str), sample_ids=samples.geo_accession.to_numpy(dtype=str), value_scale=SCALE)
    np.savez_compressed(out / "GSE108214_common_gene_values.npz", expression=aligned, mask=mask,
                        gene_ids=axis.ensembl_id.to_numpy(dtype=str), sample_ids=samples.geo_accession.to_numpy(dtype=str), value_scale=SCALE)
    outputs.extend(["GSE108214_original_probe_values.npz", "GSE108214_common_gene_values.npz"])
    with np.load(out / "GSE108214_original_probe_values.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["expression"], values)
    with np.load(out / "GSE108214_common_gene_values.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["expression"], aligned)
        np.testing.assert_array_equal(saved["mask"], mask)
        np.testing.assert_array_equal(saved["expression"][:, ledger.loc[selected, "common_gene_index"]], values[selected.to_numpy()].T)
        if not np.isnan(saved["expression"][~saved["mask"]]).all():
            raise ValueError("Unobserved genes were not preserved as NaN")
    snapshot = "align_sarin_state_expression_snapshot.py"
    (out / snapshot).write_bytes(Path(__file__).read_bytes())
    outputs.append(snapshot)
    mapped = ledger.mapping_status.eq("mapped_one_gene")
    validation = {"source_probe_count": len(expression), "sample_count": len(samples), "target_gene_count": len(axis),
                  "source_values_reread_equal": True, "all_npz_values_roundtrip_equal": True,
                  "source_value_count": int(values.size), "source_value_min": float(values.min()), "source_value_max": float(values.max()),
                  "mapping_status_counts": ledger.mapping_status.value_counts().to_dict(),
                  "gene_matrix_status_counts": ledger.gene_matrix_status.value_counts().to_dict(),
                  "resolved_axis_genes_including_multiple_probes": int(ledger.loc[mapped, "resolved_ensembl_gene_id"].nunique()),
                  "strict_unique_probe_genes": int(selected.sum()), "unfilled_common_genes": int(len(axis) - selected.sum()),
                  "transcript_ids_requested": len(wanted), "transcript_ids_found": int(transcripts.transcript_id.nunique()),
                  "mapping_uses_symbols": False, "probe_aggregation_performed": False, "normalization_performed": False,
                  "reference_groups": samples.groupby(["state_id", "dose_uM", "reference_role"], sort=True).size().reset_index(name="sample_count").to_dict("records"),
                  "function_endpoint_unit": "fold_change_relative_to_respective_untreated_control", "same_experiment_pairing": False,
                  "observed_initial_state": False, "model_input_ready": False, "full_research_feasibility_proven": False}
    (out / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append("validation.json")
    source_paths = [SOURCES / name for name in source_names] + [SOURCES / "source_inventory.json", axis_path, download_manifest]
    source_paths += [out / record["file"] for record in downloads]
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "gene_identity_alignment_verified_not_model_or_function_pairing", "measurement": "bulk_microarray",
                "value_scale": SCALE, "log_base": "not_explicitly_annotated_in_source",
                "annotation_version": {"platform": "GPL17077; GRCh37/hg19; Ensembl release unspecified",
                                       "transcript_bridge": "Ensembl release 75 GRCh37; build-compatible, not claimed original-platform exact release",
                                       "entrez_bridge": "HGNC download-date snapshot 2026-10-10; current identity crosswalk, not historical annotation"},
                "sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in source_paths],
                "outputs": {name: sha256(out / name) for name in outputs}, "validation": validation}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
