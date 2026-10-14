"""Materialize only official GSE193258 processed bulk RNA, preserving source axes."""
from __future__ import annotations
import csv
import hashlib
import json
import gzip
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/20261011-data-alignment-round16/state_function_new_sources"
OUT = ROOT / "08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193258/suppl/"
NAMES = {"estimated_counts": "GSE193258_RNAseq_estimated_counts.tsv.gz", "log2TPM": "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"}
LONG_HOURS = {"PC9": 168, "H1975": 72, "HCC827": 96, "HCC2935": 240}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}

def download(name, raw):
    url = BASE + name
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    target = raw / name
    if not target.exists():
        part = target.with_suffix(target.suffix + ".part")
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("Server did not honor resume range")
            with part.open("ab" if offset else "wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        if part.stat().st_size != size:
            raise ValueError("Downloaded size mismatch")
        part.rename(target)
    if target.stat().st_size != size:
        raise ValueError("Existing source size mismatch")
    return {"url": url, "head_bytes": size, "last_modified": head.headers.get("Last-Modified"), **digest(target)}

def validate_matrix(frame, counts):
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Duplicate gene or sample axis")
    if any(not str(x).strip() for x in frame.index):
        raise ValueError("Empty gene symbol")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite RNA values")
    if counts and (values < 0).any():
        raise ValueError("Negative estimated counts")
    return values

def sample_table(samples, columns):
    lookup = {}
    for sample in samples:
        title = sample["!Sample_title"][0].removesuffix(" [RNA-seq]")
        cell, rest = title.split("_", 1)
        rep = int(rest.rsplit("_", 1)[1])
        if rest.startswith("DMSO_"): state = "parental_DMSO"
        elif rest.startswith("osi_acute_"): state = "acute_post_treatment"
        elif rest.startswith("osi_DTP_"): state = "DTP"
        elif rest.startswith("short_washout_"): state = "short_washout"
        elif rest.startswith("long_washout_"): state = "long_washout"
        else: raise ValueError("Unknown sample title")
        suffix = {"parental_DMSO": "DMSO", "acute_post_treatment": "osi_acute", "DTP": "osi_DTP", "short_washout": "short_wash", "long_washout": "long_wash"}[state]
        name = f"{cell}_{suffix}_{rep}"
        if name in lookup or rep not in (1, 2, 3) or cell not in LONG_HOURS:
            raise ValueError("Duplicate or invalid metadata sample")
        characteristics = sample["!Sample_characteristics_ch1"]
        if f"cell line: {cell}" not in characteristics:
            raise ValueError("Metadata cell line mismatch")
        expected = {"parental_DMSO": "DMSO", "acute_post_treatment": "osimertinib acute 24 hrs.", "DTP": "osimertinib DTPs 3w.", "short_washout": "osimertinib DTPs short washout 24 hrs."}[state] if state != "long_washout" else None
        if expected is not None and f"treatment: {expected}" not in characteristics:
            raise ValueError("Metadata treatment mismatch")
        if state == "long_washout":
            duration = {"PC9": "7 days", "H1975": "72 hrs.", "HCC827": "96 hrs.", "HCC2935": "10 days"}[cell]
            if f"treatment: osimertinib DTPs long washout {duration}" not in characteristics:
                raise ValueError("Metadata long washout clock mismatch")
        protocol = sample["!Sample_treatment_protocol_ch1"][0]
        if "biological triplicates" not in protocol or "500 nM osimertinib for 21 days" not in protocol:
            raise ValueError("Metadata replicate or induction protocol mismatch")
        lookup[name] = {"matrix_column": name, "gsm": sample["gsm"], "source_title": title, "cell_line": cell, "state": state, "biological_rna_replicate": rep, "measurement_level": "bulk", "parallel_reference": state == "parental_DMSO", "pretreatment_osimertinib_nM": 500 if state in ("DTP", "short_washout", "long_washout") else 0, "pretreatment_days": 21 if state in ("DTP", "short_washout", "long_washout") else 0, "parental_culture_days": 21 if state in ("parental_DMSO", "acute_post_treatment") else 0, "acute_osimertinib_nM": 500 if state == "acute_post_treatment" else 0, "acute_or_vehicle_hours": 24 if state in ("parental_DMSO", "acute_post_treatment") else 0, "washout_hours": 24 if state == "short_washout" else LONG_HOURS[cell] if state == "long_washout" else 0, "source_characteristics": json.dumps(characteristics), "same_well_function_pairing": "not_declared", "functional_y": "not_attached"}
    if len(lookup) != 60 or len(columns) != 60 or set(columns) != set(lookup):
        raise ValueError("Matrix/metadata sample coverage mismatch")
    counts = Counter((x["cell_line"], x["state"]) for x in lookup.values())
    if len(counts) != 20 or set(counts.values()) != {3}:
        raise ValueError("Expected four backgrounds x five states x three replicates")
    return [{"column_index": i, **lookup[c]} for i, c in enumerate(columns)]

def axis_rows(source_symbols, joint_ids, joint_symbols):
    if len(set(source_symbols)) != len(source_symbols):
        raise ValueError("Duplicate source symbols cannot be mapped")
    if len(joint_ids) != len(joint_symbols):
        raise ValueError("Joint ID/symbol length mismatch")
    counts = Counter(joint_symbols)
    mapping = {symbol: row for row, symbol in enumerate(source_symbols)}
    rows = [mapping.get(symbol) if counts[symbol] == 1 and symbol else None for symbol in joint_symbols]
    return pd.DataFrame({"common_axis_index": range(len(joint_ids)), "ensembl_id": joint_ids, "joint_gene_symbol": joint_symbols, "source_row_index": pd.array(rows, dtype="Int64"), "source_observed": [row is not None for row in rows], "mapping_status": ["ambiguous_joint_symbol" if counts[symbol] > 1 else "unique_exact_symbol" if row is not None else "missing_source_symbol" for symbol, row in zip(joint_symbols, rows)]})

def main():
    raw = OUT / "raw"
    derived = OUT / "rna"
    raw.mkdir(parents=True, exist_ok=True)
    derived.mkdir(exist_ok=True)
    sources = {key: download(name, raw) for key, name in NAMES.items()}
    frames = {key: pd.read_csv(raw / name, sep="\t", index_col=0, float_precision="round_trip") for key, name in NAMES.items()}
    first = frames["estimated_counts"]
    other = frames["log2TPM"]
    if not first.index.equals(other.index) or not first.columns.equals(other.columns):
        raise ValueError("Two source matrices have different axes/order")
    samples = json.loads((SOURCE / "GSE193258_60samples.json").read_text())
    metadata = sample_table(samples, first.columns.tolist())
    pd.DataFrame(metadata).to_csv(derived / "rna_samples.csv", index=False)
    genes = pd.DataFrame({"source_row_index": range(len(first)), "source_gene_id": first.index, "source_gene_symbol": first.index})
    genes.to_csv(derived / "rna_genes.csv", index=False)
    stats = {}
    for key, frame in frames.items():
        values = validate_matrix(frame, key == "estimated_counts")
        path = derived / f"{key}.npy"
        np.save(path, values, allow_pickle=False)
        if not np.array_equal(values, np.load(path, allow_pickle=False)):
            raise ValueError("Source value readback mismatch")
        with gzip.open(raw / NAMES[key], "rt", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            header = next(reader)
            if header[1:] != frame.columns.tolist(): raise ValueError("Source header mismatch")
            row_count = 0
            for row_count, row in enumerate(reader, 1):
                if row[0] != frame.index[row_count - 1] or not np.array_equal(np.array([float(x) for x in row[1:]]), values[row_count - 1]):
                    raise ValueError("Independent full source-value readback mismatch")
            if row_count != len(frame): raise ValueError("Source row count mismatch")
        stats[key] = {"shape": list(values.shape), "min": float(values.min()), "max": float(values.max()), "finite": True, "negative_values": int((values < 0).sum()), "noninteger_values": int((values != np.floor(values)).sum()), "exact_float64_readback": True, "independent_csv_all_values_checked": int(values.size)}
    axis = ROOT / "08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/genes.parquet"
    joint = pd.read_parquet(axis)
    candidates = [c for c in joint.columns if c in ("ensembl_id", "gene_id")]
    if len(candidates) != 1: raise ValueError(f"Ambiguous common axis columns: {joint.columns.tolist()}")
    ids = joint[candidates[0]].astype(str).tolist()
    if len(ids) != 32461 or len(set(ids)) != len(ids): raise ValueError("Unexpected joint axis")
    symbols = joint["gene_symbol"].fillna("").astype(str).tolist()
    coverage = axis_rows(genes.source_gene_symbol.tolist(), ids, symbols)
    coverage.query("mapping_status == 'ambiguous_joint_symbol'").to_csv(derived / "quarantined_axis_symbols.csv", index=False)
    coverage.to_csv(derived / "common_axis_coverage.csv", index=False)
    # No expression matrix on the common axis is emitted: missing entries remain absent.
    pd.DataFrame(metadata).query("state == 'parental_DMSO'").to_csv(derived / "parallel_baseline_samples.csv", index=False)
    if not pd.read_csv(derived / "rna_samples.csv").matrix_column.tolist() == first.columns.tolist(): raise ValueError("Metadata order readback mismatch")
    if not pd.read_csv(derived / "rna_genes.csv").source_gene_id.tolist() == first.index.tolist(): raise ValueError("Gene order readback mismatch")
    (derived / "geo_samples_source.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "processed_RNA_materialized_and_validated_only", "source_accession": "GSE193258", "source_files": sources, "units": {"estimated_counts": "tximport countsFromAbundance=lengthScaledTPM; Salmon estimated gene counts, noninteger allowed", "log2TPM": "log2(TPM + 0.01), negatives allowed"}, "source_metadata": digest(SOURCE / "GSE193258_60samples.json"), "rna_gene_count": len(first), "rna_sample_count": 60, "baseline_samples": 12, "background_count": 4, "states_per_background": 5, "biological_replicates_per_state": 3, "gene_annotation": "Ensembl v79; hg38 (GEO data_processing)", "source_axis_type": "gene_symbol; no Ensembl IDs or version suffixes supplied", "duplicate_source_gene_symbols": 0, "matrix_checks": stats, "common_axis": {"path": str(axis.relative_to(ROOT)), **digest(axis), "size": len(ids), "covered": int(coverage.source_observed.sum()), "missing": int((~coverage.source_observed).sum()), "coverage_fraction": float(coverage.source_observed.mean()), "comparison_only": True, "mapping_method": "exact symbol unique on both source and joint axes; no alias/version inference", "ambiguous_joint_rows": int((coverage.mapping_status == "ambiguous_joint_symbol").sum()), "missing_is_not_measured_zero": True}, "limitations": ["Bulk RNA; not single-cell", "DMSO is parental parallel reference, not measured same-cell t0", "Acute RNA is post-treatment", "Same-well RNA-functional pairing not declared; no Y copied to RNA triplicates", "No function values, model, training, forward or prediction", "No raw FASTQ, SRA or annotation re-quantification", "No technical/biological functional repeat inference from RNA replicates"], "files": {str(p.relative_to(OUT)): digest(p) for p in derived.iterdir() if p.is_file()}}
    (OUT / "rna_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
