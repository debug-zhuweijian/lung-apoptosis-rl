"""Recover source-native public expression without requiring unpublished caches.

No normalization, split fitting, gene-name guessing, or endpoint pseudo-pairing.
--gene-axis projects onto supplied coordinates; --reference-gene-axis separately
checks an actually available frozen reference. --gene-axis-manifest binds the
supplied axis bytes to a source receipt. Neither certifies model compatibility.
Missing genes remain NaN.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from lung.data.external_gene_axis import align_gene_axis
from lung.data.source_bridge import axis_identity, read_gene_axis, verify_axis_reference
from scripts.align_sarin_state_expression import (
    mapping_ledger, read_transcripts, transcript_ids, aligned_matrix,
)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_soft(path):
    """Preserve repeated GEO metadata fields and source ordering."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        text = f.read()
    samples = []
    for block in re.split(r"(?=\^SAMPLE = )", text)[1:]:
        metadata = defaultdict(list)
        for line in block.splitlines():
            if line.startswith("!Sample_") and " = " in line:
                key, value = line.split(" = ", 1)
                metadata[key[8:]].append(value)
        accession = block.splitlines()[0].split(" = ", 1)[1]
        samples.append((accession, dict(metadata), block))
    if not samples or len({s[0] for s in samples}) != len(samples):
        raise ValueError("Missing or duplicate GEO samples")
    return text, samples


def table_block(text, kind):
    start, end = f"!{kind}_table_begin\n", f"!{kind}_table_end"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one {kind} table")
    return pd.read_csv(StringIO(text.split(start)[1].split(end)[0]), sep="\t",
                       dtype=str, keep_default_na=False)


def sarin_expression(soft):
    text, records = read_soft(soft)
    platform = table_block(text.split("^SAMPLE = ", 1)[0], "platform")
    if not platform.ID.is_unique:
        raise ValueError("Duplicate platform probes")
    expression, rows = None, []
    for accession, meta, block in records:
        table = table_block(block, "sample")
        if not table.ID_REF.is_unique:
            raise ValueError("Duplicate sample probes")
        if expression is None:
            expression = table[["ID_REF"]].copy()
        if table.ID_REF.tolist() != expression.ID_REF.tolist():
            raise ValueError("Probe order differs between samples")
        expression[accession] = pd.to_numeric(table.VALUE, errors="raise")
        chars = dict(v.split(": ", 1) for v in meta["characteristics_ch1"])
        if chars["sensitivity"] not in {"sensitive", "resistant"}:
            raise ValueError("Unrecognized A549 state")
        protocol = " ".join(" ".join(meta.get("treatment_protocol_ch1", [])).split())
        if meta.get("platform_id") != ["GPL17077"] or "treated with cisplatin for 24 h" not in protocol:
            raise ValueError("Unverified Sarin platform or treatment duration")
        dose = float(chars["treatment conc"])
        if dose not in {0, 11, 34}:
            raise ValueError("Unverified cisplatin dose")
        rows.append({"geo_accession": accession, "sample_title": meta["title"][0],
                     "state_id": "A549" if chars["sensitivity"] == "sensitive" else "A549rCDDP2000",
                     "dose_uM": dose, "treatment_h": 24, "drug": "cisplatin" if dose else "control",
                     "reference_role": "parallel_24h_control" if dose == 0 else "treated_24h",
                     "observed_initial_state": False, "paired_to_function_replicate": False,
                     "batch_id": None, "replicate_identity": "source_GSM_not_cross_assay_pair"})
    annotation = platform.set_index("ID").loc[expression.ID_REF].reset_index()
    annotation = annotation.rename(columns={annotation.columns[0]: "probe_id"})
    return expression, pd.DataFrame(rows), annotation, records


def sarin_annexin(docx):
    """Read original S8 Fig5 cells, including blanks; never zip unrelated repeats."""
    with zipfile.ZipFile(docx) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    candidates = []
    for table in root.findall(".//w:tbl", ns):
        rows = [["".join(c.itertext()).strip() for c in r.findall("./w:tc", ns)]
                for r in table.findall("./w:tr", ns)]
        if any(r == ["FITC-Annexin [fold change to untreated control]"] for r in rows):
            candidates.append(rows)
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely identify original Fig5 Annexin table")
    rows = candidates[0]
    if rows[1] != ["Treatment concentration", "11 µM", "11 µM", "34 µM"]:
        raise ValueError("Fig5 dose columns changed")
    start = rows.index(["FITC-Annexin [fold change to untreated control]"]) + 1
    result = []
    for row_number, row in enumerate(rows[start:], start + 1):
        if row[0] == "Mean (SEM)":
            break
        if len(row) != 4:
            raise ValueError("Unexpected Annexin data row")
        for column, (state, dose) in enumerate([("A549", 11), ("A549rCDDP2000", 11), ("A549rCDDP2000", 34)], 1):
            if row[column] == "":
                continue
            value = float(row[column])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid Annexin fold value")
            result.append({"state_id": state, "drug": "cisplatin", "dose_uM": dose,
                           "treatment_h": 24, "value": value,
                           "unit": "fold_change_to_respective_untreated_control",
                           "source_table_row_1based": row_number, "source_table_column_1based": column + 1,
                           "cross_condition_replicate_pairing": "not_established",
                           "paired_to_RNA_sample": False})
    return pd.DataFrame(result)


def hm_expression(matrix, soft):
    frame = pd.read_csv(matrix, sep="\t")
    _, records = read_soft(soft)
    mapping = {}
    for accession, meta, _ in records:
        title = meta["title"][0]
        match = re.fullmatch(r"(A549|H841|SBC5)_(Notreatment|HM97662)_rep([123])", title)
        if not match:
            raise ValueError(f"Unrecognized HM97662 sample: {title}")
        cell, treatment, rep = match.groups()
        protocol = " ".join(" ".join(meta.get("treatment_protocol_ch1", [])).split())
        if protocol != "Cells were treated with 1 μM HM97662 or left untreated for 6 days.":
            raise ValueError("Unverified HM97662 dose or exposure protocol")
        column = f"{cell}_{'NT' if treatment == 'Notreatment' else 'HM'}-{rep}"
        if column in mapping:
            raise ValueError("Duplicate source-to-GSM identity")
        mapping[column] = {"matrix_column_id": column, "geo_accession": accession,
                           "cell_line": cell, "treatment": treatment, "dose_uM": 0 if treatment == "Notreatment" else 1,
                           "exposure_h": 144, "sample_time_h": 144, "replicate_label": rep,
                           "batch_id": None, "paired_to_function_replicate": False,
                           "observed_initial_state": False}
    columns = [c for c in frame if c not in {"gene_id", "gene_name"}]
    if set(columns) != set(mapping):
        raise ValueError("Matrix and GEO samples differ")
    if not frame.gene_id.is_unique or not frame.gene_id.str.fullmatch(r"ENSG\d+(?:\.\d+)?").all():
        raise ValueError("Invalid HM Ensembl gene identity")
    return frame, pd.DataFrame([mapping[c] for c in columns]), records


def history_expression(matrix, soft):
    """GSE96779 is bulk FPKM; repeated symbols/loci stay distinct source rows."""
    frame = pd.read_csv(matrix, sep="\t")
    empty_columns = [c for c in frame if c.startswith("Unnamed:")]
    if any(frame[c].notna().any() for c in empty_columns):
        raise ValueError("Unexpected nonempty unnamed column")
    frame = frame.drop(columns=empty_columns)
    _, records = read_soft(soft)
    prefixes = {"A549 cells parental": "Parental", "MSR-A549 cells": "MSR",
                "MSR-A549 cells drugs withdrawal": "Withdrawn", "MSR-A549 cells with JQ1 treatment": "JQ1"}
    treatments = {"Parental": "DMSO", "MSR": "5uM momelotinib and 1uM selumetinib",
                  "Withdrawn": "DMSO", "JQ1": "5uM momelotinib, 1uM selumetinib and 200nM JQ1"}
    mapping = {}
    for accession, meta, _ in records:
        title, rep = meta["title"][0].rsplit(" ", 1)
        if title not in prefixes or rep not in {"1", "2", "3"}:
            raise ValueError("Unverified history sample identity")
        group = prefixes[title]
        protocol = " ".join(" ".join(meta.get("treatment_protocol_ch1", [])).split())
        if "200nM JQ1 for 24 hours" not in protocol or "for 10 days" not in protocol:
            raise ValueError("Unverified history exposure/withdrawal protocol")
        chars = dict(v.split(": ", 1) for v in meta["characteristics_ch1"])
        if chars["drugs treated"] != treatments[group]:
            raise ValueError("History co-treatment changed")
        column = f"{group}_{rep}_FPKM"
        if column in mapping:
            raise ValueError("Duplicate history sample")
        mapping[column] = {"matrix_column_id": column, "geo_accession": accession, "cell_line": "A549",
                           "state_id": group, "replicate_label": rep, "batch_id": None,
                           "co_treatment": treatments[group], "history": "parental" if group == "Parental" else "2_month_MMB_Selu_selection",
                           "JQ1_exposure_h": 24 if group == "JQ1" else None,
                           "withdrawal_h": 240 if group == "Withdrawn" else None,
                           "total_exposure_h": None, "paired_to_function_replicate": False,
                           "observed_initial_state": False}
    columns = [c for c in frame if c not in {"gene_id", "locus"}]
    if set(columns) != set(mapping):
        raise ValueError("History matrix and GEO samples differ")
    # gene_id is a symbol, not an Ensembl ID. Row+symbol+locus preserve ambiguous duplicates.
    frame["source_feature_id"] = [f"row:{i}|{gene}|{locus}" for i, (gene, locus) in enumerate(zip(frame.gene_id, frame.locus))]
    return frame, pd.DataFrame([mapping[c] for c in columns]), records


def pulse_expression(matrix, extra_matrix, soft, paper_xml):
    """Recover GSE161800 counts and resolve its documented mg/µg discrepancy."""
    paper = " ".join(" ".join(ET.parse(paper_xml).getroot().itertext()).split())
    if not re.search(r"For cell sorting and RNAseq analysis.{0,130}5 [μµ]g/ml.{0,50}2 hr", paper):
        raise ValueError("Primary RNAseq protocol does not support the dose resolution")
    frames = [pd.read_csv(p, dtype={"gene.id": str}) for p in (matrix, extra_matrix)]
    for frame in frames:
        if not frame["gene.id"].is_unique or not frame["gene.id"].str.fullmatch(r"ENSG\d+").all():
            raise ValueError("Invalid pulse source gene identities")
    if set(frames[0]["gene.id"]) != set(frames[1]["gene.id"]):
        raise ValueError("Pulse count tables have different gene supports")
    columns = [[c for c in frame if c.startswith("AGR_")] for frame in frames]
    if set(columns[0]) & set(columns[1]):
        raise ValueError("Overlapping pulse count sample columns")
    frame = frames[0][["gene.id"] + columns[0]].copy()
    # Tables are DE-sorted differently. Join explicitly by gene, never by row number.
    reordered = frames[1].set_index("gene.id").loc[frame["gene.id"], columns[1]]
    for column in columns[1]:
        frame[column] = reordered[column].to_numpy()
    _, records = read_soft(soft)
    mapping = {}
    for accession, meta, _ in records:
        column = meta["description"][0]
        if column in mapping:
            raise ValueError("Duplicate AGR-to-GSM binding")
        chars = dict(v.split(": ", 1) for v in meta["characteristics_ch1"])
        protocol = " ".join(" ".join(meta["treatment_protocol_ch1"]).split())
        if chars["cell line"] != "A549" or "cisplatin (5 mg/mL) for 2 hr" not in protocol:
            raise ValueError("Unrecognized pulse source protocol; re-review resolution")
        control = "Control" in meta["title"][0]
        mapping[column] = {"matrix_column_id": column, "geo_accession": accession,
                           "cell_line": "A549", "sample_title": meta["title"][0],
                           "phenotype": chars["phenotype"], "sorting": chars["facs sorting"],
                           "drug": "control" if control else "cisplatin",
                           "paper_dose_ug_per_ml": 0 if control else 5,
                           "GEO_protocol_dose_text": "5 mg/mL",
                           "dose_resolution": "primary_RNAseq_methods_5_ug_ml_overrides_GEO_5_mg_ml_conflict",
                           "exposure_h": 2, "recovery_h": 72, "batch_id": None,
                           "observed_initial_state": False, "paired_to_function_replicate": False}
    all_columns = columns[0] + columns[1]
    if set(all_columns) != set(mapping):
        raise ValueError("Pulse matrices do not cover exact GEO sample identities")
    return frame, pd.DataFrame([mapping[c] for c in all_columns]), records


def brd9_history_expression(directory, soft):
    """Read four source histories without inventing a mid-treatment RNA state."""
    _, records = read_soft(soft)
    if len(records) != 12:
        raise ValueError("GSE178640 requires 12 registered libraries")
    rows, series, paths = [], [], []
    reference = None
    for accession, meta, _ in records:
        chars = meta.get("characteristics_ch1", [])
        if "cell line: A549" not in chars:
            raise ValueError("A549 background required")
        treatment = next((x.split(": ", 1)[1] for x in chars if x.startswith("treatment: ")), None)
        allowed = {f"{a} pretreated {b}" for a in ["dBRD9-A", "DMSO"] for b in ["plus IFN-alpha2", "no IFN"]}
        if treatment not in allowed:
            raise ValueError("Unreviewed history category")
        protocol = " ".join(meta.get("treatment_protocol_ch1", []))
        for token in ["125nM", "for 6h", "containing again", "100IU/mL", "incubated for 6h"]:
            if token not in protocol:
                raise ValueError("Reviewed exposure/retreatment changed")
        urls = meta.get("supplementary_file_1", [])
        if len(urls) != 1:
            raise ValueError("Unique per-sample count file required")
        path = directory / urls[0].rsplit("/", 1)[1]
        frame = pd.read_csv(path, sep="\t")
        if list(frame.columns) != ["Identifier", "matchCounts"] or not frame.Identifier.is_unique:
            raise ValueError("Unique source gene/count columns required")
        if not frame.Identifier.str.fullmatch(r"ENSG\d{11}").all():
            raise ValueError("Source Ensembl identity changed")
        if reference is None:
            reference = frame.Identifier.tolist()
        if set(frame.Identifier) != set(reference):
            raise ValueError("Sample gene universes differ")
        values = frame.set_index("Identifier").loc[reference, "matchCounts"]
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Raw counts must be finite nonnegative integers")
        series.append(values.rename(accession));paths.append(path)
        first = "dBRD9-A" if treatment.startswith("dBRD9-A") else "DMSO"
        has_ifn = "plus IFN-alpha2" in treatment
        rows.append(dict(geo_accession=accession, matrix_column_id=accession,
            source_title=meta["title"][0], cell_background="A549", source_history=treatment,
            first_agent=first, first_dose_nM=125 if first=="dBRD9-A" else None,
            first_start_h=0, media_refresh_h=6, first_agent_readded=True, first_total_exposure_h=12,
            second_agent="IFN-alpha2" if has_ifn else None,
            second_dose_IU_per_mL=100 if has_ifn else None,
            second_dose_uM=None, second_start_h=6 if has_ifn else None,
            second_exposure_h=6 if has_ifn else None, endpoint_h=12,
            intermediate_RNA_measured=False, paired_repeat_id=None,
            small_molecule_only_candidate_eligible=False, apoptosis_endpoint_available=False,
            source_protocol=protocol))
    samples=pd.DataFrame(rows)
    if not samples.groupby("source_history").size().eq(3).all() or samples.source_history.nunique()!=4:
        raise ValueError("Four source histories with three samples each required")
    result=pd.concat(series,axis=1).reset_index().rename(columns={"Identifier":"ensembl_id"})
    return result, samples, records, paths


def save_native(out, frame, samples, ids, scale):
    columns = samples.matrix_column_id.tolist() if "matrix_column_id" in samples else samples.geo_accession.tolist()
    if not frame.columns.is_unique or len(set(columns)) != len(columns) or not samples.geo_accession.is_unique:
        raise ValueError("Duplicate sample identity or expression column")
    if not columns or samples.geo_accession.isna().any() or samples.geo_accession.astype(str).str.strip().eq("").any():
        raise ValueError("Missing source sample identity")
    values = frame[columns].to_numpy(dtype=np.float64).T
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source measurements")
    if scale == "bulk_raw_counts" and ((values < 0).any() or (values != np.floor(values)).any()):
        raise ValueError("Raw counts must be nonnegative integers")
    if scale in {"bulk_estimated_counts", "bulk_FPKM"} and (values < 0).any():
        raise ValueError("Negative estimated counts")
    np.savez_compressed(out / "source_expression.npz", expression=values,
                        feature_ids=frame[ids].to_numpy(dtype=str), sample_ids=samples.geo_accession.to_numpy(dtype=str),
                        observed_mask=np.isfinite(values), value_scale=scale)
    with np.load(out / "source_expression.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["expression"], values)
        np.testing.assert_array_equal(saved["sample_ids"], samples.geo_accession.to_numpy(dtype=str))
    samples.to_csv(out / "samples.csv", index=False)
    return {"feature_count": len(frame), "sample_count": len(samples), "measurement_values": int(values.size),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "noninteger_values": int(np.count_nonzero(values != np.floor(values))),
            "roundtrip_exact": True, "scale": scale}


def load_projection_axis(path, reference_manifest=None):
    """Separate verified projection coordinates from frozen-training identity."""
    axis = read_gene_axis(path)
    scope = {"axis_sha256": digest(path), "source_reference_verified": False,
             "frozen_asset_identity_verified": False, "reference_manifest_sha256": None}
    if reference_manifest is not None:
        reference = json.loads(Path(reference_manifest).read_text(encoding="utf-8"))
        expected = reference["outputs"]["genes.parquet"]
        if (expected["sha256"] != scope["axis_sha256"] or expected["bytes"] != Path(path).stat().st_size
                or reference["validation"]["shared_genes"] != len(axis)):
            raise ValueError("Reference axis manifest does not match the supplied axis")
        scope.update(source_reference_verified=True, reference_manifest_sha256=digest(reference_manifest))
        # This source manifest attests bytes, not unavailable training assets.
    return axis, scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["sarin", "hm97662", "gse96779", "gse161800", "gse178640"])
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--annexin-docx", type=Path)
    parser.add_argument("--extra-matrix", type=Path, help="GSE161800 7vs8 original count table")
    parser.add_argument("--paper-xml", type=Path, help="GSE161800 primary paper XML for unit adjudication")
    parser.add_argument("--gene-axis", type=Path, help="Supplied CSV/Parquet coordinates, not automatically a frozen asset")
    parser.add_argument("--reference-gene-axis", type=Path, help="Optional actual frozen CSV/Parquet for explicit identity/order comparison")
    parser.add_argument("--gene-axis-manifest", type=Path, help="Source reference manifest; does not authenticate frozen training assets")
    parser.add_argument("--hgnc", type=Path, help="Versioned HGNC TSV for Sarin probe mapping")
    parser.add_argument("--gtf", type=Path, help="Versioned GRCh37 transcript GTF.gz for Sarin")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; frozen data are never overwritten")
    if args.gene_axis_manifest and not args.gene_axis:
        parser.error("--gene-axis-manifest requires --gene-axis")
    # Reference mismatch is rejected before any output or source matrix is read.
    axis, axis_provenance = load_projection_axis(args.gene_axis, args.gene_axis_manifest) if args.gene_axis else (None, None)
    reference_check = {"reference_axis_match_verified": False, "model_compatibility_verified": False}
    if args.reference_gene_axis:
        if axis is None:
            parser.error("--reference-gene-axis requires --gene-axis")
        reference_check = verify_axis_reference(axis, read_gene_axis(args.reference_gene_axis))
    # Validate source before creating output, so failures cannot look like completed recovery.
    paths = [args.soft]
    if args.reference_gene_axis:
        paths.append(args.reference_gene_axis)
    if args.source == "sarin":
        if not args.annexin_docx:
            parser.error("Sarin requires --annexin-docx")
        if args.gene_axis and not (args.hgnc and args.gtf):
            parser.error("Sarin axis projection requires --hgnc and --gtf; gene symbols alone are insufficient")
        frame, samples, annotation, records = sarin_expression(args.soft)
        endpoints = sarin_annexin(args.annexin_docx)
        paths.append(args.annexin_docx)
        identifier, scale = "ID_REF", "deposited_background_corrected_quantile_normalized_microarray_VALUE"
    elif args.source == "gse178640":
        if not args.matrix or not args.matrix.is_dir():
            parser.error("GSE178640 --matrix is the directory of 12 original count files")
        frame, samples, records, count_paths = brd9_history_expression(args.matrix, args.soft)
        paths.extend(count_paths)
        identifier, scale = "ensembl_id", "bulk_raw_counts"
    elif args.source == "gse161800":
        if not (args.matrix and args.extra_matrix and args.paper_xml):
            parser.error("GSE161800 requires two count tables and --paper-xml for unit adjudication")
        frame, samples, records = pulse_expression(args.matrix, args.extra_matrix, args.soft, args.paper_xml)
        paths.extend([args.matrix, args.extra_matrix, args.paper_xml])
        identifier, scale = "gene.id", "bulk_raw_counts"
    elif args.source == "gse96779":
        if not args.matrix or args.gene_axis:
            parser.error("GSE96779 requires --matrix; historical symbol mapping requires a separate verified bridge")
        frame, samples, records = history_expression(args.matrix, args.soft)
        paths.append(args.matrix)
        identifier, scale = "source_feature_id", "bulk_FPKM"
    else:
        if not args.matrix:
            parser.error("HM97662 requires --matrix")
        frame, samples, records = hm_expression(args.matrix, args.soft)
        paths.append(args.matrix)
        identifier, scale = "gene_id", "bulk_estimated_counts"
    samples["data_role"] = "unassigned_external_source"
    args.output.mkdir(parents=True)
    report = save_native(args.output, frame, samples, identifier, scale)
    if args.source == "sarin":
        annotation.to_csv(args.output / "probe_annotation.csv.gz", index=False,
                          compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
        frame.to_csv(args.output / "probe_expression.csv.gz", index=False,
                     compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
        endpoints.to_csv(args.output / "annexin_source_values.csv", index=False)
        # A condition-level many-to-many link, NEVER replicate matching or Cartesian pseudo-observations.
        links = []
        for (state, dose), group in endpoints.groupby(["state_id", "dose_uM"]):
            match = samples.loc[samples.state_id.eq(state) & samples.dose_uM.eq(dose)]
            control = samples.loc[samples.state_id.eq(state) & samples.dose_uM.eq(0)]
            if match.empty or control.empty:
                raise ValueError("Missing RNA treatment/control condition")
            links.append({"state_id": state, "dose_uM": float(dose), "treatment_h": 24,
                          "RNA_GSMs": match.geo_accession.tolist(), "parallel_control_GSMs": control.geo_accession.tolist(),
                          "endpoint_n": len(group), "endpoint_mean": float(group.value.mean()),
                          "endpoint_sem": float(group.value.std(ddof=1) / np.sqrt(len(group))),
                          "connection": "cross_study_condition_reference_not_paired_measurement",
                          "initial_state_RNA_measured": False,
                          "known_confound": "Annexin seeding density: parental 250000 vs resistant 500000 cells/well"})
        (args.output / "condition_links.json").write_text(json.dumps(links, indent=2))
        report.update(endpoint_values=len(endpoints), condition_links=len(links), endpoint_unit=endpoints.unit.iloc[0])
    if args.source == "gse96779":
        frame[["source_feature_id", "gene_id", "locus"]].to_csv(args.output / "source_features.csv.gz", index=False)
        report.update(duplicate_symbol_rows=int(frame.gene_id.duplicated(keep=False).sum()),
                      repeated_symbols=int(frame.loc[frame.gene_id.duplicated(keep=False), "gene_id"].nunique()),
                      endpoint_link_verified=False, history_observed=True, arbitrary_sequence_support=False)
    if args.gene_axis:
        if args.source == "sarin":
            hgnc = pd.read_csv(args.hgnc, sep="\t", dtype=str, keep_default_na=False)
            wanted = set().union(*(set(transcript_ids(row)) for row in annotation.to_dict("records")))
            transcripts = read_transcripts(args.gtf, wanted)
            ledger = mapping_ledger(annotation, axis, hgnc, transcripts)
            aligned, mask = aligned_matrix(frame.iloc[:, 1:].to_numpy(dtype=np.float64), ledger, len(axis))
            paths.extend([args.hgnc, args.gtf])
        else:
            aligned, mask, ledger = align_gene_axis(frame.rename(columns={identifier: "ensembl_id"}), axis,
                identifier="ensembl_id", samples=samples.matrix_column_id.tolist(), scale=scale)
        np.savez_compressed(args.output / "common_axis_expression.npz", expression=aligned, observed_mask=mask,
                            sample_ids=samples.geo_accession.to_numpy(dtype=str),
                            ensembl_ids=axis.ensembl_id.to_numpy(dtype=str), value_scale=scale)
        with np.load(args.output / "common_axis_expression.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["expression"], aligned)
            np.testing.assert_array_equal(saved["observed_mask"], mask)
            np.testing.assert_array_equal(saved["sample_ids"], samples.geo_accession.to_numpy(dtype=str))
            np.testing.assert_array_equal(saved["ensembl_ids"], axis.ensembl_id.to_numpy(dtype=str))
            if saved["value_scale"].item() != scale:
                raise ValueError("Expression measurement scale changed")
            if not np.isnan(saved["expression"][~mask]).all():
                raise ValueError("Missing genes were filled")
        ledger.to_parquet(args.output / "gene_mapping.parquet", index=False)
        paths.append(args.gene_axis)
        if args.gene_axis_manifest:
            paths.append(args.gene_axis_manifest)
        report["common_axis_matched_genes"] = int(mask.all(axis=0).sum())
    report.update(reference_check)
    report["axis_projection_verified"] = axis is not None
    report["common_axis_identity"] = axis_identity(axis) if axis is not None else None
    report["gene_axis_provenance"] = axis_provenance
    # Legacy flag now requires explicit reference comparison, not merely a file argument.
    report.update(common_axis_verified=reference_check["reference_axis_match_verified"], same_cell_pre_post_pairing=False,
                  normalization_performed=False, training_performed=False, model_input_ready=False, full_research_feasibility_proven=False)
    (args.output / "source_metadata.json").write_text(json.dumps({a: m for a, m, _ in records}, ensure_ascii=False, indent=2))
    manifest = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": args.source, "validation": report,
                "sources": [{"path": str(p.resolve()), "sha256": digest(p), "bytes": p.stat().st_size} for p in paths],
                "script_sha256": digest(__file__),
                "outputs": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
