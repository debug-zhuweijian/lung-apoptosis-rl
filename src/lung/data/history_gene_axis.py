"""Evidence-constrained hg19 symbol/locus to Ensembl projection of bulk FPKM.

No averaging, count conversion, expression-dependent mapping or axis-based
resolution of ambiguous biology. A coordinate projection is not assay alignment.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ENSG = re.compile(r"ENSG\d{11}")
CANONICAL_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}


def read_grch37_genes(path: Path) -> pd.DataFrame:
    """Read gene features, requiring the assembly declaration in the GTF itself."""
    rows, declarations = [], []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#"):
                if "genome-build " in line or "genome-version " in line:
                    declarations.append(line.strip().split()[-1])
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Invalid GTF record at line {line_number}")
            if fields[2] != "gene":
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]*)";', fields[8]))
            gene = attrs.get("gene_id", "")
            if not ENSG.fullmatch(gene):
                raise ValueError(f"Invalid annotation gene ID at line {line_number}")
            start, end = int(fields[3]), int(fields[4])
            if not 0 < start <= end:
                raise ValueError(f"Invalid GTF interval at line {line_number}")
            rows.append({"ensembl_id": gene, "gene_name": attrs.get("gene_name", ""),
                         "chromosome": fields[0], "start_1based": start,
                         "end_1based": end, "strand": fields[6],
                         "annotation_line_1based": line_number})
    if not declarations or any(re.fullmatch(r"GRCh37(?:\.p\d+)?", d) is None for d in declarations):
        raise ValueError("Explicit, consistent GRCh37 assembly declaration required")
    if not rows:
        raise ValueError("No GRCh37 gene features found")
    return pd.DataFrame(rows)


def align_history_fpkm(frame: pd.DataFrame, genes: pd.DataFrame, hgnc: pd.DataFrame,
                       axis_ids, *, samples: list[str], source_build: str):
    """Map only unique symbol AND genomic-locus supported measurements.

    GTF coordinates are 1-based inclusive. The source's browsing locus may be
    0-based half-open or 1-based inclusive; map only when BOTH interpretations
    select the identical candidate set. Boundary-sensitive rows stay missing.
    Historical HGNC previous symbols are allowed; general aliases are not used.
    Missing target genes retain NaN+False, including ambiguous/duplicate matches.
    """
    if source_build != "hg19":
        raise ValueError("This adapter requires verified hg19 source coordinates")
    axis = list(axis_ids)
    if not axis or len(set(axis)) != len(axis) or any(not isinstance(x, str) or not ENSG.fullmatch(x) for x in axis):
        raise ValueError("Target axis must contain unique canonical Ensembl gene IDs")
    if not samples or len(set(samples)) != len(samples):
        raise ValueError("Unique nonempty sample columns required")
    required = {"source_feature_id", "gene_id", "locus", *samples}
    if not required <= set(frame.columns) or not frame.columns.is_unique:
        raise ValueError("Missing/duplicate source feature or sample fields")
    if frame.empty:
        raise ValueError("Source feature table must be nonempty")
    if (frame.source_feature_id.isna().any() or frame.source_feature_id.duplicated().any()
            or frame.source_feature_id.astype(str).str.strip().eq("").any()):
        raise ValueError("Source feature identity must be nonmissing and unique")
    if not {"symbol", "prev_symbol", "ensembl_gene_id"} <= set(hgnc.columns):
        raise ValueError("HGNC approved and previous symbol evidence required")
    if not {"ensembl_id", "gene_name", "chromosome", "start_1based", "end_1based"} <= set(genes.columns):
        raise ValueError("Annotation coordinates and gene identity required")
    values = frame[samples].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("FPKM must be finite and nonnegative")

    # Keep competing annotations even when they fall outside the target axis.
    annotations = defaultdict(list)
    symbol_to_genes = defaultdict(set)
    evidence = defaultdict(set)
    for row in genes.to_dict("records"):
        gene, name = str(row["ensembl_id"]), str(row["gene_name"])
        if not ENSG.fullmatch(gene) or not 0 < int(row["start_1based"]) <= int(row["end_1based"]):
            raise ValueError("Invalid annotation identity or interval")
        annotations[gene].append(row)
        if name and name != "nan":
            symbol_to_genes[name].add(gene)
            evidence[(name, gene)].add("GRCh37_GTF_gene_name")
    for row in hgnc.fillna("").to_dict("records"):
        gene = str(row["ensembl_gene_id"])
        if not ENSG.fullmatch(gene):
            continue
        names = [(str(row["symbol"]), "HGNC_approved_symbol")]
        names += [(name, "HGNC_previous_symbol") for name in str(row["prev_symbol"]).split("|") if name]
        for name, why in names:
            if name:
                symbol_to_genes[name].add(gene)
                evidence[(name, gene)].add(why)

    positions = {gene: i for i, gene in enumerate(axis)}
    rows = []
    for index, source in enumerate(frame.to_dict("records")):
        symbol = "" if pd.isna(source["gene_id"]) else str(source["gene_id"]).strip()
        locus = "" if pd.isna(source["locus"]) else str(source["locus"]).strip()
        row = {"source_row": index, "source_feature_id": source["source_feature_id"],
               "source_symbol": symbol, "source_locus": locus,
               "candidate_ensembl_ids_json": "[]", "matched_ensembl_id": None,
               "common_gene_index": None, "identity_evidence": None,
               "mapping_status": "unsupported_source_locus"}
        match = re.fullmatch(r"chr([0-9]+|X|Y):(\d+)-(\d+)", locus)
        if not symbol or any(c in symbol for c in ",;|"):
            row["mapping_status"] = "missing_or_composite_symbol"
        elif not match or match[1] not in CANONICAL_CHROMS or int(match[2]) >= int(match[3]):
            pass
        else:
            chrom, start, end = match[1], int(match[2]), int(match[3])
            name_candidates = symbol_to_genes.get(symbol, set())
            choices = []
            for offset in (0, 1):
                left = max(1, start + offset)
                choices.append({gene for gene in name_candidates
                                if any(str(a["chromosome"]) == chrom
                                       and max(left, int(a["start_1based"])) <= min(end, int(a["end_1based"]))
                                       for a in annotations.get(gene, []))})
            union = choices[0] | choices[1]
            row["candidate_ensembl_ids_json"] = json.dumps(sorted(union))
            if choices[0] != choices[1]:
                row["mapping_status"] = "coordinate_convention_sensitive"
            elif len(union) > 1:
                row["mapping_status"] = "ambiguous_symbol_locus"
            elif not union:
                row["mapping_status"] = "no_symbol_locus_agreement"
            else:
                gene = next(iter(union))
                row["matched_ensembl_id"] = gene
                row["identity_evidence"] = ";".join(sorted(evidence[(symbol, gene)])) + ";same_chr_overlap_both_coordinate_conventions"
                row["mapping_status"] = "matched" if gene in positions else "not_in_target_axis"
                row["common_gene_index"] = positions.get(gene)
        rows.append(row)
    ledger = pd.DataFrame(rows)
    eligible = ledger.mapping_status.eq("matched")
    repeated = ledger.loc[eligible, "matched_ensembl_id"].duplicated(keep=False)
    ledger.loc[repeated.index[repeated], "mapping_status"] = "duplicate_source_target"
    ledger.loc[ledger.mapping_status.ne("matched"), "common_gene_index"] = np.nan
    aligned = np.full((len(samples), len(axis)), np.nan, dtype=np.float64)
    for row in ledger.loc[ledger.mapping_status.eq("matched")].itertuples():
        aligned[:, int(row.common_gene_index)] = values[int(row.source_row)]
    return aligned, np.isfinite(aligned), ledger
