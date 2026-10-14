"""Frozen, conservative unique-intron annotation; no reads or model execution.

UCSC genePredExtended coordinates are zero-based half-open. All source exons
are excluded irrespective of strand, then segments belonging to multiple NCBI
genes are excluded. This is a new processing definition, not an exact recovery
of the unknown historical annotation used by Bothe et al.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PRIMARY = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Transcript:
    accession: str
    chrom: str
    strand: str
    start: int
    end: int
    exons: tuple[tuple[int, int], ...]
    symbol: str

    @property
    def length(self) -> int:
        return sum(end - start for start, end in self.exons)

    @property
    def introns(self) -> tuple[tuple[int, int], ...]:
        return tuple((a[1], b[0]) for a, b in zip(self.exons, self.exons[1:]) if a[1] < b[0])


def parse_genepred(line: str) -> Transcript:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 16:
        raise ValueError("genePredExtended must have exactly 16 fields including bin")
    accession, chrom, strand = fields[1:4]
    start, end, cds_start, cds_end, count = map(int, fields[4:9])
    starts = tuple(int(x) for x in fields[9].rstrip(",").split(","))
    ends = tuple(int(x) for x in fields[10].rstrip(",").split(","))
    frames = tuple(int(x) for x in fields[15].rstrip(",").split(","))
    int(fields[0]); int(fields[11])
    if not accession or not chrom or strand not in ("+", "-"):
        raise ValueError("invalid accession/chromosome/strand")
    if count < 1 or len(starts) != count or len(ends) != count or len(frames) != count:
        raise ValueError("exonCount does not match exonStarts/exonEnds/exonFrames")
    if not (0 <= start < end and start <= cds_start <= cds_end <= end):
        raise ValueError("invalid transcript/CDS bounds")
    exons = tuple(zip(starts, ends))
    if starts[0] != start or ends[-1] != end:
        raise ValueError("transcript bounds must match outer exons")
    if any(not (start <= a < b <= end) for a, b in exons):
        raise ValueError("invalid exon bounds")
    if any(a[1] > b[0] for a, b in zip(exons, exons[1:])):
        raise ValueError("exons must be sorted and nonoverlapping")
    if any(x not in (-1, 0, 1, 2) for x in frames):
        raise ValueError("invalid exon frame")
    if fields[13] not in ("none", "unk", "incmpl", "cmpl") or fields[14] not in ("none", "unk", "incmpl", "cmpl"):
        raise ValueError("invalid CDS status")
    return Transcript(accession, chrom, strand, start, end, exons, fields[12])


def read_sources(source: Path):
    links = {}
    with gzip.open(source / "ncbiRefSeqLink.txt.gz", "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 19 or not fields[0]:
                raise ValueError(f"invalid ncbiRefSeqLink row {number}")
            gene = fields[6]
            if gene and (not gene.isdigit() or int(gene) <= 0):
                raise ValueError(f"invalid NCBI GeneID row {number}")
            value = (gene, fields[2])
            if fields[0] in links and links[fields[0]] != value:
                raise ValueError(f"conflicting link id row {number}")
            links[fields[0]] = value
    transcripts = []
    with gzip.open(source / "ncbiRefSeq.txt.gz", "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                transcripts.append(parse_genepred(line))
            except ValueError as exc:
                raise ValueError(f"ncbiRefSeq row {number}: {exc}") from exc
    return transcripts, links


def select_transcripts(transcripts, links):
    grouped = defaultdict(list)
    ledger = []
    for tx in transcripts:
        gene, symbol = links.get(tx.accession, ("", ""))
        reason = "outside_primary_scope" if tx.chrom not in PRIMARY else (
            "missing_link_id" if tx.accession not in links else "missing_gene_id" if not gene else "")
        if reason:
            ledger.append((gene, tx.accession, tx.chrom, tx.strand, reason))
        else:
            grouped[gene].append(tx)
    selected = {}
    for gene, variants in grouped.items():
        # Distinct chromosome/strand or disconnected transcript spans are not
        # silently merged. Overlapping spans form one locus (transitive union).
        locations = {(tx.chrom, tx.strand) for tx in variants}
        spans = sorted((tx.start, tx.end) for tx in variants)
        reach = spans[0][1]
        disconnected = False
        for start, end in spans[1:]:
            if start >= reach:
                disconnected = True
            reach = max(reach, end)
        reason = "primary_chromosome_or_strand_conflict" if len(locations) > 1 else (
            "disconnected_primary_loci" if disconnected else "")
        if reason:
            ledger.extend((gene, tx.accession, tx.chrom, tx.strand, reason) for tx in variants)
            continue
        chosen = min(variants, key=lambda tx: (-tx.length, tx.accession, tx.start, tx.end, tx.exons))
        selected[gene] = chosen
        ledger.extend((gene, tx.accession, tx.chrom, tx.strand,
                       "selected" if tx == chosen else "non_longest_or_tie_loser") for tx in variants)
    return selected, ledger


def unique_segments(transcripts, selected):
    """Disjoin sweep: exon blockers and active intronic gene owners, both strands."""
    events = defaultdict(lambda: defaultdict(list))
    for tx in transcripts:
        if tx.chrom in PRIMARY:
            for start, end in tx.exons:
                events[tx.chrom][start].append(("exon", 1))
                events[tx.chrom][end].append(("exon", -1))
    for gene, tx in selected.items():
        for start, end in tx.introns:
            events[tx.chrom][start].append((gene, 1))
            events[tx.chrom][end].append((gene, -1))
    retained = []
    stats = Counter()
    for chrom in sorted(events):
        exon_depth = 0
        owners = Counter()
        points = sorted(events[chrom])
        for index, start in enumerate(points[:-1]):
            for owner, delta in events[chrom][start]:
                if owner == "exon":
                    exon_depth += delta
                else:
                    owners[owner] += delta
                    if owners[owner] == 0:
                        del owners[owner]
            end = points[index + 1]
            if owners:
                stats["candidate_union_bp"] += end - start
                if exon_depth:
                    stats["excluded_exonic_bp"] += end - start
                elif len(owners) > 1:
                    stats["excluded_multiple_gene_bp"] += end - start
                else:
                    gene = next(iter(owners))
                    retained.append((chrom, start, end, gene, selected[gene].strand))
                    stats["retained_bp"] += end - start
        # Validate event balance, including the terminal position.
        for owner, delta in events[chrom][points[-1]]:
            if owner == "exon":
                exon_depth += delta
            else:
                owners[owner] += delta
                if owners[owner] == 0:
                    del owners[owner]
        if exon_depth != 0 or owners:
            raise ValueError("unbalanced interval sweep")
    return retained, dict(stats)


def write_tsv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def verify_readback(output, transcripts, selected):
    """Independent binary-search overlap/containment checks on serialized SAF."""
    from bisect import bisect_left, bisect_right
    blockers = defaultdict(list)
    candidates = defaultdict(list)
    for tx in transcripts:
        if tx.chrom in PRIMARY:
            blockers[tx.chrom].extend(tx.exons)
    for gene, tx in selected.items():
        candidates[tx.chrom].extend((a, b, gene) for a, b in tx.introns)
    # Prefix-max ends make arbitrary nested source intervals searchable.
    indexes = {}
    for chrom in PRIMARY:
        rows = sorted(blockers[chrom])
        starts = [a for a, _ in rows]
        maximum = []
        for _, end in rows:
            maximum.append(max(end, maximum[-1] if maximum else -1))
        indexes[chrom] = (starts, maximum)
    with (output / "feature_gene_mapping.tsv").open(encoding="utf-8") as handle:
        mapping = {row["feature_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    count, bases = 0, 0
    last = {}
    seen = set()
    with (output / "unique_introns.saf").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            feature, chrom = row["GeneID"], row["Chr"]
            start, end = int(row["Start"]) - 1, int(row["End"])
            if feature in seen or feature not in mapping or chrom not in PRIMARY or start < 0 or start >= end:
                raise ValueError("invalid SAF feature or coordinate")
            seen.add(feature)
            gene = mapping[feature]["ncbi_gene_id"]
            entry = mapping[feature]
            if (entry["chrom"] != chrom or entry["strand"] != row["Strand"]
                    or int(entry["start_0based"]) != start or int(entry["end_exclusive"]) != end
                    or int(entry["length_bp"]) != end - start):
                raise ValueError("feature mapping coordinates mismatch")
            tx = selected[gene]
            if tx.chrom != chrom or tx.strand != row["Strand"]:
                raise ValueError("feature mapping locus mismatch")
            if not any(a <= start < end <= b for a, b in tx.introns):
                raise ValueError("feature outside selected introns")
            if start < last.get(chrom, -1):
                raise ValueError("features overlap or are unsorted")
            last[chrom] = end
            starts, maximum = indexes[chrom]
            position = bisect_left(starts, end) - 1
            if position >= 0 and maximum[position] > start:
                raise ValueError("feature overlaps source exon")
            # Independent owner sweep is intentionally simple, indexed by starts.
            count += 1
            bases += end - start
    if seen != set(mapping):
        raise ValueError("mapping and SAF feature axes differ")
    # Verify ownership through a separate interval-event sweep, including SAF.
    owner_events = defaultdict(lambda: defaultdict(list))
    for chrom, rows in candidates.items():
        for a, b, gene in rows:
            owner_events[chrom][a].append((gene, 1))
            owner_events[chrom][b].append((gene, -1))
    for chrom, points in owner_events.items():
        active = Counter()
        ownership = []
        positions = sorted(points)
        for i, a in enumerate(positions[:-1]):
            for gene, delta in points[a]:
                active[gene] += delta
                if not active[gene]:
                    del active[gene]
            ownership.append((a, positions[i + 1], frozenset(active)))
        candidates[chrom] = ownership
    with (output / "unique_introns.saf").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chrom, a, b = row["Chr"], int(row["Start"]) - 1, int(row["End"])
            rows = candidates[chrom]
            # Cached ownership starts, distinct from exon index.
            key = "owners:" + chrom
            if key not in indexes:
                indexes[key] = [x[0] for x in rows]
            i = bisect_right(indexes[key], a) - 1
            gene = mapping[row["GeneID"]]["ncbi_gene_id"]
            while i < len(rows) and rows[i][0] < b:
                if i < 0 or rows[i][2] != frozenset([gene]):
                    raise ValueError("feature has ambiguous intronic gene ownership")
                i += 1
    return {"features_read": count, "bp_read": bases, "all_source_exons_disjoint": True,
            "unique_intronic_gene_ownership": True, "feature_mapping_axis_equal": True}


def build(source: Path, output: Path, receipt: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("refuse to overwrite a nonempty annotation version")
    transcripts, links = read_sources(source)
    selected, ledger = select_transcripts(transcripts, links)
    segments, stats = unique_segments(transcripts, selected)
    output.mkdir(parents=True, exist_ok=True)
    receipt.mkdir(parents=True, exist_ok=True)
    saf, mapping = [], []
    counts = Counter()
    lengths = Counter()
    for number, (chrom, start, end, gene, strand) in enumerate(segments, 1):
        feature = f"UIV1_{number:08d}"
        saf.append((feature, chrom, start + 1, end, strand))
        mapping.append((feature, gene, selected[gene].accession, links[selected[gene].accession][1], chrom, strand, start, end, end - start))
        counts[gene] += 1
        lengths[gene] += end - start
    write_tsv(output / "unique_introns.saf", ["GeneID", "Chr", "Start", "End", "Strand"], saf)
    write_tsv(output / "feature_gene_mapping.tsv", ["feature_id", "ncbi_gene_id", "selected_refseq", "symbol", "chrom", "strand", "start_0based", "end_exclusive", "length_bp"], mapping)
    genes = [(gene, tx.accession, links[tx.accession][1], tx.chrom, tx.strand, tx.start, tx.end,
              tx.length, len(tx.exons), len(tx.introns), counts[gene], lengths[gene],
              "retained" if counts[gene] else "no_source_intron" if not tx.introns else "all_intronic_bases_excluded")
             for gene, tx in sorted(selected.items(), key=lambda pair: int(pair[0]))]
    write_tsv(output / "gene_table.tsv", ["ncbi_gene_id", "selected_refseq", "symbol", "chrom", "strand", "tx_start_0based", "tx_end_exclusive", "spliced_length_bp", "exon_count", "source_intron_count", "retained_interval_count", "retained_intron_bp", "status"], genes)
    write_tsv(output / "transcript_scope_ledger.tsv", ["ncbi_gene_id", "refseq", "chrom", "strand", "reason"], ledger)
    validation = verify_readback(output, transcripts, selected)
    manifest = {"schema": "bothe-unique-introns-v1", "status": "annotation_built_and_readback_verified_counts_not_recovered",
                "method": {"source": "current UCSC hg19 ncbiRefSeq + ncbiRefSeqLink; historical author version unknown",
                           "coordinate_input": "0-based half-open", "saf": "1-based closed",
                           "scope": sorted(PRIMARY), "strand_policy": "both-strands exon subtraction and intronic uniqueness",
                           "selection": "longest sum(exon_end-exon_start); tie: lexicographic accession, start, end, exon tuple",
                           "locus_policy": "quarantine primary chromosome/strand conflicts and disconnected transcript-span loci; ignore alternate contigs for selection; preserve ledger",
                           "gene_id": "ncbiRefSeqLink.id joined to transcript accession; locusLinkId is stable NCBI GeneID",
                           "count_contract": "SAF GeneID is unique feature_id; featureCounts -f; downstream sum by feature_gene_mapping.ncbi_gene_id",
                           "equivalence": "new processing method; not exact author reproduction; strandness unconfirmed"},
                "source_files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(source.iterdir()) if p.is_file()},
                "generator_files": {str(Path(__file__).name): sha(Path(__file__))},
                "axes": {"source_transcript_rows": len(transcripts), "source_unique_accessions": len({t.accession for t in transcripts}),
                         "source_link_ids": len(links), "source_link_gene_ids": len({g for g, _ in links.values() if g}),
                         "primary_transcript_rows": sum(t.chrom in PRIMARY for t in transcripts),
                         "selected_genes": len(selected), "retained_genes": len(counts), "retained_features": len(saf)},
                "scope_reason_rows": dict(Counter(row[-1] for row in ledger)), "interval_stats": stats,
                "gene_status_counts": dict(Counter(row[-1] for row in genes)), "validation": validation,
                "output_files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}}
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (output / "manifest.json").write_text(payload, encoding="utf-8")
    (receipt / "annotation_build.json").write_text(payload, encoding="utf-8")
    return manifest
