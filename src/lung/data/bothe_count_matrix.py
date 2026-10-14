"""Strict Bothe featureCounts boundary. Real execution qualification is pending.

Audit fixtures never confer RNA/model eligibility. Missing measurements use empty
cells, while integer zero is reserved for an observed, measurable gene.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SAMPLES = ROOT / "experiments/20261011-data-alignment-round17/bothe_protocol_source/raw_sample_mapping.json"
ANNOTATION = ROOT / "08-datasets/Bothe2021/E-MTAB-9923/reference/unique-introns-v1"
SAMPLE_HASH = "eac6bced59a1c41e3ecea8311590edc89e6c288ba591cd9587e9bccc31e86ec0"
ANNOTATION_HASH = "c50c69db266c8de1b5ce91f8f0aa364cf2cffbb5a7e185625314037f6200e9c2"
MAX_COUNT = 2**63 - 1


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("invalid TSV header")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("ragged TSV")
    return rows


def _integer(value):
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError("count/coordinate must be nonnegative integer text")
    number = int(value)
    if number > MAX_COUNT:
        raise ValueError("integer overflow")
    return number


def sample_axis():
    if sha(SAMPLES) != SAMPLE_HASH:
        raise ValueError("frozen Round17 sample mapping changed")
    rows = _json(SAMPLES)
    if len(rows) != 12 or len({r["sample"] for r in rows}) != 12 or len({r["run"] for r in rows}) != 12:
        raise ValueError("invalid 12-sample axis")
    if sorted(Counter(r["history"] for r in rows).values()) != [3, 3, 3, 3]:
        raise ValueError("history replicate axis changed")
    return rows


def annotation_axis(directory=ANNOTATION, *, frozen=True):
    directory = Path(directory).resolve()
    manifest = _json(directory / "manifest.json")
    if frozen and (directory != ANNOTATION.resolve() or sha(directory / "manifest.json") != ANNOTATION_HASH):
        raise ValueError("requires frozen Round18 annotation")
    for name in ("unique_introns.saf", "feature_gene_mapping.tsv", "gene_table.tsv"):
        if sha(directory / name) != manifest["output_files"][name]["sha256"]:
            raise ValueError("annotation output binding changed")
    genes = _rows(directory / "gene_table.tsv")
    gene_ids = [r["ncbi_gene_id"] for r in genes]
    gene_set = set(gene_ids)
    if len(set(gene_ids)) != len(genes):
        raise ValueError("duplicate gene")
    mapping_rows = _rows(directory / "feature_gene_mapping.tsv")
    mapping = {r["feature_id"]: r for r in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("duplicate feature mapping")
    features = {}
    counts, lengths = Counter(), Counter()
    for row in _rows(directory / "unique_introns.saf"):
        feature = row["GeneID"]
        if feature in features or feature not in mapping:
            raise ValueError("duplicate/unmapped SAF feature")
        start, end = _integer(row["Start"]), _integer(row["End"])
        entry = mapping[feature]
        gene = entry["ncbi_gene_id"]
        if start < 1 or end < start or row["Strand"] not in ("+", "-") or gene not in gene_set:
            raise ValueError("invalid feature coordinate/gene")
        expected = (row["Chr"], str(start), str(end), row["Strand"], str(end-start+1))
        mapped = (entry["chrom"], str(_integer(entry["start_0based"])+1),
                  entry["end_exclusive"], entry["strand"], entry["length_bp"])
        if expected != mapped:
            raise ValueError("SAF mapping coordinate mismatch")
        features[feature] = (gene, expected)
        counts[gene] += 1
        lengths[gene] += end-start+1
    if set(features) != set(mapping):
        raise ValueError("missing SAF features")
    for row in genes:
        gene = row["ncbi_gene_id"]
        if (_integer(row["retained_interval_count"]) != counts[gene]
                or _integer(row["retained_intron_bp"]) != lengths[gene]
                or (row["status"] == "retained") != bool(counts[gene])
                or row["status"] not in ("retained", "no_source_intron", "all_intronic_bases_excluded")):
            raise ValueError("gene measurable status/count mismatch")
    if frozen and (len(genes), len(features), len(counts)) != (29634, 207555, 23125):
        raise ValueError("frozen annotation cardinalities changed")
    return genes, features


def parse_featurecounts(path, column_samples, features):
    """column_samples is audit-only. Real mappings must come from future executor."""
    if not column_samples or len(set(column_samples.values())) != len(column_samples):
        raise ValueError("empty/duplicate sample columns")
    with Path(path).open("rb") as stream:
        # Native Windows Program comments can contain CP936 executable paths.
        # Skip comments before decoding; header/data remain strict UTF-8.
        reader = csv.reader((line.decode("utf-8") for line in stream
                             if not line.startswith(b"#")), delimiter="\t")
        header = next(reader, [])
        if header[:6] != ["Geneid", "Chr", "Start", "End", "Strand", "Length"]:
            raise ValueError("invalid featureCounts metadata header")
        if len(set(header)) != len(header) or set(header[6:]) != set(column_samples):
            raise ValueError("unbound featureCounts sample column")
        totals = {sample: Counter() for sample in column_samples.values()}
        seen = set()
        for row in reader:
            if len(row) != len(header) or row[0] not in features or row[0] in seen:
                raise ValueError("duplicate/unknown/ragged featureCounts row")
            seen.add(row[0])
            gene, expected = features[row[0]]
            if tuple(row[1:6]) != expected:
                raise ValueError("featureCounts annotation metadata mismatch")
            for column, value in zip(header[6:], row[6:]):
                sample = column_samples[column]
                totals[sample][gene] += _integer(value)
                if totals[sample][gene] > MAX_COUNT:
                    raise ValueError("gene count overflow")
    if seen != set(features):
        raise ValueError("missing featureCounts rows")
    return totals


def _matrix(annotation, count_inputs):
    samples = sample_axis()
    genes, features = annotation_axis(annotation, frozen=False)
    allowed = {r["sample"] for r in samples}
    totals = {}
    for item in count_inputs:
        if not set(item["column_samples"].values()) <= allowed:
            raise ValueError("unknown sample")
        parsed = parse_featurecounts(item["path"], item["column_samples"], features)
        if set(totals) & set(parsed):
            raise ValueError("sample supplied more than once")
        totals.update(parsed)
    rows = []
    for gene in genes:
        key = gene["ncbi_gene_id"]
        rows.append([key, gene["status"]] + [str(totals[s["sample"]][key])
                    if gene["status"] == "retained" and s["sample"] in totals else ""
                    for s in samples])
    return samples, rows, sorted(totals)


def build_real(*args, **kwargs):
    """No caller-provided RNA/hash/receipt can authenticate real execution."""
    raise ValueError("pending_verified_count_executor: real build is not implemented")


def build_audit_mock(output, count_inputs, *, annotation=ANNOTATION):
    output = Path(output)
    if output.exists():
        raise ValueError("refuse to overwrite existing matrix version")
    annotation = Path(annotation).resolve()
    inputs = [{"path": str(Path(i["path"]).resolve()), "column_samples": dict(i["column_samples"])}
              for i in count_inputs]
    samples, rows, available = _matrix(annotation, inputs)
    output.mkdir(parents=True)
    with (output / "counts.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["ncbi_gene_id", "annotation_status"] + [r["sample"] for r in samples])
        writer.writerows(rows)
    bindings = {str(SAMPLES.resolve()): sha(SAMPLES), str(Path(__file__).resolve()): sha(__file__)}
    for name in ("manifest.json", "unique_introns.saf", "feature_gene_mapping.tsv", "gene_table.tsv"):
        bindings[str(annotation / name)] = sha(annotation / name)
    for item in inputs:
        bindings[item["path"]] = sha(item["path"])
    manifest = {"schema": "bothe-count-matrix-contract-v1", "evidence": "audit_mock",
                "status": "prepared_partial" if len(available) < 12 else "audit_complete",
                "readyRNA": False, "model_eligible": False, "real_executor_qualified": False,
                "annotation": str(annotation), "inputs": inputs, "bindings": bindings,
                "samples": samples, "available_samples": available,
                "missing_samples": [r["sample"] for r in samples if r["sample"] not in available],
                "counts_sha256": sha(output / "counts.tsv"),
                "missing_semantics": "empty: missing sample or unmeasurable gene; zero: observed measurable gene"}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return manifest


def load_matrix(directory, *, allow_audit_mock=False):
    directory = Path(directory)
    manifest = _json(directory / "manifest.json")
    if manifest.get("evidence") != "audit_mock" or not allow_audit_mock:
        raise ValueError("real count qualification unavailable; audit_mock requires explicit opt-in")
    if manifest.get("readyRNA") is not False or manifest.get("model_eligible") is not False or manifest.get("real_executor_qualified") is not False:
        raise ValueError("audit fixture cannot confer eligibility")
    samples, expected, available = _matrix(manifest["annotation"], manifest["inputs"])
    required = {str(SAMPLES.resolve()), str(Path(__file__).resolve())}
    required.update(str(Path(manifest["annotation"]) / n) for n in
                    ("manifest.json", "unique_introns.saf", "feature_gene_mapping.tsv", "gene_table.tsv"))
    required.update(i["path"] for i in manifest["inputs"])
    if set(manifest["bindings"]) != required or any(sha(p) != h for p, h in manifest["bindings"].items()):
        raise ValueError("matrix source/code binding changed")
    with (directory / "counts.tsv").open(encoding="utf-8", newline="") as stream:
        actual = list(csv.reader(stream, delimiter="\t"))
    missing = [r["sample"] for r in samples if r["sample"] not in available]
    if (actual != [["ncbi_gene_id", "annotation_status"] + [r["sample"] for r in samples]] + expected
            or sha(directory / "counts.tsv") != manifest["counts_sha256"]
            or manifest["samples"] != samples or manifest["available_samples"] != available
            or manifest["missing_samples"] != missing
            or manifest["status"] != ("prepared_partial" if len(available) < 12 else "audit_complete")
            or manifest["schema"] != "bothe-count-matrix-contract-v1"):
        raise ValueError("matrix cache/provenance mismatch")
    return manifest, tuple(tuple(row) for row in expected)
