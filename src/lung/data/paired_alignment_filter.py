"""Disk-backed, whole-template SAM filtering. No model or raw-data qualification."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3


POLICY = {
    "version": "bothe-paired-sam-v1",
    "author_known": "SAMtools removes records MAPQ<10; primary/bothEnds/noChimera counting",
    "project_additions": "actual unique primary R1/R2; both MAPQ>=10; reciprocal coordinates/strand flags; same chromosome; reject malformed templates",
    "secondary": "exclude records", "supplementary": "exclude records (project choice)",
    "proper_pair": "not required", "duplicate": "retain and report",
    "NH": "reject present NH !=1; absent allowed and reported unknown",
    "strandness": "unknown; downstream unstranded (-s 0)",
    "fragment": "one retained template = two records; no length cutoff; reciprocal TLEN required",
    "MAPQ255": "reject unavailable quality (project choice)",
    "output_order": "original input order; header bytes preserved",
    "spec": "https://samtools.github.io/hts-specs/SAMv1.pdf (b5341fb, 2025-08-12), sections 1.3-1.5",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_record(line, references):
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 11 or not re.fullmatch(r"[!-?A-~]{1,254}", fields[0]) or fields[0] == "*":
        raise ValueError("invalid fields/QNAME")
    flag, pos, quality, matepos, tlen = (int(fields[i]) for i in (1, 3, 4, 7, 8))
    if not 0 <= flag <= 65535 or not 0 <= quality <= 255 or not 0 <= pos <= 2147483647 or not 0 <= matepos <= 2147483647 or not -2147483648 <= tlen <= 2147483647:
        raise ValueError("numeric range")
    cigar = fields[5]
    if cigar != "*":
        if not re.fullmatch(r"(?:[1-9][0-9]*[MIDNSHP=X])+", cigar):
            raise ValueError("invalid CIGAR")
        ops = [(int(n), op) for n, op in re.findall(r"([0-9]+)([MIDNSHP=X])", cigar)]
        core = [op for _, op in ops]
        if not re.fullmatch(r"H?S?[MIDNP=X]+S?H?", ''.join(core)):
            raise ValueError("invalid clipping order")
        length = sum(n for n, op in ops if op in "MIS=X")
        if fields[9] != "*" and len(fields[9]) != length:
            raise ValueError("CIGAR/SEQ length")
        refspan = sum(n for n, op in ops if op in "MDN=X")
        if not flag & 4 and (refspan == 0 or pos < 1 or fields[2] not in references or pos + refspan - 1 > references[fields[2]]):
            raise ValueError("reference bounds")
    elif not flag & 4:
        raise ValueError("mapped missing CIGAR")
    if fields[10] != "*" and (fields[9] == "*" or len(fields[10]) != len(fields[9])):
        raise ValueError("SEQ/QUAL length")
    tags = {}
    for tag in fields[11:]:
        pieces = tag.split(":", 2)
        if len(pieces) != 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]", pieces[0]) or pieces[0] in tags:
            raise ValueError("invalid/duplicate tag")
        tags[pieces[0]] = pieces[1:]
    nh = None
    if "NH" in tags:
        kind, value = tags["NH"]
        if kind != "i" or not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError("invalid NH")
        nh = int(value)
    return fields, flag, pos, quality, matepos, tlen, nh


def pair_reasons(records):
    reasons = set()
    if len(records) != 2:
        return {"orphan" if len(records) < 2 else "duplicate_primary"}
    if sorted(r[1] & 192 for r in records) != [64, 128]:
        reasons.add("invalid_R1_R2")
    for a, b in (records, records[::-1]):
        f, flag, pos, quality, matepos, tlen, nh = a
        if not flag & 1 or flag & 12:
            reasons.add("not_paired_both_mapped")
        if quality < 10 or quality == 255:
            reasons.add("MAPQ")
        if nh is not None and nh != 1:
            reasons.add("NH_multimap")
        if f[2] != b[0][2]:
            reasons.add("cross_chromosome")
        if (f[2] if f[6] == "=" else f[6]) != b[0][2] or matepos != b[2]:
            reasons.add("nonreciprocal_coordinates")
        if bool(flag & 32) != bool(b[1] & 16):
            reasons.add("nonreciprocal_strand_flags")
        if tlen != -b[5]:
            reasons.add("nonreciprocal_TLEN")
    return reasons


def filter_sam(source, output_dir):
    """Create a new output directory. Structural errors fail closed, with receipt."""
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    counters = Counter()
    receipt = {"status": "failed", "source": str(source), "policy": POLICY,
               "scope": "SAM filtering only; no raw/count qualification"}
    db = sqlite3.connect(output_dir / "templates.sqlite")
    try:
        receipt["source_sha256"] = sha256(source)
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, q TEXT, raw BLOB, primary_record INTEGER, malformed INTEGER)")
        db.execute("CREATE INDEX qnames ON records(q)")
        db.execute("CREATE TABLE accepted(q TEXT PRIMARY KEY)")
        references, headers = {}, []
        body = False
        with source.open("rb") as stream:
            for index, raw in enumerate(iter(lambda: stream.readline(1024 * 1024 + 1), b""), 1):
                if len(raw) > 1024 * 1024:
                    raise ValueError(f"line {index}: exceeds 1MiB processing bound")
                if raw.startswith(b"@"):
                    if body:
                        raise ValueError(f"line {index}: header after alignment (concatenated SAM rejected)")
                    header_tags = {}
                    if not raw.startswith(b"@CO\t"):
                        for field in raw.rstrip(b"\r\n").split(b"\t")[1:]:
                            tag, value = field.split(b":", 1)
                            if tag in header_tags:
                                raise ValueError("duplicate header tag")
                            header_tags[tag] = value
                    if raw.startswith(b"@SQ\t"):
                        name, length = header_tags[b"SN"].decode("utf-8"), int(header_tags[b"LN"])
                        if name in references or length <= 0:
                            raise ValueError("invalid/duplicate SQ")
                        references[name] = length
                    headers.append(raw)
                    counters["header_lines"] += 1
                    counters["header_bytes"] += len(raw)
                    if counters["header_bytes"] > 16 * 1024 * 1024:
                        raise ValueError("header exceeds 16MiB processing bound")
                    continue
                text = raw.decode("utf-8")
                body = True
                counters["input_records"] += 1
                q = text.split("\t", 1)[0]
                if not q or q == text.rstrip("\r\n"):
                    raise ValueError(f"line {index}: no recoverable QNAME")
                try:
                    record = parse_record(text, references)
                    primary = not bool(record[1] & (256 | 2048))
                    malformed = 0
                    if not primary:
                        counters["excluded_nonprimary_records"] += 1
                except (ValueError, KeyError):
                    primary, malformed = 0, 1
                    counters["malformed_records"] += 1
                db.execute("INSERT INTO records VALUES(?,?,?,?,?)", (index, q, raw, int(primary), malformed))
                if index % 10000 == 0:
                    db.commit()
        db.commit()
        for q, primary_count, malformed in db.execute("SELECT q,SUM(primary_record),MAX(malformed) FROM records GROUP BY q"):
            counters["templates_seen"] += 1
            if malformed:
                reasons = {"malformed"}
            elif primary_count != 2:
                reasons = {"orphan" if primary_count < 2 else "duplicate_primary"}
            else:
                records = [parse_record(raw.decode("utf-8"), references) for (raw,) in db.execute("SELECT raw FROM records WHERE q=? AND primary_record=1", (q,))]
                reasons = pair_reasons(records)
            if reasons:
                counters["templates_rejected"] += 1
                for reason in reasons:
                    counters["rejected_" + reason] += 1
            else:
                db.execute("INSERT INTO accepted VALUES(?)", (q,))
                counters["templates_retained"] += 1
                counters["retained_NH_unknown_templates"] += int(any(r[6] is None for r in records))
                counters["retained_duplicate_flag_templates"] += int(any(r[1] & 1024 for r in records))
                counters["retained_not_proper_templates"] += int(any(not r[1] & 2 for r in records))
        db.commit()
        output = output_dir / "filtered.sam"
        pending = output_dir / "unqualified_filtered.sam.part"
        with pending.open("xb") as stream:
            for header in headers:
                stream.write(header)
            for (raw,) in db.execute("SELECT r.raw FROM records r JOIN accepted a ON r.q=a.q WHERE r.primary_record=1 ORDER BY r.id"):
                stream.write(raw)
                counters["output_records"] += 1
        assert counters["output_records"] == 2 * counters["templates_retained"]
        if sha256(source) != receipt["source_sha256"]:
            raise ValueError("source changed during filtering")
        output_hash = sha256(pending)
        pending.rename(output)
        receipt.update(status="complete", output=str(output), output_sha256=output_hash)
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        db.close()
        receipt["counters"] = dict(sorted(counters.items()))
        receipt["record_accounting"] = {"input": counters["input_records"], "output": counters["output_records"], "dropped": counters["input_records"] - counters["output_records"]}
        receipt_pending = output_dir / "unqualified_receipt.json.part"
        try:
            receipt_pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            receipt_pending.rename(output_dir / "receipt.json")
        except Exception:
            published = output_dir / "filtered.sam"
            if published.exists():
                published.rename(output_dir / "unqualified_filtered.sam.part")
            raise
    return receipt
