"""Fixed interval cases: source validity, locus quarantine and serialized axes."""
import gzip
import csv
import pytest
from src.lung.data.intron_annotation import (
    Transcript, build, parse_genepred, read_sources, select_transcripts,
    unique_segments, verify_readback,
)


def row(name="NM_A.1", chrom="chr1", strand="+", exons=((0, 10), (30, 40)), symbol="A"):
    return "\t".join(map(str, [0, name, chrom, strand, exons[0][0], exons[-1][1],
                               exons[0][0], exons[-1][1], len(exons),
                               ",".join(str(x[0]) for x in exons) + ",",
                               ",".join(str(x[1]) for x in exons) + ",",
                               0, symbol, "cmpl", "cmpl", ",".join("0" for _ in exons) + ","]))


def tx(**kwargs):
    return parse_genepred(row(**kwargs))


def test_halfopen_and_spliced_length():
    value = tx()
    assert value.exons == ((0, 10), (30, 40))
    assert value.introns == ((10, 30),)
    assert value.length == 20


@pytest.mark.parametrize("position,value", [
    (3, "."), (4, "-1"), (5, "39"), (6, "41"), (8, "3"),
    (9, "0,8,"), (10, "10,30,"), (13, "invalid"), (15, "0,3,"),
])
def test_invalid_genepred(position, value):
    fields = row().split("\t")
    fields[position] = value
    with pytest.raises(ValueError):
        parse_genepred("\t".join(fields))


def test_reject_wrong_schema():
    with pytest.raises(ValueError, match="16 fields"):
        parse_genepred(row() + "\textra")


def test_adjacent_exons_and_single_exon_have_no_empty_introns():
    assert tx(exons=((0, 10), (10, 20))).introns == ()
    assert tx(exons=((0, 10),)).introns == ()


def test_longest_is_spliced_not_genomic_and_tie_is_deterministic():
    a = tx(name="NM_A.2", exons=((0, 5), (100, 105)))
    b = tx(name="NM_B.1", exons=((0, 20), (30, 50)))
    c = tx(name="NM_A.1", exons=((0, 20), (30, 50)))
    links = {x.accession: ("12", "same") for x in (a, b, c)}
    for values in ((a, b, c), (c, b, a)):
        chosen, _ = select_transcripts(values, links)
        assert chosen["12"] == c


@pytest.mark.parametrize("other", [
    tx(name="NM_B", chrom="chr2"), tx(name="NM_B", strand="-"),
    tx(name="NM_B", exons=((100, 110), (130, 140))),
])
def test_primary_conflicts_quarantined(other):
    a = tx()
    selected, ledger = select_transcripts([a, other], {a.accession: ("1", "A"), other.accession: ("1", "A")})
    assert selected == {}
    assert all("conflict" in x[-1] or "loci" in x[-1] for x in ledger)


def test_alt_locus_is_out_of_scope_not_longest():
    a = tx()
    alt = tx(name="NM_ALT", chrom="chr1_alt", exons=((0, 100), (200, 300)))
    chosen, ledger = select_transcripts([a, alt], {a.accession: ("1", "A"), alt.accession: ("1", "A")})
    assert chosen["1"] == a
    assert ledger[0][-1] == "outside_primary_scope"


def test_all_transcript_exons_removed_even_opposite_strand_or_missing_id():
    a = tx(exons=((0, 10), (50, 60)))
    blocker = tx(name="unmapped", strand="-", exons=((20, 30),))
    selected, _ = select_transcripts([a, blocker], {a.accession: ("1", "A")})
    intervals, stats = unique_segments([a, blocker], selected)
    assert [(x[1], x[2]) for x in intervals] == [(10, 20), (30, 50)]
    assert stats["excluded_exonic_bp"] == 10


def test_multiple_genes_overlap_both_strands_and_boundaries():
    a = tx(exons=((0, 10), (50, 60)))
    b = tx(name="B", strand="-", exons=((0, 20), (40, 60)))
    selected, _ = select_transcripts([a, b], {a.accession: ("1", "A"), b.accession: ("2", "B")})
    intervals, stats = unique_segments([a, b], selected)
    assert intervals == []
    assert stats == {"candidate_union_bp": 40, "excluded_exonic_bp": 20, "excluded_multiple_gene_bp": 20}


def sources(path):
    path.mkdir()
    with gzip.open(path / "ncbiRefSeq.txt.gz", "wt", encoding="utf-8") as handle:
        handle.write(row() + "\n" + row(name="single", chrom="chr2", exons=((0, 20),)) + "\n")
    with gzip.open(path / "ncbiRefSeqLink.txt.gz", "wt", encoding="utf-8") as handle:
        for accession, gene in (("NM_A.1", "123"), ("single", "456")):
            values = [""] * 19
            values[0], values[2], values[6] = accession, "SHARED_SYMBOL", gene
            handle.write("\t".join(values) + "\n")


def test_build_readback_saf_conversion_stable_gene_ids_and_empty_genes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    sources(source)
    manifest = build(source, output, tmp_path / "receipt")
    assert manifest["validation"]["features_read"] == 1
    assert (output / "unique_introns.saf").read_text() == "GeneID\tChr\tStart\tEnd\tStrand\nUIV1_00000001\tchr1\t11\t30\t+\n"
    with (output / "gene_table.tsv").open() as handle:
        genes = list(csv.DictReader(handle, delimiter="\t"))
    assert [x["ncbi_gene_id"] for x in genes] == ["123", "456"]
    assert genes[1]["status"] == "no_source_intron"
    assert all(x["symbol"] == "SHARED_SYMBOL" for x in genes)
    with pytest.raises(ValueError, match="nonempty"):
        build(source, output, tmp_path / "receipt")
    transcripts, links = read_sources(source)
    chosen, _ = select_transcripts(transcripts, links)
    saf = output / "unique_introns.saf"
    saf.write_text(saf.read_text().replace("\t11\t30", "\t10\t30"))
    with pytest.raises(ValueError, match="coordinates mismatch"):
        verify_readback(output, transcripts, chosen)


def test_link_schema_and_geneid_validation(tmp_path):
    source = tmp_path / "source"
    sources(source)
    with gzip.open(source / "ncbiRefSeqLink.txt.gz", "wt") as handle:
        values = [""] * 19
        values[0], values[6] = "NM_A.1", "not-a-gene"
        handle.write("\t".join(values) + "\n")
    with pytest.raises(ValueError, match="GeneID"):
        read_sources(source)
