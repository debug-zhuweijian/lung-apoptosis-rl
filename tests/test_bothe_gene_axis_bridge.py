import pandas as pd
import pytest
from lung.data.bothe_gene_axis_bridge import build_tables
from lung.data.bothe_gene_axis_bridge import read_gene_info
import gzip
import json
from pathlib import Path
from lung.data.bothe_gene_axis_bridge import load_bridge, REFERENCE
from lung.data import bothe_gene_axis_bridge as bridge
from lung.data import bothe_count_production as production
from types import SimpleNamespace


@pytest.fixture
def projection_fixture(monkeypatch):
    directory = REFERENCE / "gene-axis-bridge-v1"
    if not directory.exists():
        pytest.skip("Registered metadata fixture not available")
    ledger = pd.read_csv(directory / "source_status_ledger.tsv", sep="\t", dtype="string")
    ledger["common_gene_index"] = ledger.common_gene_index.astype("Int64")
    ledger["id_bridge_mask"] = ledger.id_bridge_mask.eq("True")
    axis = pd.read_csv(directory / "joint_axis.tsv", sep="\t", dtype="string")
    axis["common_gene_index"] = axis.common_gene_index.astype(int)
    axis["id_bridge_mask"] = axis.id_bridge_mask.eq("True")
    counts = SimpleNamespace(version="isolated_fixture", gene_ids=tuple(ledger.ncbi_gene_id),
        annotation_status=tuple(ledger.annotation_status), sample_ids=bridge.SAMPLE_IDS,
        observed_samples=(bridge.SAMPLE_IDS[0],), counts_complete=False,
        domain="bulk_total_RNA_unique_intron_fragment_counts", model_eligible=False, function_eligible=False)
    counts.values = tuple((0,) + (None,) * 11 if status == "retained" else (None,) * 12
                          for status in counts.annotation_status)
    monkeypatch.setattr(bridge, "load_bridge", lambda directory: (ledger, axis))
    monkeypatch.setattr(production, "load_production", lambda version: counts)
    return counts, ledger, axis


def test_projection_partial_zero_large_integer_and_unmapped(projection_fixture):
    counts, ledger, axis = projection_fixture
    gene = ledger.loc[ledger.id_bridge_mask].iloc[0]
    index = counts.gene_ids.index(gene.ncbi_gene_id)
    rows = list(counts.values)
    rows[index] = (2**63 - 1,) + (None,) * 11
    counts.values = tuple(rows)
    projected = bridge.project_production_counts("isolated_fixture")
    assert projected.values[int(gene.common_gene_index)][0] == 2**63 - 1
    assert type(projected.values[int(gene.common_gene_index)][0]) is int
    assert projected.observed_mask[int(gene.common_gene_index)] == (True,) + (False,) * 11
    zero = next(i for i, row in enumerate(projected.values) if row[0] == 0)
    assert projected.observed_mask[zero][0] is True
    absent = axis.index[~axis.id_bridge_mask][0]
    assert projected.values[absent] == (None,) * 12
    assert projected.observed_mask[absent] == (False,) * 12
    assert projected.model_eligible is projected.function_eligible is False
    assert not projected.counts_complete


def test_projection_full_all_samples(projection_fixture):
    counts, _, _ = projection_fixture
    counts.observed_samples = bridge.SAMPLE_IDS
    counts.counts_complete = True
    counts.values = tuple((0,) * 12 if status == "retained" else (None,) * 12 for status in counts.annotation_status)
    projected = bridge.project_production_counts("isolated_fixture")
    assert projected.counts_complete is True
    assert sum(sum(row) for row in projected.observed_mask) == 17107 * 12


@pytest.mark.parametrize("mutation", ["gene_order", "sample_order", "status", "float", "bool", "negative", "overflow", "missing", "unavailable", "completeness", "no_feature", "shape"])
def test_projection_rejects_invalid_source(projection_fixture, mutation):
    counts, _, _ = projection_fixture
    if mutation == "gene_order":
        counts.gene_ids = counts.gene_ids[::-1]
    elif mutation == "sample_order":
        counts.sample_ids = counts.sample_ids[::-1]
    elif mutation == "status":
        counts.annotation_status = ("wrong",) + counts.annotation_status[1:]
    elif mutation == "completeness":
        counts.counts_complete = True
    elif mutation == "shape":
        counts.values = counts.values[:-1]
    else:
        i = next(i for i, status in enumerate(counts.annotation_status) if status != "retained") if mutation == "no_feature" else counts.annotation_status.index("retained")
        values = list(counts.values)
        row = list(values[i])
        bad = {"float": 1.0, "bool": True, "negative": -1, "overflow": 2**63, "missing": None}
        if mutation == "no_feature":
            row[0] = 0
        elif mutation == "unavailable":
            row[1] = 0
        else:
            row[0] = bad[mutation]
        values[i] = tuple(row)
        counts.values = tuple(values)
    with pytest.raises(ValueError):
        bridge.project_production_counts("isolated_fixture")


def test_projection_pending_propagates_without_placeholders(monkeypatch):
    def pending(version):
        raise FileNotFoundError("registered production pending")
    monkeypatch.setattr(production, "load_production", pending)
    monkeypatch.setattr(bridge, "load_bridge", lambda _: pytest.fail("pending should not load bridge"))
    with pytest.raises(FileNotFoundError):
        bridge.project_production_counts("pending_fixture")


def test_projection_rejects_duplicate_mapping(projection_fixture):
    _, ledger, _ = projection_fixture
    indices = ledger.index[ledger.id_bridge_mask][:2]
    ledger.loc[indices[1], "ensembl_id"] = ledger.loc[indices[0], "ensembl_id"]
    with pytest.raises(ValueError, match="one-to-one"):
        bridge.project_production_counts("isolated_fixture")


def test_projection_public_api_has_no_caller_counts_or_runner():
    with pytest.raises(TypeError):
        bridge.project_production_counts("isolated_fixture", counts=())
    with pytest.raises(TypeError):
        bridge.project_production_counts("isolated_fixture", runner=lambda: None)


def fixtures():
    source = pd.DataFrame({"ncbi_gene_id": ["1", "2", "3", "4", "5", "6", "7"],
                           "status": ["retained"] * 6 + ["no_features"]})
    ids = [f"ENSG{i:011d}" for i in range(1, 6)]
    axis = pd.DataFrame({"common_gene_index": range(5), "ensembl_id": ids})
    official = pd.DataFrame({"GeneID": ["1", "2", "3", "4", "6", "7", "99"],
                             "dbXrefs": [f"Ensembl:{ids[0]}", f"Ensembl:{ids[1]}|Ensembl:{ids[2]}",
                                         f"Ensembl:{ids[3]}", "-", "Ensembl:ENSG99999999999",
                                         f"Ensembl:{ids[4]}", f"Ensembl:{ids[3]}"]})
    return source, axis, official


def test_exact_ids_ambiguity_and_missing_are_masked():
    ledger, axis = build_tables(*fixtures())
    assert ledger.mapping_status.tolist() == ["mapped", "ambiguous_one_to_many", "ambiguous_many_to_one",
        "no_ensembl_dbxref", "absent_current_ncbi_deprecation_unconfirmed", "not_in_joint_axis", "annotation_unmeasurable"]
    assert ledger.loc[~ledger.id_bridge_mask, "common_gene_index"].isna().all()
    assert axis.id_bridge_mask.tolist() == [True, False, False, False, False]
    assert axis.loc[~axis.id_bridge_mask, "ncbi_gene_id"].isna().all()


def test_duplicate_source_and_shuffled_axis_rejected():
    source, axis, official = fixtures()
    with pytest.raises(ValueError, match="Duplicate"):
        build_tables(pd.concat([source, source.iloc[:1]]), axis, official)
    with pytest.raises(ValueError, match="order"):
        build_tables(source, axis.iloc[::-1], official)


def test_versioned_official_id_not_silently_rewritten():
    source, axis, official = fixtures()
    official.loc[0, "dbXrefs"] += ".1"
    with pytest.raises(ValueError, match="Unexpected"):
        build_tables(source, axis, official)


def test_human_file_other_taxa_are_explicitly_filtered(tmp_path):
    path = tmp_path / "gene_info.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("#tax_id\tGeneID\tdbXrefs\n9606\t1\tEnsembl:ENSG00000000001\n63221\t2\t-\n")
    assert read_gene_info(path).GeneID.tolist() == ["1"]


def test_duplicate_official_id_rejected():
    source, axis, official = fixtures()
    with pytest.raises(ValueError, match="Duplicate official"):
        build_tables(source, axis, pd.concat([official, official.iloc[:1]]))


@pytest.mark.parametrize("field", ["GeneID", "dbXrefs"])
def test_missing_official_identity_rejected(tmp_path, field):
    path = tmp_path / "gene_info.gz"
    values = {"GeneID": "1", "dbXrefs": "-"}
    values[field] = ""
    with gzip.open(path, "wt") as handle:
        handle.write(f"#tax_id\tGeneID\tdbXrefs\n9606\t{values['GeneID']}\t{values['dbXrefs']}\n")
    with pytest.raises(ValueError, match="GeneID"):
        read_gene_info(path)


@pytest.mark.parametrize("mutation", ["fake_gene_table", "fake_official", "missing_ledger", "model", "status", "counts"])
def test_loader_rejects_caller_certified_manifest(tmp_path, mutation):
    original = REFERENCE / "gene-axis-bridge-v1/manifest.json"
    if not original.exists():
        pytest.skip("Registered real metadata snapshot not available")
    manifest = json.loads(original.read_text(encoding="utf-8"))
    if mutation.startswith("fake_"):
        key = "gene_table" if mutation == "fake_gene_table" else "official"
        fake = tmp_path / "fake_source"
        fake.write_text("caller certified tiny source", encoding="utf-8")
        from lung.data.bothe_gene_axis_bridge import sha256
        manifest["inputs"][key] = {"path": str(fake), "sha256": sha256(fake), "bytes": fake.stat().st_size}
    elif mutation == "missing_ledger":
        manifest["outputs"].pop("scope_status_ledger.tsv")
    elif mutation == "model":
        manifest["qualifications"]["model"] = True
    elif mutation == "status":
        manifest["status"] = "scientific_pass"
    else:
        manifest["counts_recovered"] = 12
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_bridge(tmp_path)
