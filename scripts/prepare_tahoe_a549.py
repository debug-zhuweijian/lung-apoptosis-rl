"""Audit observed A549 coverage from pinned official Tahoe metadata.

Reads selected columns from all remote obs row groups, not a sample/cell-line
Cartesian product. Full expression data are deliberately not downloaded.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

REPO = "tahoebio/Tahoe-100M"
REVISION = "2dc57900b7981cfcf5e211527169a0b006546a95"
ROOT = Path(__file__).resolve().parents[1] / "08-datasets" / "Tahoe-100M"
BASE = ROOT / REVISION
A549 = "CVCL_0023"
META_NAMES = ["gene_metadata.parquet", "cell_line_metadata.parquet",
              "sample_metadata.parquet", "drug_metadata.parquet"]
OBS_COLUMNS = ["plate", "sample", "cell_line", "pass_filter",
               "BARCODE_SUB_LIB_ID"]


def url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_small_metadata() -> None:
    metadata_url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/metadata?limit=100"
    response = requests.get(metadata_url, timeout=90)
    response.raise_for_status()
    source_entries = {row["path"]: row for row in response.json()}
    dump_json(BASE / "metadata_source_manifest.json", response.json())
    for name in META_NAMES:
        path = BASE / "metadata" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            data = requests.get(url(f"metadata/{name}"), timeout=120)
            data.raise_for_status()
            path.write_bytes(data.content)
        entry = source_entries[f"metadata/{name}"]
        assert path.stat().st_size == entry["size"], name
        assert sha256(path) == entry["lfs"]["oid"], name
    readme = BASE / "README.md"
    if not readme.exists():
        response = requests.get(url("README.md"), timeout=90)
        response.raise_for_status()
        readme.write_bytes(response.content)


def scan_observations(workers: int) -> pd.DataFrame:
    source = url("metadata/obs_metadata.parquet")
    with fsspec.open(source, "rb", block_size=65536).open() as handle:
        metadata = pq.ParquetFile(handle).metadata
    dump_json(BASE / "obs_remote_schema.json", {
        "url": source, "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "columns": metadata.schema.names,
        "selected_columns": OBS_COLUMNS,
        "source_full_file_sha256": "e51b24f3666f5f433c59ff01dd6fa33934627091455e89d929f2c1ddc87401fe",
        "hash_note": "Official LFS digest; full source file was not downloaded or locally hashed.",
    })
    directory = BASE / "derived" / "a549_obs_row_groups"
    directory.mkdir(parents=True, exist_ok=True)

    def read_group(index: int) -> Path:
        destination = directory / f"a549_obs_{index:03d}.parquet"
        if destination.exists():
            return destination
        with fsspec.open(source, "rb", block_size=65536).open() as handle:
            table = pq.ParquetFile(handle, metadata=metadata).read_row_group(
                index, columns=OBS_COLUMNS, use_threads=False)
        subset = table.filter(pc.equal(table["cell_line"], A549))
        subset = subset.append_column("source_row_group", pa.array([index] * len(subset)))
        partial = destination.with_suffix(".partial")
        pq.write_table(subset, partial, compression="zstd")
        partial.replace(destination)
        return destination

    files = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_group, index): index
                   for index in range(metadata.num_row_groups)}
        for index, future in enumerate(as_completed(futures), start=1):
            files.append(future.result())
            if index % 8 == 0 or index == metadata.num_row_groups:
                print(f"Scanned obs row groups {index}/{metadata.num_row_groups}", flush=True)
    table = pa.concat_tables([pq.ParquetFile(path).read() for path in sorted(files)])
    result = table.to_pandas()
    assert result["BARCODE_SUB_LIB_ID"].is_unique
    assert result["cell_line"].eq(A549).all()
    result.to_parquet(BASE / "derived" / "a549_observed_cells.parquet", index=False)
    return result


def prepare_conditions(observations: pd.DataFrame) -> dict:
    samples = pd.read_parquet(BASE / "metadata/sample_metadata.parquet")
    assert samples["sample"].is_unique
    counts = observations.groupby(["sample", "plate", "pass_filter"], observed=True).size().unstack(fill_value=0)
    counts = counts.rename(columns={name: f"n_cells_{name}" for name in counts.columns})
    counts["n_cells_obs_total"] = counts.sum(axis=1)
    counts = counts.reset_index()
    assert set(counts["sample"]).issubset(set(samples["sample"]))
    observed = counts.merge(samples, on=["sample", "plate"], validate="one_to_one")
    assert len(observed) == len(counts)
    parsed = observed["drugname_drugconc"].map(ast.literal_eval)
    observed["n_compounds_recorded"] = parsed.map(len)
    observed["dose_value"] = parsed.map(lambda x: x[0][1] if len(x) == 1 else None)
    observed["dose_unit"] = parsed.map(lambda x: x[0][2] if len(x) == 1 else "multiple")
    observed["cell_line_id"] = A549
    observed["cell_name"] = "A549"
    observed["is_vehicle_control"] = observed["drug"].eq("DMSO_TF")
    # Source sample labels have trailing spaces for Erdafitinib and Selinexor.
    # Retain raw labels; strip only whitespace for this metadata join.
    drug_meta = pd.read_parquet(BASE / "metadata/drug_metadata.parquet")
    drug_meta = drug_meta.assign(drug_key=drug_meta["drug"].str.strip())
    assert drug_meta["drug_key"].is_unique
    observed["drug_key"] = observed["drug"].str.strip()
    observed = observed.merge(drug_meta[["drug_key", "canonical_smiles", "pubchem_cid"]],
                              on="drug_key", how="left", validate="many_to_one", indicator=True)
    observed["drug_metadata_matched"] = observed["_merge"].eq("both")
    assert observed.loc[~observed["drug_metadata_matched"], "is_vehicle_control"].all()
    observed = observed.drop(columns="_merge")
    observed["structure_status"] = "official_structure_available"
    observed.loc[observed["canonical_smiles"].isna(), "structure_status"] = "missing_in_official_metadata"
    observed.loc[observed["is_vehicle_control"], "structure_status"] = "vehicle_not_compound"
    # Time is a study-level protocol assertion, not a measured per-cell field.
    observed["treatment_hours"] = 24
    observed["time_source"] = "Tahoe preprint v1 Methods: Compound sourcing and treatment; study-level protocol"
    observed["biological_replicate_plate"] = observed["plate"].map({"plate6": "plate14", "plate14": "plate6"}).fillna("")
    observed["replicate_status"] = observed["plate"].map({"plate6": "paired_plate6_14", "plate14": "paired_plate6_14"}).fillna("not_specified_as_biological_replicate")
    controls = observed[observed["is_vehicle_control"]].groupby("plate", observed=True)["sample"].apply(lambda x: ";".join(sorted(x)))
    observed["matched_control_samples"] = observed["plate"].map(controls).fillna("")
    observed["expression_full_downloaded"] = False
    observed["expression_source"] = url("data/train-*.parquet")
    observed["expression_location_note"] = "Repository shard pattern, not per-cell shard mapping; full expression extraction pending"
    # No filtering of minimal cells: downstream decisions must be explicit.
    observed.to_csv(BASE / "derived/a549_observed_sample_conditions.csv", index=False, encoding="utf-8-sig")
    missing = samples[~samples["sample"].isin(observed["sample"])].copy()
    missing["a549_status"] = "no_observed_A549_row_in_complete_obs_scan"
    missing.to_csv(BASE / "derived/samples_without_observed_a549.csv", index=False, encoding="utf-8-sig")
    links = [(row["sample"], control, row["plate"], A549)
             for _, row in observed.iterrows()
             for control in row["matched_control_samples"].split(";") if control]
    pd.DataFrame(links, columns=["sample", "control_sample", "plate", "cell_line_id"]).to_csv(
        BASE / "derived/a549_plate_control_links.csv", index=False, encoding="utf-8-sig")
    summary = {
        "revision": REVISION, "source_obs_all_rows": 100648790,
        "observed_a549_cells": len(observations),
        "filter_counts": observations["pass_filter"].value_counts().to_dict(),
        "observed_a549_samples": len(observed), "metadata_samples": len(samples),
        "samples_without_observed_a549": len(missing),
        "observed_a549_plates": observed["plate"].nunique(),
        "observed_a549_drug_names_including_vehicle": observed["drug"].nunique(),
        "observed_a549_drug_dose_labels_including_vehicle": observed["drugname_drugconc"].nunique(),
        "vehicle_samples": int(observed["is_vehicle_control"].sum()),
        "nonvehicle_samples_missing_smiles": int((~observed["is_vehicle_control"] & observed["canonical_smiles"].isna()).sum()),
        "samples_without_plate_matched_control": int(observed["matched_control_samples"].eq("").sum()),
        "full_expression_extraction": "not_run",
        "time_field": "24 hours assigned from study protocol; absent in sample/obs metadata",
    }
    dump_json(BASE / "derived/a549_coverage_summary.json", summary)
    return summary


def write_manifest() -> None:
    rows = []
    for path in sorted(BASE.rglob("*")):
        if path.is_file() and path.name != "local_file_manifest.json":
            relative = path.relative_to(BASE).as_posix()
            rows.append({"path": relative, "bytes": path.stat().st_size,
                         "sha256": sha256(path),
                         "source": url(relative) if relative.startswith("data/") or relative in ["README.md"] + [f"metadata/{name}" for name in META_NAMES] else "derived_or_audit_record"})
    dump_json(BASE / "local_file_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPO, "revision": REVISION, "files": rows})


def expression_smoke() -> dict:
    """Read two original shards to verify raw-count parsing, not train a model."""
    response = requests.get(
        f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data?limit=2", timeout=90)
    response.raise_for_status()
    entries = response.json()
    dump_json(BASE / "expression_smoke_source_manifest.json", entries)
    subsets = []
    for entry in entries:
        path = BASE / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with requests.get(url(entry["path"]), timeout=120, stream=True) as source:
                source.raise_for_status()
                partial = path.with_suffix(".partial")
                with partial.open("wb") as handle:
                    for chunk in source.iter_content(1024 * 1024):
                        handle.write(chunk)
                partial.replace(path)
        assert path.stat().st_size == entry["size"]
        assert sha256(path) == entry["lfs"]["oid"]
        table = pq.read_table(path)
        subset = table.filter(pc.equal(table["cell_line_id"], A549))
        subsets.append(subset.append_column("source_shard", pa.array([entry["path"]] * len(subset))))
    result = pa.concat_tables(subsets)
    pq.write_table(result, BASE / "derived/a549_expression_smoke.parquet", compression="zstd")
    data = result.to_pandas()
    valid_tokens = set(pd.read_parquet(BASE / "metadata/gene_metadata.parquet")["token_id"])
    lengths_match = all(len(g) == len(x) for g, x in zip(data.genes, data.expressions))
    invalid_after_cls = set(int(t) for genes in data.genes for t in genes[1:]) - valid_tokens
    assert lengths_match and not invalid_after_cls
    assert data["BARCODE_SUB_LIB_ID"].is_unique
    obs = pd.read_parquet(BASE / "derived/a549_observed_cells.parquet")
    joined = data.merge(obs[["BARCODE_SUB_LIB_ID", "sample", "plate", "pass_filter"]],
                        on="BARCODE_SUB_LIB_ID", validate="one_to_one", suffixes=("_expr", "_obs"))
    assert len(joined) == len(data)
    assert joined["sample_expr"].eq(joined["sample_obs"]).all()
    assert joined["plate_expr"].eq(joined["plate_obs"]).all()
    raw_counts_valid = all(all(v >= 0 and float(v).is_integer() for v in row[1:])
                           for row in data.expressions)
    assert raw_counts_valid
    info = {"a549_cells": len(data), "samples": data["sample"].nunique(),
            "plates": sorted(data["plate"].unique()),
            "vehicle_cells": int(data.drug.eq("DMSO_TF").sum()),
            "first_gene_tokens": sorted({int(x[0]) for x in data.genes}),
            "first_expression_values": sorted({float(x[0]) for x in data.expressions}),
            "gene_expression_lengths_match": lengths_match,
            "unknown_gene_tokens_after_cls": len(invalid_after_cls),
            "joined_to_obs": len(joined),
            "all_raw_counts_nonnegative_integer": raw_counts_valid,
            "official_filter_counts": {str(k): int(v) for k, v in joined["pass_filter"].value_counts().items()},
            "scope": "two first repository shards only; not random or complete A549 subset"}
    dump_json(BASE / "derived/expression_smoke_summary.json", info)
    return info


def bridge_condition_observations() -> None:
    """Count actual A549/H1792 cells at exact Trametinib 0.5 uM and controls."""
    samples = pd.read_parquet(BASE / "metadata/sample_metadata.parquet")
    target_samples = samples[samples["sample"].isin(["smp_1625", "smp_2523"])]
    target_plates = set(target_samples["plate"])
    keep = samples[samples["sample"].isin(target_samples["sample"]) |
                   (samples["plate"].isin(target_plates) & samples["drug"].eq("DMSO_TF"))]
    source = url("metadata/obs_metadata.parquet")
    groups = []
    with fsspec.open(source, "rb", block_size=65536).open() as handle:
        parquet = pq.ParquetFile(handle)
        for index in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(index).column(0).statistics
            if stats and not any(stats.min <= plate <= stats.max for plate in target_plates):
                continue
            table = parquet.read_row_group(index, columns=["sample", "cell_line", "pass_filter"])
            table = table.filter(pc.and_(pc.is_in(table["sample"], value_set=pa.array(keep["sample"])),
                                        pc.is_in(table["cell_line"], value_set=pa.array([A549, "CVCL_1495"]))))
            groups.append(table.to_pandas().groupby(["sample", "cell_line", "pass_filter"], observed=True).size())
    counts = pd.concat(groups).groupby(level=[0, 1, 2], observed=True).sum().rename("n_observed_cells").reset_index()
    counts = counts.merge(keep[["sample", "plate", "drug", "drugname_drugconc"]], on="sample", validate="many_to_one")
    counts["time_hours_protocol"] = 24
    counts.to_csv(BASE / "derived/trametinib_bridge_observed_counts.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expression-smoke", action="store_true")
    parser.add_argument("--bridge-observations", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    download_small_metadata()
    observations = scan_observations(args.workers)
    summary = prepare_conditions(observations)
    if args.expression_smoke:
        summary["expression_smoke"] = expression_smoke()
    if args.bridge_observations:
        bridge_condition_observations()
    write_manifest()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
