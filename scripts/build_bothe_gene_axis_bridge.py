"""Build the metadata bridge without accessing expression, FASTQ or checkpoints."""
import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from lung.data.bothe_gene_axis_bridge import (build_tables, load_bridge, read_gene_info, sha256,
                                             FROZEN_HASHES, QUALIFICATIONS, summarize, supplemental_tables)
from lung.data.bothe_gene_axis_bridge import project_production_counts, REFERENCE


def main():
    base = ROOT / "08-datasets/Bothe2021/E-MTAB-9923/reference"
    output = base / "gene-axis-bridge-v1"
    annotation = base / "unique-introns-v1"
    target = ROOT / "08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1"
    inputs = {"official": output / "official/Homo_sapiens.gene_info.gz",
              "official_readme": output / "official/README.txt",
              "official_head": output / "official/head.json",
              "gene_table": annotation / "gene_table.tsv",
              "scope_ledger": annotation / "transcript_scope_ledger.tsv",
              "annotation_manifest": annotation / "manifest.json",
              "axis": target / "genes.parquet", "axis_manifest": target / "manifest.json"}
    frozen = FROZEN_HASHES
    entries = {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
               for name, path in inputs.items()}
    for name, expected in frozen.items():
        if entries[name]["sha256"] != expected:
            raise ValueError(f"Frozen {name} hash mismatch")
    if (output / "manifest.json").exists():
        raise FileExistsError("Published bridge already exists")
    source = pd.read_csv(inputs["gene_table"], sep="\t", dtype="string")
    axis = pd.read_parquet(inputs["axis"])
    official = read_gene_info(inputs["official"])
    raw_official = pd.read_csv(inputs["official"], sep="\t", dtype="string")
    ledger, full = build_tables(source, axis, official)
    scope = pd.read_csv(inputs["scope_ledger"], sep="\t", dtype="string")
    scope, tax_excluded = supplemental_tables(scope, raw_official)
    outputs = {"source_status_ledger.tsv": ledger, "joint_axis.tsv": full,
               "scope_status_ledger.tsv": scope, "official_tax_excluded.tsv": tax_excluded}
    for name, frame in outputs.items():
        frame.to_csv(output / name, sep="\t", index=False)
    manifest = {"schema": "bothe-gene-axis-bridge-v1", "validation_revision": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "ready_id_bridge_only", "counts_recovered": 0,
                "source_domain": "Bothe bulk unique intronic RNA counts; distinct from joint single-cell expression",
                "tax_id": 9606, "official_url": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
                "official_description_url": "https://ftp.ncbi.nlm.nih.gov/gene/README",
                "official_release_last_modified": json.loads(inputs["official_head"].read_text())["Last-Modified"],
                "relation": "GeneID exact → gene_info dbXrefs Ensembl exact; globally bijective only; no symbols or coordinates",
                "deprecation_policy": "Absent current GeneID is unresolved; gene_history not read, no replacement inferred",
                "inputs": entries, "outputs": {name: sha256(output / name) for name in outputs},
                "summary": summarize(source, full, ledger, official, scope, tax_excluded),
                "qualifications": QUALIFICATIONS}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rebuilt = load_bridge(output)
    pd.testing.assert_frame_equal(ledger, rebuilt[0])
    pd.testing.assert_frame_equal(full, rebuilt[1])
    validation = {"metadata_readback": "passed", "rebuild_from_official_ids": "passed",
                  "axis_order_equal": full.ensembl_id.tolist() == axis.ensembl_id.tolist(),
                  "mapped_source_unique": not ledger.loc[ledger.id_bridge_mask, "ncbi_gene_id"].duplicated().any(),
                  "mapped_target_unique": not ledger.loc[ledger.id_bridge_mask, "ensembl_id"].duplicated().any(),
                  "unmapped_target_ids_are_na": bool(full.loc[~full.id_bridge_mask, "ncbi_gene_id"].isna().all()),
                  "unmapped_source_target_indices_are_na": bool(ledger.loc[~ledger.id_bridge_mask, "common_gene_index"].isna().all()),
                  "counts_recovered": 0, "manifest_sha256": sha256(output / "manifest.json")}
    report = ROOT / "experiments/20261011-data-alignment-round20/bothe_gene_axis_bridge"
    report.mkdir(parents=True, exist_ok=True)
    (report / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "readback", "project"), default="build")
    parser.add_argument("--version", help="Registered source-bound production version; project mode only")
    args = parser.parse_args()
    if args.mode == "project":
        if not args.version:
            parser.error("project requires --version")
        projection = project_production_counts(args.version)
        print(json.dumps({"version": projection.version, "domain": projection.domain,
                          "genes": len(projection.ensembl_ids), "samples": len(projection.sample_ids),
                          "observed_samples": projection.observed_samples, "counts_complete": projection.counts_complete,
                          "model_eligible": False, "function_eligible": False}, indent=2))
    elif args.version:
        parser.error("--version is valid only for project mode")
    elif args.mode == "readback":
        ledger, axis = load_bridge(REFERENCE / "gene-axis-bridge-v1")
        print(json.dumps({"source_genes": len(ledger), "joint_genes": len(axis), "mapped": int(axis.id_bridge_mask.sum()),
                          "counts_recovered": 0, "model_eligible": False, "function_eligible": False}, indent=2))
    else:
        main()
