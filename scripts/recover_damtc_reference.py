"""Recover GSE29698 and source-reported Annexin condition references.

This is an external reference, not paired RNA/flow replicates or a training set.
The source's solvent discrepancy remains unresolved and blocks qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.recover_public_expression import read_soft, table_block, save_native, digest


def recover(soft, paper):
    text, records = read_soft(soft)
    root = ET.parse(paper).getroot()
    if not any(e.get("pub-id-type") == "doi" and e.text == "10.1038/cddis.2012.141"
               for e in root.findall(".//article-id")):
        raise ValueError("Wrong DAMTC source article")
    body = " ".join("".join(root.find("body").itertext()).split())
    if "160 microgram/ml for 24 h" not in text or "10 mg/ml in DMSO" not in body:
        raise ValueError("Source treatment/stock protocol changed; re-review")
    if "DMSO-treated cells (0.1%)" not in body:
        raise ValueError("Source vehicle protocol changed; re-review")
    figure = root.find(".//fig[@id='fig8']/caption")
    if figure is None or "four independent experiments" not in "".join(figure.itertext()):
        raise ValueError("Endpoint experiment count not verified")
    # Extract reported numbers from the specific drug-only/combinations paragraph.
    paragraphs = [" ".join("".join(p.itertext()).split()) for p in root.findall(".//body//p")]
    paragraphs = [p for p in paragraphs if "32.37% apoptotic cells" in p and "57.1% apoptotic cells" in p]
    if len(paragraphs) != 1:
        raise ValueError("Cannot identify unique Fig8a numerical paragraph")
    paragraph = paragraphs[0]
    patterns = [
        (80, 0, r"DAMTC \(80 μg/ml\) alone resulted in ([\d.]+)%"),
        (160, 0, r"DAMTC \(160 μg/ml\) alone resulted in ([\d.]+)%"),
        (0, 10, r"etoposide \(10 μM\) alone resulted in ([\d.]+)%"),
        (80, 10, r"DAMTC \(80 μg/ml\) in combination with etoposide resulted in ([\d.]+)%"),
        (160, 10, r"DAMTC \(160 μg/ml\) in combination with etoposide resulted in ([\d.]+)%"),
    ]
    endpoints = []
    for dose, etoposide, pattern in patterns:
        match = re.search(pattern, paragraph)
        if not match:
            raise ValueError("Endpoint sentence changed; do not guess numbers")
        value = float(match.group(1))
        if not 0 <= value <= 100:
            raise ValueError("Invalid percentage")
        endpoints.append(dict(cell_line="A549", DAMTC_ug_ml=dose, etoposide_uM=etoposide,
                              exposure_h=24, value=value, unit="Annexin_positive_percent",
                              statistic="source_reported_condition_summary", source_figure="8a",
                              independent_experiments_reported=4, raw_replicates_available=False,
                              paired_to_RNA_replicate=False, combination_order="simultaneous" if etoposide and dose else "single_agent",
                              data_role="unassigned_external_reference"))
    expression, rows = None, []
    for gsm, meta, block in records:
        title = meta["title"][0]
        match = re.fullmatch(r"A549-(control|treated with DAMTC) rep([123])", title)
        if not match or meta["platform_id"] != ["GPL6884"]:
            raise ValueError("Unverified sample identity/platform")
        if meta["data_processing"] != ["Average Normalization using Illumina Beadstudio"]:
            raise ValueError("Expression scale changed")
        table = table_block(block, "sample")
        if not table.ID_REF.is_unique:
            raise ValueError("Duplicate source probes")
        if expression is None:
            expression = table[["ID_REF"]].copy()
        if table.ID_REF.tolist() != expression.ID_REF.tolist():
            raise ValueError("Probe order mismatch")
        expression[gsm] = pd.to_numeric(table.VALUE, errors="raise")
        rows.append(dict(geo_accession=gsm, sample_title=title, cell_line="A549",
                         DAMTC_ug_ml=0 if match[1] == "control" else 160, exposure_h=24,
                         replicate_label=match[2], source_array_location=meta["description"][0],
                         observed_initial_state=False, paired_to_function_replicate=False,
                         data_role="unassigned_external_reference"))
    samples = pd.DataFrame(rows)
    if len(samples) != 6 or samples.groupby("DAMTC_ug_ml").size().to_dict() != {0: 3, 160: 3}:
        raise ValueError("Expected three biological RNA replicates per source condition")
    return expression, samples, pd.DataFrame(endpoints)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--soft", type=Path, required=True)
    ap.add_argument("--paper-xml", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new directory")
    frame, samples, endpoints = recover(args.soft, args.paper_xml)
    args.output.mkdir(parents=True)
    report = save_native(args.output, frame, samples, "ID_REF", "Illumina_Beadstudio_average_normalized_VALUE")
    endpoints.to_csv(args.output / "endpoint_references.csv", index=False)
    links = dict(RNA_GSMs=samples.loc[samples.DAMTC_ug_ml.eq(160), "geo_accession"].tolist(),
                 RNA_control_GSMs=samples.loc[samples.DAMTC_ug_ml.eq(0), "geo_accession"].tolist(),
                 endpoint_figure="8a", DAMTC_ug_ml=160, etoposide_uM=0, exposure_h=24,
                 link_level="same_study_nominal_condition_not_replicate_pair",
                 endpoint_value=float(endpoints.loc[endpoints.DAMTC_ug_ml.eq(160) & endpoints.etoposide_uM.eq(0), "value"].iloc[0]),
                 endpoint_unit="Annexin_positive_percent",
                 vehicle_control_endpoint=None, absolute_rate_calibration_qualified=False,
                 protocol_qualified=False,
                 unresolved="10 mg/ml DMSO stock implies 1.6% solvent at 160 ug/ml by direct dilution; paper reports 0.1% vehicle. Actual preparation is not established.")
    (args.output / "condition_link.json").write_text(json.dumps(links, indent=2))
    report.update(endpoint_conditions=len(endpoints), matched_treated_RNA_conditions=1,
                  full_research_feasibility_proven=False, training_performed=False, model_input_ready=False)
    manifest = dict(validation=report, sources=[dict(path=str(p.resolve()), sha256=digest(p)) for p in (args.soft, args.paper_xml)],
                    script_sha256=digest(__file__), outputs={p.name: digest(p) for p in args.output.iterdir()})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
