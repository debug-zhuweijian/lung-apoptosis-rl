"""Re-read primary source values into explicit condition-level endpoint tables.

No expression rows are synthesized, no original source or frozen bundle is changed,
and no outcome is marked as a same-cell expression/endpoint pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.data.endpoint_alignment import (aggregate_units, file_hash, stable_id,
                                          validate_alignment)

DATA = ROOT / "08-datasets"
DEFAULT_OUT = DATA / "LungCancer_A549_Integration/endpoint-alignment-v1/curated"


def workbook(path):
    # The local runtime already provides openpyxl; no new project dependency.
    try:
        import openpyxl
    except ImportError:
        bundled = Path("C:/Users/Windows11/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages")
        if not bundled.exists():
            raise
        sys.path.append(str(bundled))
        import openpyxl
    return openpyxl.load_workbook(path, read_only=True, data_only=True)


class EndpointBuilder:
    def __init__(self):
        self.conditions = {}
        self.interventions = []
        self.observations = []
        self.sources = {}
        self.protocols = {}

    def source(self, path):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in self.sources:
            self.sources[relative] = {"file": relative, "sha256": file_hash(path)}
        return self.sources[relative]

    def condition(self, source, protocol, block, line, state, components, **extra):
        key = stable_id("condition", source, protocol, block, line, state, components)
        if key not in self.conditions:
            self.conditions[key] = {
                "condition_id": key, "source_id": source, "protocol_id": protocol,
                "experiment_block": block, "cell_line": line, "initial_state_id": state,
                "initial_state_kind": "reference", "initial_expression_sample_id": None,
                "treatment_hours": 24.0,
                "exposure_design": "simultaneous_combination" if len(components) > 1 else "single_exposure",
                "is_control": not components, "split_group": source,
                "feature_status": "pending_generation", "same_sample_expression_endpoint": False,
                **extra,
            }
            for name, dose in components:
                self.interventions.append({
                    "condition_id": key, "compound_id": source + ":" + name,
                    "source_compound_name": name, "dose_value": float(dose), "dose_unit": "uM",
                    "identity_status": "nominal_identity_structure_review_pending",
                    "dose_basis": "as_reported_in_source_protocol",
                })
        return key

    def observation(self, condition_id, endpoint, value, replicate, source, locator,
                    exclusion="", measurement_status="observed", **extra):
        record = self.source(source)
        self.observations.append({
            "observation_id": stable_id("observation", record["sha256"], locator, endpoint),
            "condition_id": condition_id, "endpoint": endpoint, "value": value,
            "unit": "fraction", "label_kind": "measured", "replicate_id": str(replicate),
            "source_file": record["file"], "source_sha256": record["sha256"],
            "source_locator": locator, "measurement_status": measurement_status,
            "exclusion_reason": exclusion, "numeric_source_verified": True,
            "replicate_kind": "independent_experiment", **extra,
        })


def add_macaya(builder):
    directory = DATA / "GSE161218/GSE161218"
    source = directory / "sources/41467_2023_41828_MOESM7_ESM.xlsx"
    book = workbook(source)
    for suffix, filename, second in [
        ("S2C", "annexin_v_fig_s2c_long.csv", "Lestaurtinib"),
        ("S3DE", "annexin_v_fig_s3de_midostaurin_long.csv", "Midostaurin"),
    ]:
        protocol = "Macaya2023:" + suffix
        builder.protocols[protocol] = {
            "source": "https://doi.org/10.1038/s41467-023-41828-z",
            "endpoint": "Annexin_positive_fraction",
            "denominator": "events retained by published flow gate; not initial culture cell count",
            "sampling_gate": "FSC/SSC alive and singlet gating; Annexin V/7AAD; supplement p5",
            "control_definition": "untreated; do not assume identical to RNA DMSO control",
            "replicate_type": "independent_experiments_as_reported",
            "expression_status": "A549 RNA not supplied by this study",
        }
        frame = pd.read_csv(directory / filename)
        builder.source(directory / filename)
        for row in frame[frame.cell_line == "A549"].to_dict("records"):
            cell = book[row["source_sheet"]][row["source_cell"]].value
            reported = row["annexin_v_positive_percent"]
            if cell is None:
                assert pd.isna(reported)
            else:
                assert abs(float(cell) - reported) < 1e-10
            components = []
            if row["trametinib_uM"] > 0:
                components.append(("Trametinib", row["trametinib_uM"]))
            dose = row[second.lower() + "_uM"]
            if dose > 0:
                components.append((second, dose))
            condition = builder.condition("Macaya2023", protocol, suffix, "A549",
                                          "A549_parental_unmeasured_reference", components)
            builder.observation(condition, "Annexin_positive_fraction",
                                None if cell is None else float(cell) / 100,
                                row["experiment_number_within_endpoint"], source,
                                row["source_sheet"] + "!" + row["source_cell"],
                                "source_cell_missing" if cell is None else "",
                                "missing" if cell is None else "observed")
    book.close()


def add_mphahlele(builder):
    directory = DATA / "Mphahlele2023_Annexin/s00044-023-03136-5"
    source = directory / "44_2023_3136_MOESM1_ESM.docx"
    builder.source(directory / "TableS3_original_image26.png")
    builder.source(directory / "table2_verified.html")
    frame = pd.read_csv(directory / "TableS3_event_transcription_review.csv")
    builder.source(directory / "TableS3_event_transcription_review.csv")
    builder.protocols["Mphahlele2023:S3"] = {
        "source": "https://doi.org/10.1007/s00044-023-03136-5",
        "endpoint": "Annexin_positive_fraction",
        "denominator": "10000 acquired events in each reported panel",
        "sampling_gate": "Table S3 Q2 plus Q4 Annexin-positive quadrants",
        "dose_definition": "each compound's Table2 A549 MTT IC50; condition-specific dose",
        "replicate_type": "one_reported_panel; biological replicate series unavailable",
        "scope": "doses exceed current Tahoe range; source chemical series, not broad drug panel",
    }
    for row in frame.to_dict("records"):
        name = row["source_compound_id"]
        value = (row["Q2_events"] + row["Q4_events"]) / row["all_events"]
        assert abs(value * 100 - row["annexin_positive_percent_from_events"]) < 1e-10
        components = [] if name == "untreated" else [(name, row["dose_um"])]
        condition = builder.condition("Mphahlele2023", "Mphahlele2023:S3", "S3", "A549",
                                      "A549_parental_unmeasured_reference", components,
                                      response_domain="dose_extrapolation")
        builder.observation(condition, "Annexin_positive_fraction", value,
                            "one_reported_panel", source, "word/media/image26.png:TableS3:" + name,
                            replicate_kind="reported_panel_not_raw_replicate",
                            total_events=int(row["all_events"]),
                            positive_events=int(row["Q2_events"] + row["Q4_events"]))


def add_inde(builder):
    directory = DATA / "Inde2020_FractionalKilling/9bccjb75wp-v1"
    source = directory / "261 library + DMSO.xlsx"
    book = workbook(source)
    values = {sheet.title: list(sheet.values) for sheet in book
              if sheet.title.startswith("Lethal Fraction rep")}
    frame = pd.read_csv(directory / "single_drug_24h_review.csv")
    builder.source(directory / "single_drug_24h_review.csv")
    builder.source(directory / "PMC7409774.xml")
    builder.protocols["Inde2020:SYTOX24h"] = {
        "source": "https://doi.org/10.1016/j.celrep.2020.107845",
        "endpoint": "SYTOX_lethal_fraction",
        "denominator": "SYTOX/(mKate2+SYTOX-overlap), author lethal fraction definition",
        "sampling_gate": "image-based live/dead counting; not flow Annexin",
        "replicate_type": "three independent biological experiments as reported",
        "scope": "A549 nuclear-mKate2 derivative, single-drug+DMSO arm, 24h",
    }
    for row in frame.to_dict("records"):
        r, c = row["source_row_1based"], row["source_column_1based"]
        cells = values[row["sheet"]]
        value = cells[r - 1][c - 1]
        assert abs(value - row["lethal_fraction"]) < 1e-12
        assert cells[r - 1][0] == 24
        assert cells[0][c - 1] == row["source_compound_name"]
        condition = builder.condition("Inde2020", "Inde2020:SYTOX24h", "library_DMSO",
                                      "A549_nuclear_mKate2", "A549_N_reference",
                                      [(row["source_compound_name"], 5.0)],
                                      response_domain="cell_derivative_transfer")
        builder.observation(condition, "SYTOX_lethal_fraction", float(value),
                            row["replicate"], source, f"{row['sheet']}!R{r}C{c}",
                            "author_autofluorescence_exclusion" if row["author_autofluorescence_exclusion"] else "")
    book.close()


def add_reviewed_literature(builder):
    """Exact published numbers, not ChEMBL-only values or digitized bar heights."""
    directory = DATA / "LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chembl/raw"
    studies = [
        ("MCT1_2023", "PMC9841531.xml", "10.1021/acs.jmedchem.2c01612", "24",
         [(5.0, 24.0, 6.68), (0.0, 24.0, 0.51)],
         "Results and Figure9; PI-only excluded, Annexin-only plus double-positive",
         "reported_panel; raw independent replicate values unavailable"),
        ("NaturalProduct2018", "PMC6448791.xml", "10.1021/acs.jnatprod.7b00578", "14",
         [(2.0, 24.0, 14.8 + 3.80), (4.0, 24.0, 26.5 + 4.40),
          (8.0, 24.0, 30.3 + 4.87), (0.0, 24.0, 1.43 + 2.97)],
         "Results and Figure2; Annexin+/PI- plus Annexin+/PI+",
         "reported_panel; caption reports three experiments but their raw values are unavailable"),
        ("MCT2014", "PMC4111373.xml", "10.1021/jm5003588", "9a",
         [(0.1, 24.0, 3.7), (0.0, 24.0, 1.1), (0.1, 48.0, 34.1), (0.0, 48.0, 2.0)],
         "Results and Figure3B; total Annexin V-positive population",
         "published_summary; independent replicate series unavailable"),
        ("Mostafa2018", "Mostafa2018_Figure6_publisher.jpg", "10.1016/j.ejmech.2018.07.004", "19",
         [(2.0, 24.0, 3.27 + 3.31)],
         "Results2.2.5 and Figure6B left panel; Annexin-only plus double-positive",
         "reported_panel; raw replicates and complete denominator unavailable"),
    ]
    for source_id, filename, doi, compound, values, gate, repeat_kind in studies:
        source = directory / filename
        if not source.exists():
            raise FileNotFoundError(f"Primary article was not obtained: {source}")
        protocol = source_id + ":Annexin"
        builder.protocols[protocol] = {
            "source": "https://doi.org/" + doi, "endpoint": "Annexin_positive_fraction",
            "denominator": "flow analyzed population as reported, not initial cell count",
            "sampling_gate": gate, "replicate_type": repeat_kind,
            "numeric_basis": "exact numbers in primary article body; no plot digitization",
            "structure_status": "source compound ID; registry separately reviewed",
        }
        for dose, time, percent in values:
            condition = builder.condition(source_id, protocol, f"Annexin_{time:g}h", "A549",
                                          "A549_parental_unmeasured_reference",
                                          [(compound, dose)] if dose else [], treatment_hours=time)
            builder.observation(condition, "Annexin_positive_fraction", percent / 100,
                                "reported_value", source,
                                f"{gate.split(';')[0]}:{compound if dose else 'control'}:{dose:g}uM:{time:g}h",
                                exclusion="published_quadrant_sum_not_100_denominator_review" if source_id == "Mostafa2018" else "",
                                replicate_kind="published_value_not_raw_replicate")


def write_alignment(builder, out):
    out = out if out.is_absolute() else ROOT / out
    tables = {"conditions": pd.DataFrame(builder.conditions.values()),
              "interventions": pd.DataFrame(builder.interventions),
              "observations": pd.DataFrame(builder.observations)}
    checks = validate_alignment(**tables, protocols=builder.protocols)
    tables["condition_units"] = aggregate_units(tables["conditions"], tables["observations"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Completed alignment is immutable; use a new --output directory")
    records = []
    for name, frame in tables.items():
        path = out / (name + ".parquet")
        frame.to_parquet(path, index=False)
        records.append({"file": path.name, "sha256": file_hash(path), "rows": len(frame)})
        # Human-readable parallel exports; verified table is parquet.
        frame.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
    path = out / "protocols.json"
    path.write_text(json.dumps(builder.protocols, ensure_ascii=False, indent=2), encoding="utf-8")
    records.append({"file": path.name, "sha256": file_hash(path)})
    manifest = {
        "schema_version": 1, "version": out.name,
        "supervision_unit": "measured_experimental_condition_not_individual_generated_cells",
        "numeric_alignment_verified": True,
        "chemical_alignment_status": "pending_explicit_structure_registry_join",
        "feature_status": "pending_generation_from_frozen_models",
        "formal_endpoint_training_ready": False,
        "full_research_data_feasibility_verified": False,
        "known_development_sources": sorted(tables["conditions"].source_id.unique()),
        "test_partition_status": "not_declared_blind; sources inspected during design",
        "checks": checks, "files": records, "source_files": list(builder.sources.values()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), **checks}, ensure_ascii=False))


def build(out):
    builder = EndpointBuilder()
    add_macaya(builder)
    add_mphahlele(builder)
    add_inde(builder)
    add_reviewed_literature(builder)
    write_alignment(builder, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    build(parser.parse_args().output)
