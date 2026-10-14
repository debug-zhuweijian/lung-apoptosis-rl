"""Add primary-read printed flow values without overwriting the r3 snapshot."""

import argparse
from decimal import Decimal
from pathlib import Path

from build_endpoint_alignment_v1 import EndpointBuilder, ROOT, write_alignment
from lung.data.endpoint_alignment import load_alignment

BASE = ROOT / "08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1"
REVIEW = BASE / "source_reviews/chembl_round2/raw"


def quadrant_total(printed):
    """Retain printed precision; accept only sums consistent with its rounding."""
    values = [Decimal(x) for x in printed]
    if len(values) != 4 or any(not x.is_finite() or x < 0 or x > 100 for x in values):
        raise ValueError("invalid printed quadrant percentages")
    tolerance = sum(Decimal(5).scaleb(x.as_tuple().exponent - 1) for x in values)
    if abs(sum(values) - 100) > tolerance:
        raise ValueError("quadrant sum exceeds printed rounding uncertainty")
    # Order: double negative, dye-only, Annexin-only, double positive.
    return float((values[2] + values[3]) / 100)


STUDIES = [
    {"source": "Huang2019_6k", "doi": "10.1039/c8md00620b", "compound": "6k",
     "hours": 24, "figure": "PMC6484948_fig6.jpg", "text": "primary_text_6k.md",
     "gate": "Fig6 Annexin V-FITC horizontal, PI vertical; Q3 plus Q2; pregate not specified",
     "methods": "sec46 Apoptosis analysis; DMEM+10%FBS, BD kit, CellQuest; floating-cell pooling not specified",
     "chemical_limit": "primary_C39H53FN4O5_disagrees_with_ChEMBL_C39H55FN4O4",
     "hold": "", "rows": [
         (0, ["96.9", "1.56", "1.57", "0.010"]),
         (5, ["87.8", "0.397", "9.25", "2.55"]),
         (10, ["70.5", "0.116", "26.5", "2.92"]),
         (20, ["59.4", "0.190", "36.0", "4.38"])]},
    {"source": "Li2012_AS_E", "doi": "10.1016/j.bmc.2012.09.056", "compound": "1a_naphthol_AS_E",
     "hours": 48, "figure": "PMC3587690_fig3.jpg", "text": "PMC3587690.xml",
     "gate": "Fig3 Annexin V-FITC horizontal, PI vertical; lower-right plus upper-right; pregate not specified",
     "methods": "secS31 Flow cytometry; BD Annexin V-FITC Kit I, FACSCalibur; Fig3 Weasel v2.5",
     "chemical_limit": "source_name_to_full_structure_review_pending",
     "hold": "", "rows": [
         (0, ["91.6", "1.5", "3.9", "2.98"]),
         (5, ["89.1", "1.36", "6.5", "3.06"]),
         (7.5, ["85.6", "1.42", "10.1", "2.9"]),
         (10, ["75.9", "0.58", "19.4", "4.2"])]},
    {"source": "C9MD00494G_8h", "doi": "10.1039/c9md00494g", "compound": "8h",
     "hours": 48, "figure": "PMC7489057_fig4.jpg", "text": "primary_text_8h.md",
     "gate": "Fig4A A549 top row gated P1; Annexin V-FITC horizontal, 7-AAD vertical; right two quadrants",
     "methods": "Fig4A 48h; Fig4B 24h is Western only; detailed supplement assay method not accessed",
     "chemical_limit": "source_numbered_compound_to_structure_review_pending",
     "hold": "supplement_assay_method_not_verified", "rows": [
         (0, ["93.10", "0.70", "3.26", "2.94"]),
         (.3, ["79.43", "8.73", "4.92", "6.92"]),
         (1, ["78.34", "4.23", "8.03", "9.40"]),
         (3, ["0.71", "1.65", "3.91", "93.73"])]},
]


def extend(out, source):
    previous = load_alignment(source, ROOT)
    builder = EndpointBuilder()
    builder.conditions = {r["condition_id"]: r for r in previous["conditions"].to_dict("records")}
    builder.interventions = previous["interventions"].to_dict("records")
    builder.observations = previous["observations"].to_dict("records")
    builder.protocols = previous["protocols"].copy()
    builder.sources = {r["file"]: r for r in previous["manifest"]["source_files"]}
    builder.source(source / "manifest.json")
    for study in STUDIES:
        source_id = study["source"]
        protocol = source_id + ":Annexin"
        if protocol in builder.protocols:
            raise ValueError("round2 source already present; do not duplicate it")
        builder.source(REVIEW / study["text"])
        builder.protocols[protocol] = {"source": "https://doi.org/" + study["doi"],
            "endpoint": "Annexin_positive_fraction", "denominator": "published flow analyzed population; not seeded cell count",
            "sampling_gate": study["gate"], "method_evidence": study["methods"],
            "replicate_type": "one_reported_panel; raw independent replicates unavailable",
            "numeric_basis": "printed original figure quadrant percentages, personally read by main reviewer",
            "structure_status": study["chemical_limit"], "method_complete_for_intake": not bool(study["hold"])}
        for dose, quadrants in study["rows"]:
            condition = builder.condition(source_id, protocol, f"Annexin_{study['hours']}h", "A549",
                "A549_parental_unmeasured_reference", [(study["compound"], dose)] if dose else [],
                treatment_hours=study["hours"], chemical_review_note=study["chemical_limit"])
            builder.observation(condition, "Annexin_positive_fraction", quadrant_total(quadrants),
                "published_panel", REVIEW / study["figure"],
                f"A549:{study['compound'] if dose else 'control'}:{dose:g}uM:{study['hours']}h:right_two_quadrants",
                exclusion=study["hold"], replicate_kind="published_value_not_raw_replicate",
                printed_quadrants_percent=";".join(quadrants), primary_methods_source=study["text"])
    write_alignment(builder, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "curated-20261010-r3")
    parser.add_argument("--output", type=Path, default=BASE / "curated-20261010-r4")
    args = parser.parse_args()
    extend(args.output if args.output.is_absolute() else ROOT / args.output,
           args.source if args.source.is_absolute() else ROOT / args.source)
