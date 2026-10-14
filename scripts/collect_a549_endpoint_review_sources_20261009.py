"""Preserve small public source files used in the 2026-10-09 feasibility review."""

import hashlib
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

sys.path.append("C:/Users/Windows11/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages")
import openpyxl


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "08-datasets"
SOURCES = [
    ("Inde2020_FractionalKilling/9bccjb75wp-v1/261 library + DMSO.xlsx",
     "https://data.mendeley.com/public-files/datasets/9bccjb75wp/files/eeff3433-7308-46da-bb77-2e093cdeba85/file_downloaded",
     "2f51796806586e48501753b6aba6d152a42a49e78a0028d3c9b0ed1575a36c71"),
    ("Inde2020_FractionalKilling/9bccjb75wp-v1/PMC7409774.xml",
     "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7409774/fullTextXML", None),
    ("TIDYI/E-MTAB-16351/E-MTAB-16351.study.json",
     "https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-16351", None),
    ("TIDYI/E-MTAB-16351/Chromium10x_1001_A549_metadata.csv",
     "https://www.ebi.ac.uk/biostudies/files/E-MTAB-16351/Chromium10x_1001_A549_metadata.csv",
     "0bfd2c9aa83ea3d6c2f133d4ec7f2882e7082171f4a15a687f98d4d155d43a62"),
    ("TIDYI/E-MTAB-16351/Chromium10x_1002_A549_metadata.csv",
     "https://www.ebi.ac.uk/biostudies/files/E-MTAB-16351/Chromium10x_1002_A549_metadata.csv",
     "349cd7c3f456c31f23c5bb28a5580b6dbbe480249c333353abca670e890530c9"),
    ("Mphahlele2023_Annexin/s00044-023-03136-5/44_2023_3136_MOESM1_ESM.docx",
     "https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs00044-023-03136-5/MediaObjects/44_2023_3136_MOESM1_ESM.docx",
     "6ac58a4129072e6a7956438a9bb8c2d091a85c76472c74bbc2d0db20c0a7e7fb"),
    ("Mphahlele2023_Annexin/s00044-023-03136-5/table2_verified.html",
     "https://link.springer.com/article/10.1007/s00044-023-03136-5/tables/2", None),
]


def main():
    manifests = {}
    for relative, url, expected in SOURCES:
        path = DATA / relative
        if path.exists():
            content = path.read_bytes()
        else:
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.mendeley.com/datasets/9bccjb75wp/1"} if "data.mendeley.com" in url else {}
            response = requests.get(url, headers=headers, timeout=40)
            response.raise_for_status()
            content = response.content
            if path.suffix == ".html" and b"<table" not in content:
                raise ValueError(f"No source table in HTTP response: {relative}")
            if expected and hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Source changed: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
        digest = hashlib.sha256(content).hexdigest()
        if expected and digest != expected:
            raise ValueError(f"Existing source hash mismatch: {relative}")
        manifests.setdefault(path.parent, []).append({"file": path.name, "url": url, "bytes": len(content), "sha256": digest})
        print(path.name, len(content), digest, flush=True)
        if path.suffix == ".docx":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                image = archive.read("word/media/image26.png")
            image_path = path.parent / "TableS3_original_image26.png"
            if not image_path.exists():
                with image_path.open("xb") as handle:
                    handle.write(image)
    for directory, entries in manifests.items():
        manifest = directory / "source_verification_20261009.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(), "status": "source_files_only_no_training_integration", "files": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_inde()
    transcribe_mphahlele()


def audit_inde():
    directory = DATA / "Inde2020_FractionalKilling/9bccjb75wp-v1"
    book = openpyxl.load_workbook(directory / "261 library + DMSO.xlsx", read_only=True, data_only=True)
    excluded = {"Sunitinib Malate", "Mitoxantrone HCl", "PHA-665752"}
    observations = []
    names = None
    for replicate in (1, 2, 3):
        sheet = book[f"Lethal Fraction rep {replicate}"]
        values = list(sheet.values)
        header = list(values[0][1:])
        if names is not None:
            assert header == names
        names = header
        matches = [(index, row) for index, row in enumerate(values, 1) if row[0] == 24]
        assert len(matches) == 1
        index, row = matches[0]
        for column, name in enumerate(header, 2):
            if not name or name == "Empty":
                continue
            value = row[column - 1]
            assert isinstance(value, (int, float)) and 0 <= value <= 1
            observations.append({"source_compound_name": name, "dose_um": 5, "time_hours": 24, "replicate": replicate, "lethal_fraction": value, "endpoint": "SYTOX_lethal_fraction_not_Annexin", "sheet": sheet.title, "source_row_1based": index, "source_column_1based": column, "author_autofluorescence_exclusion": name in excluded, "use_for_training": False})
    frame = pd.DataFrame(observations)
    assert len(frame) == 783 and frame.source_compound_name.nunique() == 261
    frame.to_csv(directory / "single_drug_24h_review.csv", index=False, encoding="utf-8-sig")
    conditions = pd.read_parquet(DATA / "LungCancer_A549_Integration/prepared/v1/tahoe/conditions.parquet")
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "", str(value).lower())
    index = {normalize(name): name for name in conditions.drug_name.dropna().unique()}
    links = []
    for name in frame.source_compound_name.unique():
        aliases = [name.split("(")[0].strip()] + re.findall(r"\(([^)]+)\)", name)
        candidate_names = {index[normalize(part)] for alias in aliases for part in alias.split(",") if normalize(part) in index}
        for matched in candidate_names:
            subset = conditions[(conditions.drug_name == matched) & (conditions.dose_value == 5) & (conditions.treatment_hours == 24)]
            if not subset.empty:
                links.append({"source_compound_name": name, "tahoe_drug_name": matched, "dose_um": 5, "time_hours": 24, "tahoe_condition_ids": ";".join(subset.condition_id), "identity_status": "name_candidate_only_structure_review_pending", "use_as_paired_training_target": False})
    pd.DataFrame(links).to_csv(directory / "tahoe_name_dose_time_candidates.csv", index=False, encoding="utf-8-sig")
    print("Inde 24h audit:", len(frame), "observations;", len(links), "name candidates", flush=True)


def transcribe_mphahlele():
    directory = DATA / "Mphahlele2023_Annexin/s00044-023-03136-5"
    # Exact integer events read from Table S3 image26; doses from Table 2 A549 column.
    source = [("3a",16.94,4323,3264),("3b",14.33,3262,3293),("3c",7.75,6015,2443),
              ("3d",13.35,4237,3393),("3e",26.16,2002,3445),("3f",16.10,2676,4168),
              ("3g",15.18,4584,3183),("3h",14.20,4574,3163),("3i",14.67,4436,3284),
              ("3j",31.35,4828,3156),("3k",10.74,4814,2963),("untreated",0,936,164)]
    frame = pd.DataFrame(source, columns=["source_compound_id", "dose_um", "Q2_events", "Q4_events"])
    frame["all_events"] = 10000
    frame["annexin_positive_percent_from_events"] = (frame.Q2_events + frame.Q4_events) / 100
    frame["time_hours"] = 24
    frame["biological_replicate_status"] = "one_reported_panel_no_replicate_series"
    frame["chemical_structure_status"] = "source_structure_registration_pending"
    frame["use_for_training"] = False
    frame.to_csv(directory / "TableS3_event_transcription_review.csv", index=False, encoding="utf-8-sig")
    print("Mphahlele source panels transcribed:", len(frame), flush=True)


if __name__ == "__main__":
    main()
