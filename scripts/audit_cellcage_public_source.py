"""Bounded public metadata audit for CellCage A549; never downloads raw data."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/20261011-data-alignment-round22/cellcage_source_recovery"
DATA = ROOT / "08-datasets/CellCage2026/PMC13174582-v1/source-recovery-round22-v1"
DOI = "10.64898/2026.05.05.723030"
TERM = 'Cellanome OR CellCage OR "2026.05.05.723030"'


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Anchors(HTMLParser):
    def __init__(self, url):
        super().__init__()
        self.url, self.links = url, []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("mailto:", "javascript:", "#")):
                self.links.append(urljoin(self.url, href))


def fetch(name, url, params=None):
    record = {"name": name, "url": url, "params": params,
              "started_UTC": datetime.now(timezone.utc).isoformat(), "method": "GET"}
    try:
        response = requests.get(url, params=params, timeout=(10, 30), headers={
            "User-Agent": "CellCageSourceAudit/1.0 public-metadata-only",
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.1"})
        raw = response.content
        content_type = response.headers.get("Content-Type", "")
        suffix = (".json" if "json" in content_type else ".xml" if "xml" in content_type
                  else ".txt" if "text/plain" in content_type else ".html")
        path = DATA / "metadata" / (name + suffix)
        with path.open("xb") as stream:
            stream.write(raw)
        record.update(request_url=response.request.url, final_url=response.url,
                      HTTP_status=response.status_code, bytes=len(raw),
                      sha256=hashlib.sha256(raw).hexdigest(),
                      source_file=str(path.relative_to(ROOT)).replace("\\", "/"),
                      response_headers={k: response.headers[k] for k in (
                          "Content-Type", "Content-Length", "Last-Modified", "X-RateLimit-Remaining")
                          if k in response.headers})
        value = None
        if response.status_code == 200:
            try:
                value = response.json()
                record["actual_JSON_top_level"] = sorted(value) if isinstance(value, dict) else "array"
            except ValueError:
                parser = Anchors(response.url)
                parser.feed(response.text)
                record["actual_anchor_links"] = sorted(set(parser.links))
        else:
            record["access_result"] = "HTTP_failure_not_evidence_of_absent_data"
        if isinstance(value, dict) and "esearchresult" in value:
            record["actual_esearch"] = value["esearchresult"]
        elif name == "github_org_public_repos" and isinstance(value, list):
            record["public_repository_metadata"] = [{k: r.get(k) for k in (
                "full_name", "description", "html_url", "default_branch", "archived")}
                for r in value]
            record["next_page"] = response.links.get("next")
        elif name == "github_repository_search" and isinstance(value, dict):
            record["search_count"] = value.get("total_count")
            record["repository_metadata"] = [{k: r.get(k) for k in (
                "full_name", "description", "html_url", "default_branch")}
                for r in value.get("items", [])]
        elif name == "biorxiv_record" and isinstance(value, dict):
            record["paper_records"] = value.get("collection", [])
        elif name.startswith("ENA_") and isinstance(value, list):
            record["actual_study_metadata"] = value
        elif name == "zenodo_records" and isinstance(value, dict):
            record["zenodo_total"] = value.get("hits", {}).get("total")
            record["record_metadata"] = [{"id": r.get("id"), "metadata": r.get("metadata"),
                "files": r.get("files")} for r in value.get("hits", {}).get("hits", [])]
    except requests.RequestException as error:
        record.update(access_result="network_failure_not_evidence_of_absent_data",
                      exception_type=type(error).__name__, error=str(error))
    save_new(OUTPUT / (name + "_receipt.json"), record)
    print(json.dumps({k: record[k] for k in ("name", "HTTP_status", "bytes", "access_result",
        "search_count", "actual_esearch") if k in record}, ensure_ascii=False), flush=True)
    return record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (DATA / "metadata").mkdir(parents=True, exist_ok=True)
    # These are public source discovery queries, not an accession assertion.
    specs = [
        ("biorxiv_record", "https://api.biorxiv.org/details/biorxiv/" + DOI, None),
        ("github_org_public_repos", "https://api.github.com/orgs/cellanome/repos", {"per_page": 100, "type": "public"}),
        ("github_repository_search", "https://api.github.com/search/repositories", {"q": "cellanome", "per_page": 100}),
        ("NCBI_GEO", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "gds", "term": TERM, "retmode": "json", "retmax": 100}),
        ("NCBI_BioProject", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "bioproject", "term": TERM, "retmode": "json", "retmax": 100}),
        ("NCBI_SRA", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", {"db": "sra", "term": TERM, "retmode": "json", "retmax": 100}),
        ("ENA_study", "https://www.ebi.ac.uk/ena/portal/api/search", {"result": "study", "query": 'study_title="*CellCage*" OR study_title="*Cellanome*"', "fields": "study_accession,study_title", "format": "json", "limit": 100}),
        ("europepmc_databaseLinks", "https://www.ebi.ac.uk/europepmc/webservices/rest/MED/42146668/databaseLinks", {"format": "json"}),
        ("zenodo_records", "https://zenodo.org/api/records", {"q": '"' + DOI + '" OR Cellanome OR CellCage', "size": 25}),
        ("official_paper_page", "https://www.cellanome.com/publications/scalable-longitudinal-imaging-transcriptomics-dynamic-enclosures", None),
        ("official_demos_data", "https://www.cellanome.com/resources-demos-data", None),
        ("official_products", "https://www.cellanome.com/products", None),
    ]
    records = [fetch(*spec) for spec in specs]
    save_new(OUTPUT / "public_metadata_fetches_v1.json", {
        "schema": "CellCage_public_source_recovery_v1", "paper_DOI": DOI,
        "sources": records, "raw_imaging_or_sequencing_downloaded": False,
        "real_PT_NN_forward_fit": 0, "training_roles_changed": False,
        "source_specific_metadata_version": str(DATA.relative_to(ROOT)).replace("\\", "/")})


if __name__ == "__main__":
    main()
