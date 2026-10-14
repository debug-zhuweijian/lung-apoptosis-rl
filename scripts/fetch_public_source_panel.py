"""Retrieve the bounded, version-bound public source panel; no FASTQ or training.

Download status is not field validation. Failed responses/partials are retained,
including HTTP-success error documents, and never renamed as accepted sources.
"""
import argparse, concurrent.futures, datetime, hashlib, json, pathlib, urllib.request, zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    repo = pathlib.Path(__file__).resolve().parents[1]
    known = json.loads((repo/'docs/DOT_RESEARCH_EVIDENCE_2026-10-14.json').read_text())['sources']
    selected = [x for x in known if x['local_name'] in {
        'GSE108214.soft.gz', 'Sarin_S8.docx', 'HM_family.soft.gz', 'HM_counts.tsv.gz',
        'GSE161800.soft.gz', 'GSE161800_1vs2_counts.csv.gz', 'GSE161800_7vs8_counts.csv.gz',
        'GSE161800_paper.xml', 'GSE96779.soft.gz', 'GSE96779.fpkm.gz'}]
    extra = {
      'tidyi_A549_counts.csv': 'https://www.ebi.ac.uk/biostudies/files/E-MTAB-16354/SS3_Human_A549_counts.csv',
      'tidyi_A549_metadata.csv': 'https://www.ebi.ac.uk/biostudies/files/E-MTAB-16354/SS3_Human_A549_metadata.csv',
      'tidyi_study.json': 'https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-16354',
      'tahoe_gene_metadata.parquet': 'https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/gene_metadata.parquet',
      'tahoe_drug_metadata.parquet': 'https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/drug_metadata.parquet',
      'tahoe_sample_metadata.parquet': 'https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/sample_metadata.parquet',
      'tahoe_tree.json': 'https://huggingface.co/api/datasets/tahoebio/Tahoe-100M/tree/2dc57900b7981cfcf5e211527169a0b006546a95/metadata',
      'hgnc_complete_set.txt': 'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt',
      'Homo_sapiens.GRCh37.75.gtf.gz': 'https://ftp.ensembl.org/pub/release-75/gtf/homo_sapiens/Homo_sapiens.GRCh37.75.gtf.gz',
      'bothe_study.json': 'https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-9923',
      'bothe_sdrf.txt': 'https://www.ebi.ac.uk/biostudies/files/E-MTAB-9923/E-MTAB-9923.sdrf.txt',
      'bothe_DexDex_vs_EtOHDex.txt': 'https://www.ebi.ac.uk/biostudies/files/E-MTAB-9923/DEseq2_DexDexVsEtOHDex_A549_byintrons.txt',
      'bothe_DexEtOH_vs_EtOHEtOH.txt': 'https://www.ebi.ac.uk/biostudies/files/E-MTAB-9923/DEseq2_DexEtOHvsEtOHEtOH_A549_byintrons.txt',
      'bothe_EtOHDex_vs_EtOHEtOH.txt': 'https://www.ebi.ac.uk/biostudies/files/E-MTAB-9923/DEseq2_EtOHDexVsEtOHEtOH_A549_byintrons.txt',
      'bothe_paper.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8403771/fullTextXML',
      'Lee2012_paper.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3501264/fullTextXML',
      'Lee2012_supplements.zip': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC3501264/supplementaryFiles'}
    selected += [dict(local_name=k, url=v) for k,v in extra.items()]
    expected = {'tidyi_A549_counts.csv': 'b33aa1b03d71972f8e8c33864be0f0e2747ffa8b56289d00f27cd8c276cc3360',
                'tidyi_A549_metadata.csv': '6ee300a8ba36642950c94d3be7455beb51255e42d8f311273d00aa9e0868a38b'}
    def fetch(item):
        rec = dict(item, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        partial = root/(item['local_name']+'.partial')
        try:
            req=urllib.request.Request(item['url'], headers={'User-Agent':'lung-apoptosis-rl-public-source-audit/1.0'})
            with urllib.request.urlopen(req,timeout=90) as response, partial.open('wb') as out:
                rec['final_url']=response.url
                rec['http_status']=response.status
                rec['last_modified']=response.headers.get('Last-Modified')
                rec['etag']=response.headers.get('ETag')
                claimed=response.headers.get('Content-Length')
                rec['content_length']=int(claimed) if claimed else None
                if claimed and int(claimed)>120_000_000:
                    raise ValueError('Beyond authorized small-source panel; not downloaded')
                size=0; sha=hashlib.sha256()
                while True:
                    b=response.read(1024*1024)
                    if not b: break
                    size+=len(b)
                    if size>120_000_000: raise ValueError('Beyond authorized small-source panel')
                    out.write(b); sha.update(b)
            rec.update(actual_bytes=size, actual_sha256=sha.hexdigest())
            if claimed and size!=int(claimed): raise ValueError('HTTP byte count mismatch')
            if item['local_name'].endswith('.zip') and not zipfile.is_zipfile(partial):
                raise ValueError('HTTP success returned a non-ZIP body; not a usable supplement')
            target_hash=item.get('sha256',expected.get(item['local_name']))
            rec['expected_sha256']=target_hash
            if target_hash and sha.hexdigest()!=target_hash: raise ValueError('Recorded source hash changed; retain partial for review')
            partial.rename(root/item['local_name']);rec['status']='downloaded_hash_checked' if target_hash else 'downloaded_new_snapshot'
        except Exception as exc:
            rec.update(status='failed', error=type(exc).__name__+': '+str(exc))
        print(json.dumps(rec,ensure_ascii=False),flush=True)
        return rec
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts=list(pool.map(fetch,selected))
    (root/'download_receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
