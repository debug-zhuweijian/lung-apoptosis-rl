"""Download declared Bothe RNA/reference bytes with exact size and MD5 checks.

This is data acquisition only: it never installs tools, indexes, aligns, counts,
fits, loads checkpoints or runs a response model. Existing completed files are
verified; interrupted .part bytes are resumed only with an exact Content-Range.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '08-datasets/Bothe2021/E-MTAB-9923'
REF = DATA / 'reference/current-ucsc-hg19-20261011'
MAPPING = ROOT / 'experiments/20261011-data-alignment-round17/bothe_protocol_source/raw_sample_mapping.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    return {'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}


def validate_response(response, offset, expected_size):
    if offset:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if (response.status_code != 206 or not match or int(match[1]) != offset
                or int(match[2]) != expected_size - 1 or int(match[3]) != expected_size):
            raise ValueError('EXACT_RESUME_RANGE_REQUIRED')
    elif response.status_code != 200:
        raise ValueError('FULL_DOWNLOAD_HTTP_200_REQUIRED')
    if response.headers.get('Content-Encoding', 'identity') not in ('identity', ''):
        raise ValueError('UNEXPECTED_TRANSFER_ENCODING')
    if int(response.headers.get('Content-Length', -1)) != expected_size - offset:
        raise ValueError('DECLARED_RESPONSE_LENGTH_MISMATCH')


def download(item, target):
    target = Path(target).resolve()
    if (not target.is_relative_to(DATA.resolve()) or type(item['expected_bytes']) is not int
            or item['expected_bytes'] <= 0
            or not re.fullmatch('[a-f0-9]{32}', item['expected_md5'])):
        raise ValueError('INVALID_DECLARED_DATA_TARGET')
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    if not target.exists():
        if part.exists() and part.stat().st_size > item['expected_bytes']:
            raise ValueError('PART_EXCEEDS_DECLARED_SIZE')
        attempts = []
        for attempt in range(3):
            offset = part.stat().st_size if part.exists() else 0
            if offset == item['expected_bytes']:
                break
            try:
                headers = {'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with requests.get(item['url'], headers=headers, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    validate_response(response, offset, item['expected_bytes'])
                    current, last = offset, time.monotonic()
                    with part.open('ab' if offset else 'xb') as stream:
                        for block in response.iter_content(1024 * 1024):
                            if current + len(block) > item['expected_bytes']:
                                raise ValueError('TRANSFER_EXCEEDS_DECLARED_SIZE')
                            stream.write(block)
                            current += len(block)
                            if time.monotonic() - last >= 15:
                                print(json.dumps({'file': target.name, 'bytes': current,
                                    'expected_bytes': item['expected_bytes'], 'status': 'downloading'}), flush=True)
                                last = time.monotonic()
                    if current != item['expected_bytes']:
                        raise IOError('TRUNCATED_TRANSFER')
                break
            except (requests.RequestException, IOError) as error:
                attempts.append({'attempt': attempt + 1, 'error_type': type(error).__name__})
                print(json.dumps({'file': target.name, 'retry': attempts[-1]}), flush=True)
                if attempt == 2:
                    raise
        if not part.exists() or part.stat().st_size != item['expected_bytes']:
            raise IOError('DOWNLOAD_INCOMPLETE')
        digest = hashes(part)
        if digest['md5'] != item['expected_md5']:
            raise ValueError('PUBLISHER_MD5_MISMATCH_PART_PRESERVED')
        if target.exists():
            raise FileExistsError('COMPLETED_TARGET_APPEARED')
        part.rename(target)
    digest = hashes(target)
    if target.stat().st_size != item['expected_bytes'] or digest['md5'] != item['expected_md5']:
        raise ValueError('EXISTING_FILE_DECLARATION_MISMATCH')
    result = {**item, 'path': target.relative_to(ROOT).as_posix(), 'actual_bytes': target.stat().st_size,
              **digest, 'status': 'size_and_publisher_MD5_verified'}
    print(json.dumps({'file': target.name, 'status': result['status'], 'bytes': result['actual_bytes']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genome', action='store_true')
    parser.add_argument('--run', action='append', default=[])
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError('IMMUTABLE_RECEIPT_EXISTS')
    jobs = []
    if args.genome:
        declaration = read(REF / 'genome_declaration.json')
        jobs.append((declaration, REF / declaration['file']))
    mapping = {r['run']: r for r in read(MAPPING)}
    if len(args.run) != len(set(args.run)):
        raise ValueError('DUPLICATE_RUN')
    for run in args.run:
        if run not in mapping:
            raise ValueError('UNREGISTERED_BOTHE_RUN')
        row = mapping[run]
        for index, (url, checksum) in enumerate(zip(row['fastq_ftp'].split(';'), row['md5'].split(';')), 1):
            item = {'url': 'https://' + url, 'file': f'{run}_{index}.fastq.gz',
                    'expected_bytes': row[f'R{index}_bytes'], 'expected_md5': checksum,
                    'run': run, 'sample': row['sample'], 'history': row['history'], 'read_end': index}
            jobs.append((item, DATA / 'raw' / run / item['file']))
    if not jobs:
        parser.error('select --genome and/or registered --run')
    records = [download(item, target) for item, target in jobs]
    receipt = {'source_mapping_sha256': hashes(MAPPING)['sha256'],
        'genome_declaration_sha256': hashes(REF / 'genome_declaration.json')['sha256'],
        'downloader_source_sha256': hashes(__file__)['sha256'], 'files': records,
        'raw_counts_recovered': False, 'real_model_training_or_inference': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
