"""Independent, read-only completion audit; partial files never pass.

The running acquisition coordinator is deliberately left intact. Its summary
alone is not an independent raw-file completion certificate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '08-datasets/Bothe2021/E-MTAB-9923'
MAPPING = ROOT / 'experiments/20261011-data-alignment-round17/bothe_protocol_source/raw_sample_mapping.json'
PINNED_MAPPING = 'eac6bced59a1c41e3ecea8311590edc89e6c288ba591cd9587e9bccc31e86ec0'
PINNED_DOWNLOADER = '45ad5f94caaed941b36cfeca69b277cf62c6aaa597549be98786cf74b63fa540'
PILOT = ROOT / 'experiments/20261011-data-alignment-round18/bothe_download_pilot.json'
REMAINING = ROOT / 'experiments/20261011-data-alignment-round19/raw_remaining_acquisition'


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    return {'md5': md5.hexdigest(), 'sha256': sha.hexdigest()}


def verify_pair(row, receipt, *, verify_files=True):
    """No metadata-only result can certify files or expression/model readiness."""
    if (not isinstance(receipt, dict) or not isinstance(receipt.get('files'), list)
            or any(not isinstance(entry, dict) for entry in receipt['files'])):
        raise ValueError('DOWNLOAD_RECEIPT_OBJECT_AND_FILE_RECORDS_REQUIRED')
    if receipt.get('source_mapping_sha256') != PINNED_MAPPING or receipt.get('downloader_source_sha256') != PINNED_DOWNLOADER:
        raise ValueError('DOWNLOAD_SOURCE_IDENTITY_MISMATCH')
    records = [entry for entry in receipt.get('files', []) if 'run' in entry]
    if (len(records) != 2 or any(type(e.get('read_end')) is not int for e in records)
            or {e['read_end'] for e in records} != {1, 2}):
        raise ValueError('EXACT_TWO_READ_ENDS_REQUIRED')
    checksums, urls = row['md5'].split(';'), row['fastq_ftp'].split(';')
    if len(checksums) != 2 or len(urls) != 2:
        raise ValueError('INVALID_DECLARED_PAIR')
    observed = []
    for entry in sorted(records, key=lambda e: e['read_end']):
        end = entry['read_end']
        filename = f"{row['run']}_{end}.fastq.gz"
        path = DATA / 'raw' / row['run'] / filename
        expected_bytes, expected_md5 = row[f'R{end}_bytes'], checksums[end - 1]
        expected = {'run': row['run'], 'sample': row['sample'], 'history': row['history'],
                    'read_end': end, 'url': 'https://' + urls[end - 1], 'file': filename,
                    'path': path.relative_to(ROOT).as_posix(), 'expected_bytes': expected_bytes,
                    'actual_bytes': expected_bytes, 'expected_md5': expected_md5,
                    'md5': expected_md5, 'status': 'size_and_publisher_MD5_verified'}
        if (any(entry.get(k) != v for k, v in expected.items())
                or type(entry.get('expected_bytes')) is not int
                or type(entry.get('actual_bytes')) is not int
                or not isinstance(entry.get('sha256'), str)
                or not re.fullmatch('[a-f0-9]{64}', entry['sha256'])):
            raise ValueError('DECLARED_PAIR_RECEIPT_FIELD_MISMATCH')
        if verify_files:
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError('ACTUAL_COMPLETE_FILE_SIZE_MISMATCH')
            digest = hashes(path)
            if digest['md5'] != expected_md5 or digest['sha256'] != entry['sha256']:
                raise ValueError('ACTUAL_FILE_MD5_OR_SHA256_MISMATCH')
        observed.append({'read_end': end, 'path': expected['path'], 'bytes': expected_bytes,
                         'md5': expected_md5, 'sha256': entry['sha256']})
    return {'run': row['run'], 'sample': row['sample'], 'files': observed,
            'actual_file_bytes_MD5_SHA256_verified': verify_files is True,
            'verification_level': 'actual_complete_files' if verify_files else 'receipt_fields_only',
            'FASTQ_content_or_pair_sync_verified': False, 'RNA_counts_recovered': False}


def audit():
    if hashes(MAPPING)['sha256'] != PINNED_MAPPING:
        raise ValueError('FROZEN_SAMPLE_MAPPING_CHANGED')
    rows = json.loads(MAPPING.read_text(encoding='utf-8'))
    if len(rows) != 12 or len({row['run'] for row in rows}) != 12:
        raise ValueError('INVALID_12_RUN_AXIS')
    verified, pending, rejected = [], [], []
    for row in rows:
        receipt = PILOT if row['run'] == 'ERR5000305' else REMAINING / (row['run'] + '.completed.json')
        if not receipt.exists():
            pending.append(row['run'])
            continue
        try:
            value = verify_pair(row, json.loads(receipt.read_text(encoding='utf-8')), verify_files=True)
            verified.append({**value, 'receipt_sha256': hashes(receipt)['sha256']})
        except (ValueError, OSError, KeyError, TypeError) as error:
            rejected.append({'run': row['run'], 'error_type': type(error).__name__, 'error': str(error)})
    return {'schema': 'bothe-independent-download-completion-v1',
            'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'complete_files_verified' if len(verified) == 12 else 'pending_or_rejected_receipts',
            'verified_runs': verified, 'pending_runs': pending, 'rejected_receipts': rejected,
            'total_declared_runs': 12, 'total_declared_bytes': 60056725984,
            'coordinator_summary_used_as_qualification': False,
            'validator_sha256': hashes(Path(__file__))['sha256'],
            'source_mapping_sha256': PINNED_MAPPING, 'FASTQ_content_or_pair_sync_verified': False,
            'RNA_counts_recovered': False, 'model_forward_training_or_PT_load': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('NEW_COMPLETION_AUDIT_REQUIRED_NO_OVERWRITE')
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'verified_runs': len(result['verified_runs']),
                      'pending_runs': len(result['pending_runs']), 'rejected': len(result['rejected_receipts'])}))
    if result['rejected_receipts']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
