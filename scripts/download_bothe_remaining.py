"""Acquire the other eleven declared RNA runs, with two independent workers.

The pilot ERR5000305 is owned by another running job and is excluded. This
script never indexes, counts, installs software, or invokes a response model.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / 'experiments/20261011-data-alignment-round17/bothe_protocol_source/raw_sample_mapping.json'
DOWNLOADER = ROOT / 'scripts/download_bothe_inputs.py'
OUT = ROOT / 'experiments/20261011-data-alignment-round19/raw_remaining_acquisition'
RAW = ROOT / '08-datasets/Bothe2021/E-MTAB-9923/raw'
PINNED_MAPPING_SHA = 'eac6bced59a1c41e3ecea8311590edc89e6c288ba591cd9587e9bccc31e86ec0'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    if sha(MAPPING) != PINNED_MAPPING_SHA:
        raise ValueError('FROZEN_SAMPLE_MAPPING_CHANGED')
    rows = json.loads(MAPPING.read_text(encoding='utf-8'))
    if len(rows) != 12 or len({r['run'] for r in rows}) != 12:
        raise ValueError('INVALID_DECLARED_RUN_AXIS')
    rows = sorted((r for r in rows if r['run'] != 'ERR5000305'),
                  key=lambda r: (r['R1_bytes'] + r['R2_bytes'], r['run']))
    if len(rows) != 11:
        raise ValueError('PILOT_EXCLUSION_CHANGED')
    # Never compete for an existing run owned by an earlier acquisition job.
    for row in rows:
        if (RAW / row['run']).exists():
            raise FileExistsError('RUN_DIRECTORY_ALREADY_EXISTS: ' + row['run'])
    OUT.mkdir(parents=True, exist_ok=False)
    plan = {
        'schema': 'bothe-remaining-raw-acquisition-v1',
        'created_at_utc': utc(), 'parallel_download_workers': 2,
        'excluded_pilot_run': 'ERR5000305',
        'excluded_pilot_owner_root_exec_session': 28793,
        'source_mapping_sha256': sha(MAPPING),
        'downloader_sha256': sha(DOWNLOADER),
        'coordinator_sha256': sha(Path(__file__)),
        'runs': [{'run': r['run'], 'sample': r['sample'], 'history': r['history'],
                  'expected_pair_bytes': r['R1_bytes'] + r['R2_bytes']} for r in rows],
        'expected_remaining_bytes': sum(r['R1_bytes'] + r['R2_bytes'] for r in rows),
        'RNA_counts_recovered': False, 'model_forward_training_or_PT_load': False,
    }
    (OUT / 'jobs.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    results = {}

    def acquire(row):
        run = row['run']
        receipt = OUT / (run + '.completed.json')
        argv = [sys.executable, str(DOWNLOADER), '--run', run, '--receipt', str(receipt)]
        if sha(DOWNLOADER) != plan['downloader_sha256'] or sha(MAPPING) != PINNED_MAPPING_SHA:
            raise ValueError('ACQUISITION_SOURCE_CHANGED')
        with (OUT / (run + '.log')).open('xb') as log:
            process = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        result = {'run': run, 'exit_code': process.returncode,
                  'completed_at_utc': utc(), 'receipt_exists': receipt.exists()}
        if process.returncode == 0:
            value = json.loads(receipt.read_text(encoding='utf-8'))
            if (value['source_mapping_sha256'] != PINNED_MAPPING_SHA
                    or value['downloader_source_sha256'] != plan['downloader_sha256']
                    or len(value['files']) != 2
                    or {v['read_end'] for v in value['files']} != {1, 2}
                    or any(v['run'] != run or v['sample'] != row['sample']
                           or v['status'] != 'size_and_publisher_MD5_verified'
                           for v in value['files'])):
                raise ValueError('DECLARED_DOWNLOAD_RECEIPT_MISMATCH: ' + run)
            result['receipt_sha256'] = sha(receipt)
            result['status'] = 'complete_declared_pair_MD5_verified'
        else:
            result['status'] = 'failed_preserved_partial_no_restart'
        return result

    print(json.dumps({'status': 'starting', 'runs': 11, 'workers': 2,
                      'expected_remaining_bytes': plan['expected_remaining_bytes']}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(acquire, row): row['run'] for row in rows}
        for future in as_completed(futures):
            run = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {'run': run, 'status': 'failed_preserved_partial_no_restart',
                          'error_type': type(error).__name__, 'observed_at_utc': utc()}
            results[run] = result
            state = {'schema': plan['schema'], 'results': results,
                     'remaining_runs': [r['run'] for r in rows if r['run'] not in results],
                     'updated_at_utc': utc(), 'RNA_counts_recovered': False}
            (OUT / 'progress.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(result), flush=True)
    success = all(v['status'] == 'complete_declared_pair_MD5_verified' for v in results.values())
    final = {'status': 'complete' if success else 'incomplete_failed_runs_preserved',
             'results': results, 'finished_at_utc': utc(), 'RNA_counts_recovered': False,
             'model_forward_training_or_PT_load': False}
    with (OUT / 'completion.json').open('x', encoding='utf-8') as stream:
        json.dump(final, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
