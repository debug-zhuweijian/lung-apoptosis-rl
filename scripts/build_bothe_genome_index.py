"""Build a new portable Subread data index, never a training/model artifact."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '08-datasets/Bothe2021/E-MTAB-9923'
GENOME = DATA / 'reference/current-ucsc-hg19-20261011/hg19.fa.gz'
OUT = DATA / 'reference/subread-hg19-index-v1'
EXE = DATA / 'tools/subread-2.1.1/subread-2.1.1-Windows-x86_64/bin/subread-buildindex.exe'
EXPECTED_EXE_SHA = '3a28c755b57cc33ec108d269125c7192aeb76c0c60351ff976c65cb3662acf02'
EXPECTED_GENOME_MD5 = '806c02398f5ac5da8ffd6da2d1d5d1a9'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('NEW_INDEX_VERSION_REQUIRED_NO_OVERWRITE')
    if GENOME.stat().st_size != 948731419 or sha(EXE) != EXPECTED_EXE_SHA:
        raise ValueError('PINNED_GENOME_SIZE_OR_EXECUTABLE_CHANGED')
    with GENOME.open('rb') as stream:
        if hashlib.file_digest(stream, 'md5').hexdigest() != EXPECTED_GENOME_MD5:
            raise ValueError('PUBLISHER_GENOME_MD5_MISMATCH')
    OUT.mkdir(parents=True)
    arguments = [str(EXE), '-M', '8000', '-o', 'reference/subread-hg19-index-v1/hg19',
                 'reference/current-ucsc-hg19-20261011/hg19.fa.gz']
    run = {'schema': 'bothe-native-data-index-v1', 'phase': 'genome_index',
        'method': 'Subread2.1.1 new processing version not author STAR reconstruction',
        'argv': arguments, 'cwd': str(DATA), 'genome_bytes': GENOME.stat().st_size,
        'genome_md5': EXPECTED_GENOME_MD5, 'genome_sha256': sha(GENOME),
        'executable_sha256': EXPECTED_EXE_SHA, 'runner_sha256': sha(Path(__file__)),
        'status': 'running', 'started_unix': time.time(), 'RNA_counts_recovered': False,
        'model_forward_training_or_PT_load': False}
    with (OUT / 'run.json').open('x', encoding='utf-8') as stream:
        json.dump(run, stream, indent=2, ensure_ascii=False)
    print(json.dumps({'status': 'starting_genome_index', 'genome_MD5_verified': True,
                      'maximum_index_memory_MB': 8000}), flush=True)
    with (OUT / 'index.log').open('xb') as log:
        result = subprocess.run(arguments, cwd=DATA, stdout=log, stderr=subprocess.STDOUT, check=False)
    run.update({'status': 'completed' if result.returncode == 0 else 'failed',
                'exit_code': result.returncode, 'finished_unix': time.time()})
    outputs = list(OUT.glob('hg19*'))
    if result.returncode == 0 and not outputs:
        run['status'] = 'failed_missing_index_files'
    if run['status'] == 'completed':
        run['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in outputs}
        run['index_usable_for_alignment_not_yet_verified'] = True
    (OUT / 'run.json').write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': run['status'], 'exit_code': result.returncode,
                      'index_files': len(outputs), 'RNA_counts_recovered': False}), flush=True)
    if run['status'] != 'completed':
        raise SystemExit(result.returncode or 1)


if __name__ == '__main__':
    main()
