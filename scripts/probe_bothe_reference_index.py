"""Reference-DNA-derived synthetic index smoke; never registered RNA or ML."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / '08-datasets/Bothe2021/E-MTAB-9923'
OUT = ROOT / 'experiments/20261011-data-alignment-round19/reference_index_probe'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def new_json(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2)


def sam_rows(path):
    """Keep native header bytes separate; decode every alignment strictly."""
    headers, rows = [], []
    with Path(path).open('rb') as f:
        for raw in f:
            if raw.startswith(b'@'):
                try:
                    raw.decode('utf8')
                except UnicodeDecodeError:
                    headers.append(raw.split(b'\t', 1)[0].decode('ascii'))
            else:
                rows.append(raw.decode('utf8').rstrip('\r\n').split('\t'))
    return headers, rows


def coordinate_match(pair, rows):
    if len(rows) != 2 or sorted(int(x[1]) & 192 for x in rows) != [64, 128]:
        return False
    for row in rows:
        flag = int(row[1])
        mate = 1 if flag & 64 else 2
        if row[2] != pair['chr'] or int(row[3]) != pair[f'R{mate}_pos'] or row[5] != '75M':
            return False
        if bool(flag & 16) != (mate == 2) or bool(flag & 32) != (mate == 1):
            return False
    return True


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    annotation = DATA / 'reference/unique-introns-v1'
    with (annotation / 'unique_introns.saf').open() as f:
        candidates = [x for x in csv.DictReader(f, delimiter='\t')
                      if x['Chr'] == 'chr1' and int(x['End']) - int(x['Start']) >= 600
                      and 1000000 < int(x['Start']) < 2000000][:10]
    last = max(int(x['Start']) + 400 for x in candidates)
    pieces, chromosome, position = [], '', 0
    with gzip.open(DATA / 'reference/current-ucsc-hg19-20261011/hg19.fa.gz', 'rt') as f:
        for line in f:
            if line.startswith('>'):
                chromosome, position = line[1:].strip().split()[0], 0
            elif chromosome == 'chr1':
                s = line.strip().upper()
                pieces.append(s)
                position += len(s)
                if position >= last:
                    break
    genome = ''.join(pieces)
    recipe, reads = [], {1: [], 2: []}
    for i, x in enumerate(candidates):
        a = int(x['Start']) + 100
        b = a + 225
        first, second = genome[a-1:a+74], genome[b-1:b+74]
        if set(first + second) - set('ACGT'):
            continue
        name = f'reference_synthetic_probe_{i:02d}'
        recipe.append({'qname': name, 'feature': x['GeneID'], 'chr': 'chr1',
                       'R1_pos': a, 'R2_pos': b, 'length': 75})
        second = second.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
        for mate, sequence in [(1, first), (2, second)]:
            reads[mate].append(f'@{name}/{mate}\n{sequence}\n+\n'+ 'I'*75+'\n')
    assert recipe and len(recipe) <= 20
    for mate in [1, 2]:
        with gzip.GzipFile(filename=str(OUT/f'R{mate}.fastq.gz'), mode='wb', mtime=0) as f:
            f.write(''.join(reads[mate]).encode('ascii'))
    (OUT/'recipe.json').write_text(json.dumps({'nature': 'reference DNA synthetic; not measured RNA',
        'pairs': recipe, 'genome_source_run': json.loads((DATA/'reference/subread-hg19-index-v1/run.json').read_text()),
        'annotation_manifest_sha256': sha(annotation/'manifest.json'),
        'fastq_sha256': {f'R{i}': sha(OUT/f'R{i}.fastq.gz') for i in [1, 2]}}, indent=2), encoding='utf8')
    print('prepared', len(recipe), flush=True)


def execute():
    from src.lung.data.paired_alignment_filter import filter_sam
    from src.lung.data.bothe_count_matrix import annotation_axis, parse_featurecounts
    occupied = [p for p in OUT.iterdir() if p.name not in
                {'recipe.json', 'readiness_receipt.json', 'README.md', 'R1.fastq.gz', 'R2.fastq.gz',
                 'encoding_boundary_fixture.sam', 'runner_light_checks.json'}]
    if occupied:
        raise FileExistsError('NEW_PROBE_VERSION_REQUIRED_NO_OVERWRITE')
    snapshot = json.loads((OUT/'recipe.json').read_text())
    recipe = snapshot['pairs']
    for mate in [1, 2]:
        if sha(OUT/f'R{mate}.fastq.gz') != snapshot['fastq_sha256'][f'R{mate}']:
            raise ValueError('PREPARED_FASTQ_BINDING_CHANGED')
    annotation = DATA/'reference/unique-introns-v1'
    if sha(annotation/'manifest.json') != snapshot['annotation_manifest_sha256']:
        raise ValueError('ANNOTATION_MANIFEST_CHANGED')
    index = DATA/'reference/subread-hg19-index-v1'
    current_run = json.loads((index/'run.json').read_text())
    if current_run != snapshot['genome_source_run'] or current_run['status'] != 'completed' or current_run['exit_code'] != 0 or len(current_run['files']) != 6:
        raise ValueError('COMPLETED_INDEX_RUN_CHANGED')
    for name, meta in current_run['files'].items():
        if (index/name).stat().st_size != meta['bytes'] or sha(index/name) != meta['sha256']:
            raise ValueError('INDEX_FILE_COMPLETION_BINDING_CHANGED')
    completion_path = ROOT/'experiments/20261011-data-alignment-round18/index_completion_validation.json'
    completion = json.loads(completion_path.read_text(encoding='utf8'))
    if completion['run_json_sha256'] != sha(index/'run.json') or completion['independently_verified_files'] != current_run['files']:
        raise ValueError('ROOT_COMPLETION_RECEIPT_BINDING_CHANGED')
    genes, features = annotation_axis()
    exe = DATA/'tools/subread-2.1.1/subread-2.1.1-Windows-x86_64/bin'
    assert sha(exe/'subjunc.exe') == 'b26da8b3f41ed706e687d503ac4fb4aeb79709b1857ed2c9871dffec02dc4743'
    assert sha(exe/'featureCounts.exe') == '30cf41c2bac4707754b4d7faab96743a16957bdb21ecee82e64fd941905185d9'
    source_hashes = {'runner': sha(Path(__file__)),
        'filter': sha(ROOT/'src/lung/data/paired_alignment_filter.py'),
        'parser': sha(ROOT/'src/lung/data/bothe_count_matrix.py')}
    new_json(OUT/'execution_binding.json', {'nature': 'reference-DNA synthetic not real RNA',
        'schema': 'reference-index-probe-v2', 'source_sha256': source_hashes,
        'recipe_sha256': sha(OUT/'recipe.json'), 'index_run_sha256': sha(index/'run.json'),
        'index_file_hashes': current_run['files'],
        'index_hash_validation': 'all six file SHA256 recomputed before execution against exact completed run and root completion receipt',
        'root_completion_receipt_sha256': sha(completion_path),
        'filter_kwargs': 'filter_sam(source, new output_dir); existing policy',
        'parser_kwargs': {'column_samples': {'strict_pairs/filtered.sam': 'synthetic_probe'}, 'frozen_annotation': True}})
    commands = []
    def run(label, argv):
        with (OUT/(label+'.log')).open('xb') as f:
            result = subprocess.run(argv, cwd=OUT, stdout=f, stderr=subprocess.STDOUT)
        commands.append({'label': label, 'argv': argv, 'exit': result.returncode})
        new_json(OUT/(label+'.command.json'), commands[-1])
        if result.returncode:
            raise RuntimeError(label+' failed')
    run('subjunc', [str(exe/'subjunc.exe'), '-T', '1', '-P', '3', '-n', '14', '-m', '1', '-p', '1',
        '-M', '3', '-I', '5', '-d', '50', '-D', '600', '-S', 'fr', '-i',
        '../../../08-datasets/Bothe2021/E-MTAB-9923/reference/subread-hg19-index-v1/hg19',
        '-r', 'R1.fastq.gz', '-R', 'R2.fastq.gz', '--SAMoutput', '-o', 'synthetic.sam'])
    filtered = filter_sam(OUT/'synthetic.sam', OUT/'strict_pairs')
    run('featureCounts', [str(exe/'featureCounts.exe'), '-T', '1', '-F', 'SAF', '-a',
        '../../../08-datasets/Bothe2021/E-MTAB-9923/reference/unique-introns-v1/unique_introns.saf',
        '-o', 'synthetic_counts.tsv', '-f', '-p', '--countReadPairs', '-B', '-C', '--primary',
        '-Q', '10', '-s', '0', 'strict_pairs/filtered.sam'])
    totals = parse_featurecounts(OUT/'synthetic_counts.tsv', {'strict_pairs/filtered.sam': 'synthetic_probe'}, features)
    observed = {}
    non_utf8_headers, data_rows = sam_rows(OUT/'strict_pairs/filtered.sam')
    for fields in data_rows:
        observed.setdefault(fields[0], []).append(fields)
    expected, coordinate_checks = {}, []
    for pair in recipe:
        rows = observed.get(pair['qname'], [])
        matched = coordinate_match(pair, rows)
        coordinate_checks.append({'qname': pair['qname'], 'exact_coordinate_match': matched, 'retained_records': len(rows)})
        if matched:
            gene = features[pair['feature']][0]
            expected[gene] = expected.get(gene, 0)+1
    actual = {k: v for k, v in totals['synthetic_probe'].items() if v}
    result = {'nature': 'reference-derived synthetic DNA reads; no real RNA counts', 'commands': commands,
        'filter': filtered, 'coordinate_checks': coordinate_checks, 'expected_nonzero_gene_counts': expected,
        'actual_nonzero_gene_counts': actual, 'full_annotation_features': len(features),
        'oracle_passed': expected == actual and set(observed) == {p['qname'] for p in recipe} and all(x['exact_coordinate_match'] for x in coordinate_checks),
        'non_utf8_header_types': non_utf8_headers, 'alignment_data_encoding': 'strict UTF8; native headers skipped as bytes',
        'execution_source_sha256': source_hashes,
        'filter_code_sha256': sha(ROOT/'src/lung/data/paired_alignment_filter.py'),
        'parser_code_sha256': sha(ROOT/'src/lung/data/bothe_count_matrix.py'),
        'output_sha256': {name: sha(OUT/name) for name in ['synthetic.sam', 'strict_pairs/filtered.sam', 'synthetic_counts.tsv']}}
    new_json(OUT/'result.json', result)
    print(json.dumps({'oracle_passed': result['oracle_passed'], 'actual': actual}), flush=True)
    if not result['oracle_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    prepare() if sys.argv[1] == 'prepare' else execute()
