"""Recover author contrast statistics; never convert them into sample RNA counts.

This auxiliary history benchmark is separate from the registered 12-run count
consumer. RefSeq transcript versions, undefined tests and source units survive.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.recover_public_expression import digest

CONTRASTS = {
    'DEseq2_EtOHDexVsEtOHEtOH_A549_byintrons.txt': ('acute_response', 'ethanol', 'dexamethasone', 'ethanol', 'ethanol'),
    'DEseq2_DexEtOHvsEtOHEtOH_A549_byintrons.txt': ('residual_history', 'dexamethasone', 'ethanol', 'ethanol', 'ethanol'),
    'DEseq2_DexDexVsEtOHDex_A549_byintrons.txt': ('reinduction_history', 'dexamethasone', 'dexamethasone', 'ethanol', 'dexamethasone'),
}
COLUMNS = ['baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj']


def validate_contrast(frame):
    if list(frame.columns) != COLUMNS:
        raise ValueError('Author statistic schema differs; no sample-count conversion')
    if not frame.index.is_unique or not frame.index.to_series().str.fullmatch(r'N[MR]_\d+\.\d+(?:_\d+)?').all():
        raise ValueError('Unique versioned source RefSeq label required')
    values = frame.to_numpy(float)
    if np.isinf(values).any() or frame.baseMean.isna().any() or (frame.baseMean < 0).any():
        raise ValueError('Invalid mean/statistic')
    for col in ['pvalue', 'padj']:
        v = frame[col].dropna()
        if ((v < 0) | (v > 1)).any():
            raise ValueError('Invalid probability')
    if (frame.lfcSE.dropna() < 0).any():
        raise ValueError('Invalid standard error')
    # Author labels include _2, _3, ... suffixes. Retain them; stripping/summing
    # would incorrectly treat undocumented source rows as one measured gene.
    base = frame.index.to_series().str.replace(r'_\d+$', '', regex=True)
    return {'features': len(frame),
            'author_suffix_rows': int(frame.index.to_series().str.contains(r'\.\d+_\d+$').sum()),
            'rows_with_ambiguous_base_refseq': int(base.duplicated(keep=False).sum()),
            'undefined_adjusted_p': int(frame.padj.isna().sum()),
            'FDR_lt_0_001': int((frame.padj < .001).sum()),
            'FDR_lt_0_05_sensitivity_only': int((frame.padj < .05).sum()),
            'missing_by_field': {k: int(v) for k, v in frame.isna().sum().items()}}


def samples_from_sdrf(sdrf):
    if not sdrf['Characteristics[cell line]'].eq('A549').all():
        raise ValueError('Cell background changed')
    columns = ['Source Name', 'Comment[ENA_RUN]', 'Factor Value[stimulus]']
    samples = sdrf[columns].drop_duplicates()
    if len(samples) != 12 or samples['Source Name'].duplicated().any() or samples['Comment[ENA_RUN]'].duplicated().any():
        raise ValueError('Expected 12 distinct libraries/runs, not mate files as replicates')
    expected = {a + '+washout+' + b for a in ['dexamethasone', 'ethanol'] for b in ['dexamethasone', 'ethanol']}
    groups = samples.groupby('Factor Value[stimulus]').size()
    if set(groups.index) != expected or not groups.eq(3).all():
        raise ValueError('Four histories with three libraries each required')
    samples = samples.rename(columns={columns[0]: 'source_sample', columns[1]: 'ENA_run', columns[2]: 'history'})
    samples['endpoint_hours'] = 32
    samples['intermediate_RNA_measured'] = False
    samples['raw_counts_recovered'] = False
    return samples


def recover(source, output):
    if output.exists():
        raise FileExistsError('Use a new output version')
    receipts = json.loads((source / 'sources.json').read_text())
    by_name = {r['path']: r for r in receipts}
    needed = set(CONTRASTS) | {'E-MTAB-9923.sdrf.txt'}
    if not needed.issubset(by_name):
        raise ValueError('Missing official source receipts')
    for name in needed:
        if digest(source / name) != by_name[name]['sha256']:
            raise ValueError('Source bytes changed')
    samples = samples_from_sdrf(pd.read_csv(source / 'E-MTAB-9923.sdrf.txt', sep='\t'))
    stats, frames, ids = {}, [], None
    for name, (label, first, second, ref_first, ref_second) in CONTRASTS.items():
        frame = pd.read_csv(source / name, sep='\t', index_col=0)
        stats[label] = validate_contrast(frame)
        if ids is not None and set(frame.index) != ids:
            raise ValueError('Contrast feature universes differ')
        ids = set(frame.index)
        frame.index.name = 'source_feature_label'
        frame = frame.reset_index()
        for key, val in dict(contrast=label, numerator_first=first, numerator_second=second,
                             denominator_first=ref_first, denominator_second=ref_second).items():
            frame[key] = val
        frames.append(frame)
    output.mkdir(parents=True)
    samples.to_csv(output / 'source_samples.csv', index=False)
    pd.concat(frames, ignore_index=True).to_parquet(output / 'author_contrasts.parquet', index=False)
    readback = pd.read_parquet(output / 'author_contrasts.parquet')
    verified_values = 0
    for name, (label, *_) in CONTRASTS.items():
        original = pd.read_csv(source / name, sep='\t', index_col=0)
        restored = readback[readback.contrast.eq(label)].set_index('source_feature_label')
        if list(original.index) != list(restored.index):
            raise ValueError('Source feature order changed on readback')
        np.testing.assert_equal(original.to_numpy(), restored[COLUMNS].to_numpy())
        verified_values += original.size
    report = dict(source_accession='E-MTAB-9923', statistics=stats, sample_libraries=12,
        source_feature_identity='author versioned RefSeq labels, including unmerged numeric suffix rows; not a unique gene axis',
        unit='author DESeq2 contrasts of bulk unique-intron counts',
        prespecified_primary_threshold='paper Fig3 FDR < 0.001',
        source_statistic_entries_exact_readback=verified_values,
        raw_count_matrix_recovered=False, same_cell_pairing=False, intermediate_transition_labels=False,
        apoptosis_labels=False, initial_state_increment_validated=False, complete_scientific_qualification=False,
        source_receipts=receipts, script_sha256=digest(__file__),
        outputs={p.name: digest(p) for p in output.iterdir()})
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.source, args.output), indent=2))
