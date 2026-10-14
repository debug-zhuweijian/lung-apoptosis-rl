"""Rebuild the public Tahoe/TIDYI coordinate bridge and verify every TIDYI count.

This data-only CLI uses the exact public panel saved by the cloud receipt workflow.
It does not change frozen roles, normalize expression, load weights or fit models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from lung.data.source_bridge import (
    axis_identity, read_gene_axis, read_tidyi_export,
    reconstruct_public_axis, verify_axis_reference,
)

TAHOE_REVISION = '2dc57900b7981cfcf5e211527169a0b006546a95'
TIDYI_HASHES = {
    'tidyi_A549_counts.csv': 'b33aa1b03d71972f8e8c33864be0f0e2747ffa8b56289d00f27cd8c276cc3360',
    'tidyi_A549_metadata.csv': '6ee300a8ba36642950c94d3be7455beb51255e42d8f311273d00aa9e0868a38b',
}


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_sources(root):
    receipts = json.loads((root/'download_receipts.json').read_text())
    names = [x['local_name'] for x in receipts]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate source download receipt')
    by_name = {x['local_name']: x for x in receipts}
    tree = json.loads((root/'tahoe_tree.json').read_text())
    entry = [x for x in tree if x.get('path') == 'metadata/gene_metadata.parquet']
    if len(entry) != 1 or not entry[0].get('lfs'):
        raise ValueError('Official Tahoe gene file identity missing')
    expected = dict(TIDYI_HASHES, **{'tahoe_gene_metadata.parquet': entry[0]['lfs']['oid']})
    records = []
    for name in [*expected, 'tahoe_tree.json']:
        record = by_name[name]
        if not record['status'].startswith('downloaded'):
            raise ValueError(f'Source download not successful: {name}')
        path = root/name
        actual = sha256(path)
        if actual != record['actual_sha256'] or path.stat().st_size != record['actual_bytes']:
            raise ValueError(f'Download receipt mismatch: {name}')
        if name in expected and actual != expected[name]:
            raise ValueError(f'Pinned source changed: {name}')
        if name.startswith('tahoe_') and TAHOE_REVISION not in record['url']:
            raise ValueError('Wrong Tahoe source revision')
        records.append({'name': name, 'bytes': path.stat().st_size, 'sha256': actual,
                        'source_url': record['url'], 'retrieved_utc': record['retrieved_utc']})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--reference-gene-axis', type=Path,
                        help='Actual frozen genes.parquet/shared_gene_vocabulary.csv; never inferred from its size')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; no frozen assets are overwritten')
    sources = verify_sources(args.sources)
    gene_meta = pd.read_parquet(args.sources/'tahoe_gene_metadata.parquet')
    tidyi = read_tidyi_export(args.sources/'tidyi_A549_counts.csv', args.sources/'tidyi_A549_metadata.csv')
    common = reconstruct_public_axis(gene_meta, tidyi['gene_ids'])
    if (len(gene_meta), len(tidyi['gene_ids']), len(common)) != (62710, 32858, 32461):
        raise ValueError('Pinned public source dimensions changed')
    comparison = {'reference_axis_match_verified': False, 'model_compatibility_verified': False}
    if args.reference_gene_axis:
        comparison = verify_axis_reference(common, read_gene_axis(args.reference_gene_axis))
        sources.append({'name': str(args.reference_gene_axis), 'bytes': args.reference_gene_axis.stat().st_size,
                        'sha256': sha256(args.reference_gene_axis), 'source_url': None})
    meta = tidyi['metadata'].copy()
    meta['in_strict_A549_subset'] = tidyi['strict_a549']
    meta['background_exclusion_reason'] = np.where(tidyi['strict_a549'], '', 'source_Cell_line_not_explicit_A549')
    meta['data_role'] = 'unassigned_public_plate8_readback'
    meta['endpoint_scope'] = 'original_sorting_label_not_natural_apoptosis_fraction'
    selected = meta.loc[tidyi['strict_a549']].copy()
    full = tidyi['expression']
    projected = full[tidyi['strict_a549']][:, common.tidyi_column.to_numpy(dtype=np.int64)]
    # Direct one-to-one identity oracle, independent of row iteration in the mapper.
    original_column = {gene: i for i, gene in enumerate(tidyi['gene_ids'])}
    oracle_columns = np.array([original_column[g] for g in common.ensembl_id])
    np.testing.assert_array_equal(oracle_columns, common.tidyi_column)
    np.testing.assert_array_equal(projected, full[tidyi['strict_a549']][:, oracle_columns])
    args.output.mkdir(parents=True, exist_ok=False)
    common.to_parquet(args.output/'genes.parquet', index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(args.output/'genes.parquet'), common)
    meta.to_csv(args.output/'tidyi_source_metadata.csv', index=False)
    meta.loc[~tidyi['strict_a549']].to_csv(args.output/'tidyi_excluded_backgrounds.csv', index=False)
    selected.to_csv(args.output/'tidyi_a549_samples.csv', index=False)
    tidyi['qc_discrepancies'].to_csv(args.output/'tidyi_qc_discrepancies.csv', index=False)
    np.savez_compressed(args.output/'tidyi_a549_common_axis.npz', expression=projected,
                        observed_mask=np.ones(projected.shape, dtype=bool),
                        ensembl_ids=common.ensembl_id.to_numpy(dtype=str),
                        sample_ids=selected.cell_id.to_numpy(dtype=str),
                        state_labels_original=selected.cell_state.to_numpy(dtype=str),
                        value_scale='single_cell_raw_counts')
    with np.load(args.output/'tidyi_a549_common_axis.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['expression'], projected)
        np.testing.assert_array_equal(saved['ensembl_ids'], common.ensembl_id)
        np.testing.assert_array_equal(saved['sample_ids'], selected.cell_id)
        np.testing.assert_array_equal(saved['state_labels_original'], selected.cell_state)
        if not saved['observed_mask'].all() or saved['value_scale'].item() != 'single_cell_raw_counts':
            raise ValueError('TIDYI measured-zero or scale semantics changed')
    selected['well_column'] = selected.Well.str.extract(r'(\d+)$')[0].astype(int)
    cross = selected.groupby(['Plate', 'well_column', 'cell_state'], dropna=False).size().rename('cells').reset_index()
    cross.to_csv(args.output/'state_plate_position_counts.csv', index=False)
    # These are exact descriptive counts, NOT classification performance or causal state evidence.
    dominant = cross.groupby(['Plate', 'well_column']).cells.max().sum()
    validation = {
        'common_axis_identity': axis_identity(common), **comparison,
        'axis_reconstruction_verified_from_pinned_public_inputs': True,
        'unavailable_author_collection_gene_order_verified': False,
        'tidyi_source_cells': len(meta), 'tidyi_source_genes': full.shape[1],
        'source_values_read': int(full.size), 'source_observed_zeros': int(np.count_nonzero(full == 0)),
        'source_max_count': int(full.max()), 'source_integer_count_check': True,
        'strict_a549_cells': len(selected), 'excluded_unknown_or_other_background': int((~tidyi['strict_a549']).sum()),
        'common_axis_values_readback': int(projected.size), 'common_axis_observed_zeros': int(np.count_nonzero(projected == 0)),
        'common_axis_observed_genes': len(common), 'common_axis_missing_genes': 0,
        'qc_discrepancy_cells': len(tidyi['qc_discrepancies']),
        'source_states_original': selected.cell_state.value_counts().to_dict(),
        'source_plates': selected.Plate.astype(str).unique().tolist(),
        'well_column_modal_label_fraction_descriptive_only': float(dominant/len(selected)),
        'roundtrip_exact': True, 'normalization_performed': False, 'training_performed': False,
        'frozen_roles_changed': False, 'whole_project_qualified': False,
    }
    # Recheck immutable input bytes after reading, not just before.
    if verify_sources(args.sources) != sources[:4]:
        raise ValueError('Source inputs changed during execution')
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'tahoe_revision': TAHOE_REVISION, 'tidyi_accession': 'E-MTAB-16354',
                'axis_scope': 'reconstructed_public_export_intersection_not_authenticated_frozen_asset',
                'validation': validation, 'sources': sources,
                'implementation_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                    [Path(__file__).resolve(), ROOT/'src/lung/data/source_bridge.py']},
                'outputs': {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
