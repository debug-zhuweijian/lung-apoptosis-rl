"""Source-bound expression recovery, not cross-assay normalization or pairing.

Every packet preserves native values, sample identities and observed-gene masks.
The source lock and packet checksums make missing local caches recoverable without
silently adopting different source versions. Roles stay unassigned; this module
never changes a frozen experiment or certifies model/scientific effectiveness.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .external_gene_axis import align_gene_axis


def file_sha256(path: Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_source_lock(root: Path, records: list[dict]) -> list[dict]:
    root = Path(root).resolve()
    paths = [r['path'] for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source lock paths')
    verified = []
    for item in records:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source path is outside the declared source root')
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item['bytes'] or file_sha256(path) != item['sha256']:
            raise ValueError(f"Source checksum/size mismatch: {item['path']}")
        verified.append(dict(item, verified=True))
    return verified


def read_geo_samples(path: Path) -> dict[str, dict[str, list[str]]]:
    """Read literal GEO sample fields; a shared protocol is not a sample label."""
    opener = gzip.open if str(path).endswith('.gz') else open
    records: dict[str, dict[str, list[str]]] = {}
    current = None
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            line = line.rstrip('\r\n')
            if line.startswith('^'):
                current = None
                if line.startswith('^SAMPLE = '):
                    accession = line.split(' = ', 1)[1]
                    if accession in records:
                        raise ValueError(f'Duplicate GEO sample accession: {accession}')
                    current = records[accession] = {}
            elif current is not None and line.startswith('!Sample_') and ' = ' in line:
                key, value = line[8:].split(' = ', 1)
                current.setdefault(key, []).append(value)
    if not records:
        raise ValueError(f'No GEO sample metadata: {path}')
    return records


def bind_geo_samples(frame: pd.DataFrame, records: dict, bindings: list[dict],
                     *, identifier: str) -> pd.DataFrame:
    """Validate declared title/column aliases; do not infer treatment from prose.

Bindings state the explicit alias rule and protocol interpretation. Gene-level
annotation columns must be removed before this function is called. A subset of
GEO samples is allowed, but its matrix sample columns must match exactly.
"""
    ids = [b['sample_id'] for b in bindings]
    cols = [b['source_column'] for b in bindings]
    if not ids or len(set(ids)) != len(ids) or len(set(cols)) != len(cols):
        raise ValueError('Sample identities and column aliases must be unique')
    if frame.columns.duplicated().any() or set(frame.columns) != {identifier, *cols}:
        raise ValueError('Expression columns do not match the declared sample set')
    rows = []
    for binding in bindings:
        record = records[binding['sample_id']]
        if record.get('title') != [binding['source_title']]:
            raise ValueError(f"GEO title mismatch: {binding['sample_id']}")
        lines = [v.split(':', 1)[1].strip() for v in record.get('characteristics_ch1', [])
                 if v.lower().startswith('cell line:')]
        if not lines:
            lines = record.get('source_name_ch1', [])
        if lines != [binding['cell_line']]:
            raise ValueError(f"Cell background mismatch: {binding['sample_id']}")
        rows.append(dict(binding, role='unassigned', paired_to_function_replicate=False,
                         geo_fields_json=json.dumps(record, ensure_ascii=False, sort_keys=True)))
    return pd.DataFrame(rows)


def write_expression_packet(output: Path, frame: pd.DataFrame, gene_axis: pd.DataFrame,
                            samples: pd.DataFrame, *, identifier: str, scale: str,
                            metadata: dict[str, Any]) -> dict:
    """Write, reload and compare every native/mapped value; no imputation/rescaling."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new recovery namespace: {output}')
    if (not len(samples) or not samples.sample_id.is_unique
            or not samples.source_column.is_unique or samples.sample_id.isna().any()
            or samples.sample_id.astype(str).str.strip().eq('').any()):
        raise ValueError('Sample identities must be unique and nonmissing')
    axis = gene_axis.sort_values('common_gene_index').reset_index(drop=True)
    cols = samples.source_column.tolist()
    aligned, mask, ledger = align_gene_axis(frame, axis, identifier=identifier, samples=cols, scale=scale)
    native = frame[cols].to_numpy(dtype=np.float64).T
    output.mkdir(parents=True)
    samples.to_csv(output / 'samples.csv', index=False)
    axis.to_csv(output / 'genes.csv', index=False)
    ledger.to_csv(output / 'gene_mapping.csv.gz', index=False)
    sample_ids = samples.sample_id.to_numpy(dtype=str)
    np.savez_compressed(output / 'source_expression.npz', expression=native,
                        source_identifiers=frame[identifier].to_numpy(dtype=str), sample_ids=sample_ids)
    np.savez_compressed(output / 'expression.npz', expression=aligned, observed_mask=mask,
                        gene_ids=axis.ensembl_id.to_numpy(dtype=str), sample_ids=sample_ids)
    matched = ledger.mapping_status.eq('matched')
    columns = ledger.loc[matched, 'common_gene_index'].to_numpy(dtype=int)
    with np.load(output / 'source_expression.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['expression'], native)
        np.testing.assert_array_equal(saved['sample_ids'], sample_ids)
    with np.load(output / 'expression.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['expression'], aligned)
        np.testing.assert_array_equal(saved['observed_mask'], mask)
        np.testing.assert_array_equal(saved['expression'][:, columns], native[:, matched.to_numpy()])
        if not np.isnan(saved['expression'][~saved['observed_mask']]).all():
            raise ValueError('Unobserved genes were converted into measured values')
    result = dict(schema='public-expression-packet-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  measurement_domain=scale, metadata=metadata, source_value_count=int(native.size),
                  sample_count=len(samples), source_gene_count=len(frame), target_gene_count=len(axis),
                  observed_axis_genes=int(mask.any(axis=0).sum()),
                  missing_axis_genes=int((~mask.any(axis=0)).sum()),
                  observed_zero_values=int(((aligned == 0) & mask).sum()),
                  fractional_native_values=int((native != np.floor(native)).sum()),
                  mapping_status_counts=ledger.mapping_status.value_counts().to_dict(),
                  normalization_performed=False, source_values_roundtrip_equal=True,
                  aligned_values_roundtrip_equal=True, model_execution_validated=False,
                  whole_project_qualified=False)
    names = ['samples.csv', 'genes.csv', 'gene_mapping.csv.gz', 'source_expression.npz', 'expression.npz']
    result['files'] = {name: file_sha256(output / name) for name in names}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    load_expression_packet(output, expected_scale=scale, expected_gene_ids=axis.ensembl_id.tolist())
    return result


def load_expression_packet(path: Path, *, expected_scale: str,
                           expected_gene_ids: list[str] | None = None,
                           expected_background: str | None = None) -> dict:
    """Read a checked packet. Callers must name the expected measurement domain."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'public-expression-packet-v1':
        raise ValueError('Unrecognized expression packet schema')
    if manifest['measurement_domain'] != expected_scale:
        raise ValueError('Measurement domain mismatch: no implicit unit conversion')
    required_files = {'samples.csv', 'genes.csv', 'gene_mapping.csv.gz',
                      'source_expression.npz', 'expression.npz'}
    if not required_files.issubset(manifest.get('files', {})):
        raise ValueError('Packet required identity/measurement file hashes are missing')
    for name, digest in manifest['files'].items():
        if not (path / name).resolve().is_relative_to(path.resolve()):
            raise ValueError('Packet file outside declared root')
        if file_sha256(path / name) != digest:
            raise ValueError(f'Packet checksum mismatch: {name}')
    samples = pd.read_csv(path / 'samples.csv', dtype={'sample_id': str})
    genes = pd.read_csv(path / 'genes.csv', dtype={'ensembl_id': str})
    if expected_background is not None:
        if 'cell_line' not in samples or not samples.cell_line.eq(expected_background).all():
            raise ValueError('Cell background mismatch: cross-background transfer is not certified')
    with np.load(path / 'expression.npz', allow_pickle=False) as saved:
        result = {k: saved[k].copy() for k in saved.files}
    np.testing.assert_array_equal(result['sample_ids'], samples.sample_id.to_numpy(dtype=str))
    np.testing.assert_array_equal(result['gene_ids'], genes.ensembl_id.to_numpy(dtype=str))
    if expected_gene_ids is not None and list(result['gene_ids']) != list(expected_gene_ids):
        raise ValueError('Ordered gene identity mismatch')
    x, mask = result['expression'], result['observed_mask']
    if x.shape != (len(samples), len(genes)) or mask.dtype != np.bool_ or mask.shape != x.shape:
        raise ValueError('Invalid expression/mask shape or mask type')
    if not np.array_equal(mask, np.isfinite(x)) or not np.isnan(x[~mask]).all():
        raise ValueError('Observed/missing expression semantics changed')
    return dict(result, samples=samples, genes=genes, manifest=manifest)


def validate_function_observations(frame: pd.DataFrame) -> None:
    """Validate the independently measured, *unpaired* endpoint export contract.

    This audit path does not assign learning roles or create RNA/Y pairs. It
    keeps original percentages or folds alongside their explicit consumer scale.
    """
    required = ['observation_id', 'source_position', 'source_id', 'cell_line',
                'endpoint_domain', 'raw_value', 'raw_unit', 'value', 'value_status',
                'denominator', 'role', 'paired_to_rna_replicate']
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError('Missing functional observation fields or empty input')
    for field in ['observation_id','source_position','source_id','cell_line','denominator']:
        if frame[field].isna().any() or frame[field].astype(str).str.strip().eq('').any():
            raise ValueError(f'Incomplete endpoint identity: {field}')
    if not frame.observation_id.is_unique or frame.duplicated(['source_id','source_position']).any():
        raise ValueError('Functional source positions and observation IDs must be unique')
    if not frame.role.eq('unassigned').all():
        raise ValueError('This recovery path cannot silently assign frozen learning roles')
    if not frame.paired_to_rna_replicate.eq(False).all():
        raise ValueError('This independent-source export cannot certify RNA-replicate pairing')
    if not frame.value_status.isin(['observed','missing_in_source']).all():
        raise ValueError('Unknown functional missingness status')
    missing=frame.value_status.eq('missing_in_source')
    if not frame.loc[missing,['raw_value','value']].isna().all().all():
        raise ValueError('Source missing values must remain missing, not zero')
    domains={'annexin_positive_fraction':('percent',.01),
             'annexin_relative_fold':('fold',1.),
             'sytox_dead_fraction':('fraction',1.),
             'subg1_fraction':('percent',.01)}
    for domain,group in frame.groupby('endpoint_domain',dropna=False):
        if domain not in domains: raise ValueError('Unsupported independent endpoint domain')
        unit,factor=domains[domain]
        if not group.raw_unit.eq(unit).all(): raise ValueError('Endpoint unit/domain mismatch')
        real=group.loc[group.value_status.eq('observed')]
        raw=real.raw_value.to_numpy(dtype=float); value=real.value.to_numpy(dtype=float)
        if not np.isfinite(raw).all() or not np.isfinite(value).all() or (raw<0).any():
            raise ValueError('Invalid real functional values')
        if not np.allclose(value,raw*factor,rtol=1e-12,atol=1e-14):
            raise ValueError('Source unit conversion is inconsistent')
        if domain.endswith('_fraction') and ((value<0).any() or (value>1).any()):
            raise ValueError('Functional fraction outside [0,1]')


def compare_full_structures(left: str, right: str) -> dict:
    """Compare supplied molecular graphs, without authenticating experimental material."""
    from rdkit import Chem
    mols=[Chem.MolFromSmiles(s) if isinstance(s,str) else None for s in [left,right]]
    if any(m is None or m.GetNumAtoms()==0 for m in mols):
        raise ValueError('Cannot parse one or both supplied full structures')
    full=[Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) for m in mols]
    connectivity=[Chem.MolToSmiles(m,canonical=True,isomericSmiles=False) for m in mols]
    return dict(full_graph_equal=full[0]==full[1],connectivity_equal=connectivity[0]==connectivity[1],
                source_full_smiles=full[0],reference_full_smiles=full[1],
                material_identity_verified=False)


def sciplex_nominal_dose_sidecar(frame: pd.DataFrame) -> pd.DataFrame:
    """Interpret the deposited sci-Plex3 dose grid using the primary paper.

    The native table has no unit column. Preserve that distinction from the
    documented interpretation; nominal concentration is not a complete protocol.
    """
    dose = pd.to_numeric(frame['dose'], errors='raise')
    if not dose.isin([0, 10, 100, 1000, 10000]).all():
        raise ValueError('Unexpected sci-Plex3 dose grid; do not guess its unit')
    result = frame.copy()
    result['source_dose_unit_explicit_in_file'] = None
    result['dose_unit_from_primary_paper'] = 'nM'
    result['dose_um'] = dose / 1000.0
    result['dose_conversion'] = 'dose_um = source dose / 1000'
    result['dose_unit_evidence'] = (
        '10.1126/science.aax6234; Results: sci-Plex scales to thousands of '
        'samples and enables HTS; four doses 10 nM, 100 nM, 1 uM, 10 uM; '
        'GSM4150378 hashTable_metadata dose grid')
    result['active_exposure_verified_h'] = None
    return result
