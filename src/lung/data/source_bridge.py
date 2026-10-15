"""Source-bound gene coordinates and a readback of the public TIDYI export.

Coordinate agreement does not establish shared measurement units, a shared latent
space, or the identity of an unavailable frozen model/data asset.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _identities(values: Sequence, label: str) -> list[str]:
    result = list(values)
    if not result or any(not isinstance(x, str) or not x.strip() or x != x.strip()
                         or '\n' in x or '\r' in x for x in result):
        raise ValueError(f'{label} identity must be nonempty, explicit, unpadded text')
    if len(result) != len(set(result)):
        raise ValueError(f'Duplicate {label} identity')
    return result


def validate_gene_axis(axis: pd.DataFrame) -> pd.DataFrame:
    """Validate and return coordinate-sorted input without mutating its source."""
    if not axis.columns.is_unique:
        raise ValueError('Duplicate common axis column')
    if not {'common_gene_index', 'ensembl_id'} <= set(axis.columns):
        raise ValueError('Common axis needs explicit identity and coordinates')
    _identities(axis.ensembl_id.tolist(), 'common gene')
    if (not pd.api.types.is_integer_dtype(axis.common_gene_index.dtype)
            or pd.api.types.is_bool_dtype(axis.common_gene_index.dtype)):
        raise ValueError('Common gene coordinates must be integers')
    result = axis.sort_values('common_gene_index').reset_index(drop=True)
    if not np.array_equal(result.common_gene_index, np.arange(len(result))):
        raise ValueError('Common gene axis is not contiguous')
    return result


def axis_identity(axis: pd.DataFrame) -> dict:
    """Fingerprint ordered identifiers, independent of Parquet writer/version."""
    ordered = validate_gene_axis(axis)
    payload = ('\n'.join(ordered.ensembl_id) + '\n').encode('utf-8')
    return {'algorithm': 'ordered_ensembl_ids_utf8_lf_v1', 'gene_count': len(ordered),
            'ordered_ensembl_sha256': hashlib.sha256(payload).hexdigest()}


def verify_axis_reference(axis: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Compare an actually supplied reference; this never authenticates a model."""
    left, right = validate_gene_axis(axis), validate_gene_axis(reference)
    if left.ensembl_id.tolist() != right.ensembl_id.tolist():
        raise ValueError('Supplied reference axis identity/order differs')
    return {'reference_axis_match_verified': True, 'reference_axis_identity': axis_identity(right),
            'model_compatibility_verified': False}


def read_gene_axis(path: Path) -> pd.DataFrame:
    """CSV row order is explicit when a legacy vocabulary has no index column."""
    path = Path(path)
    if path.suffix == '.parquet':
        result = pd.read_parquet(path)
    elif path.suffix == '.csv':
        result = pd.read_csv(path, dtype={'ensembl_id': 'string'}, keep_default_na=False)
    else:
        raise ValueError('Gene axis must be a CSV or Parquet table')
    if 'common_gene_index' not in result:
        result.insert(0, 'common_gene_index', np.arange(len(result), dtype=np.int64))
    return validate_gene_axis(result)


def reconstruct_public_axis(tahoe: pd.DataFrame, tidyi_gene_ids: Sequence[str]) -> pd.DataFrame:
    """Reproduce the existing token-sorted intersection, not a fitted gene filter."""
    if not tahoe.columns.is_unique or not {'token_id', 'ensembl_id', 'gene_symbol'} <= set(tahoe):
        raise ValueError('Tahoe vocabulary columns missing or duplicated')
    _identities(tahoe.ensembl_id.tolist(), 'Tahoe gene')
    ids = _identities(tidyi_gene_ids, 'TIDYI gene')
    token = tahoe.token_id
    if (not pd.api.types.is_integer_dtype(token.dtype) or token.isna().any()
            or pd.api.types.is_bool_dtype(token.dtype) or token.duplicated().any() or (token < 0).any()):
        raise ValueError('Tahoe token identity must be a unique nonnegative integer')
    ordered = tahoe.sort_values('token_id').reset_index(drop=True).copy()
    ordered.insert(0, 'tahoe_column', np.arange(len(ordered), dtype=np.int64))
    ordered = ordered.rename(columns={'token_id': 'tahoe_token_id'})
    columns = pd.DataFrame({'ensembl_id': ids, 'tidyi_column': np.arange(len(ids), dtype=np.int64)})
    common = ordered.merge(columns, on='ensembl_id', how='inner', sort=False, validate='one_to_one')
    common.insert(0, 'common_gene_index', np.arange(len(common), dtype=np.int64))
    return validate_gene_axis(common)


def read_tidyi_export(counts: Path, metadata: Path) -> dict:
    """Read every original count and join by cell ID, retaining unknown backgrounds.

    CSV is parsed directly: pandas' automatic duplicate-header renaming would hide
    an invalid gene axis. All genes, including observed zeros, remain measured.
    """
    with Path(counts).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        genes = _identities(header[1:], 'TIDYI gene')
        cells, rows = [], []
        for line, record in enumerate(reader, 2):
            if len(record) != len(header):
                raise ValueError(f'Count row width mismatch at source line {line}')
            cells.append(record[0])
            try:
                row = np.asarray(record[1:], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f'Invalid count at source line {line}') from exc
            if (not np.isfinite(row).all() or (row < 0).any()
                    or (row > np.iinfo(np.int32).max).any() or (row != np.floor(row)).any()):
                raise ValueError(f'Nonfinite, negative or noninteger count at source line {line}')
            rows.append(row.astype(np.int32))
    _identities(cells, 'TIDYI cell')
    with Path(metadata).open(newline='', encoding='utf-8-sig') as handle:
        meta_header = next(csv.reader(handle))
    if len(meta_header) != len(set(meta_header)):
        raise ValueError('Duplicate metadata column')
    meta = pd.read_csv(metadata, header=0, names=meta_header, dtype={meta_header[0]: str}).rename(columns={meta_header[0]: 'cell_id'})
    needed = {'cell_id', 'Cell_line', 'cell_state', 'Plate', 'Well', 'total_counts', 'n_genes_by_counts'}
    if not needed <= set(meta):
        raise ValueError('Required TIDYI source metadata missing')
    _identities(meta.cell_id.tolist(), 'TIDYI metadata cell')
    if set(cells) != set(meta.cell_id):
        raise ValueError('Count and metadata cell identity sets differ')
    lookup = {cell: i for i, cell in enumerate(cells)}
    meta['source_count_row'] = meta.cell_id.map(lookup)
    expression = np.stack(rows)[meta.source_count_row.to_numpy(dtype=int)]
    totals = expression.sum(axis=1, dtype=np.int64)
    detected = np.count_nonzero(expression, axis=1)
    qc = pd.DataFrame({'cell_id': meta.cell_id, 'matrix_total_counts': totals,
                       'metadata_total_counts': pd.to_numeric(meta.total_counts, errors='raise'),
                       'matrix_detected_genes': detected,
                       'metadata_detected_genes': pd.to_numeric(meta.n_genes_by_counts, errors='raise')})
    qc['total_count_difference'] = qc.metadata_total_counts - qc.matrix_total_counts
    qc['detected_gene_difference'] = qc.metadata_detected_genes - qc.matrix_detected_genes
    changed = qc.total_count_difference.ne(0) | qc.detected_gene_difference.ne(0)
    return {'expression': expression, 'gene_ids': genes, 'metadata': meta,
            'strict_a549': meta.Cell_line.eq('A549').to_numpy(),
            'qc_discrepancies': qc.loc[changed].reset_index(drop=True)}
