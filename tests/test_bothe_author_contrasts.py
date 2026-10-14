import numpy as np
import pandas as pd
import pytest
from scripts.recover_bothe_author_contrasts import COLUMNS, validate_contrast, samples_from_sdrf


def test_undefined_test_is_not_zero_effect_or_significant():
    f = pd.DataFrame([[0, np.nan, np.nan, np.nan, np.nan, np.nan],
                      [5, 1, .2, 5, .00001, .0001]],
                     columns=COLUMNS, index=['NM_000001.1', 'NR_000002.2'])
    result = validate_contrast(f)
    assert result['FDR_lt_0_001'] == 1
    assert result['undefined_adjusted_p'] == 1
    assert np.isnan(f.iloc[0].log2FoldChange)


@pytest.mark.parametrize('problem', ['unversioned', 'duplicate', 'probability', 'schema'])
def test_invalid_author_statistics_rejected(problem):
    f = pd.DataFrame([[5, 1, .2, 5, .00001, .0001]] * 2,
                     columns=COLUMNS, index=['NM_000001.1', 'NM_000002.2'])
    if problem == 'unversioned': f.index = ['NM_000001', 'NM_000002']
    if problem == 'duplicate': f.index = ['NM_000001.1'] * 2
    if problem == 'probability': f.loc['NM_000001.1', 'padj'] = 1.1
    if problem == 'schema': f = f.rename(columns={'baseMean': 'sample1'})
    with pytest.raises(ValueError): validate_contrast(f)


def sdrf():
    rows = []
    for i, first in enumerate(['dexamethasone', 'ethanol']):
        for j, second in enumerate(['dexamethasone', 'ethanol']):
            for k in range(3):
                row = {'Source Name': f'{i}_{j}_{k}', 'Comment[ENA_RUN]': f'ERR{i}{j}{k}',
                       'Factor Value[stimulus]': first + '+washout+' + second,
                       'Characteristics[cell line]': 'A549'}
                rows.extend([row.copy(), row.copy()])
    return pd.DataFrame(rows)


def test_paired_mates_are_not_biological_replicates():
    x = samples_from_sdrf(sdrf())
    assert len(x) == 12 and not x.raw_counts_recovered.any()
    assert not x.intermediate_RNA_measured.any()
    assert x.endpoint_hours.eq(32).all()


def test_author_suffix_identity_preserved_without_summing():
    f = pd.DataFrame([[5, 1, .2, 5, .00001, .0001]] * 2,
                     columns=COLUMNS, index=['NM_000001.1', 'NM_000001.1_2'])
    result = validate_contrast(f)
    assert result['features'] == 2 and result['author_suffix_rows'] == 1
    assert result['rows_with_ambiguous_base_refseq'] == 2
    assert list(f.index) == ['NM_000001.1', 'NM_000001.1_2']


def test_conflicting_library_identity_rejected():
    x = sdrf(); x.loc[0, 'Comment[ENA_RUN]'] = 'different'
    with pytest.raises(ValueError): samples_from_sdrf(x)
