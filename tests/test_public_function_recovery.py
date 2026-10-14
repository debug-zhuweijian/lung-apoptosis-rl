import pandas as pd
import numpy as np
import pytest
from lung.data import public_source_recovery as m


def observations():
    return pd.DataFrame([dict(observation_id='study:sheet:A1', source_position='sheet:A1',
        source_id='study', cell_line='A549', endpoint_domain='annexin_positive_fraction',
        raw_value=25., raw_unit='percent', value=.25, value_status='observed',
        denominator='acquired singlets under reported flow gates', role='unassigned',
        paired_to_rna_replicate=False)])


def test_fraction_conversion_preserves_original_scale():
    df=observations()
    m.validate_function_observations(df)
    with pytest.raises(ValueError,match='conversion'):
        m.validate_function_observations(df.assign(value=25.))


def test_fold_is_not_fraction_or_bounded_above_by_one():
    df=observations().assign(endpoint_domain='annexin_relative_fold',raw_unit='fold',raw_value=5.6,value=5.6)
    m.validate_function_observations(df)
    with pytest.raises(ValueError,match='unit'):
        m.validate_function_observations(df.assign(endpoint_domain='annexin_positive_fraction'))


def test_source_positions_cannot_be_replicated_by_rna_join():
    df=observations()
    with pytest.raises(ValueError,match='unique'):
        m.validate_function_observations(pd.concat([df,df.assign(observation_id='other')]))


def test_missing_is_not_zero_and_no_paired_rna_claim():
    df=observations().assign(raw_value=np.nan,value=np.nan,value_status='missing_in_source')
    m.validate_function_observations(df)
    with pytest.raises(ValueError,match='missing'):
        m.validate_function_observations(df.assign(value=0))
    with pytest.raises(ValueError,match='pair'):
        m.validate_function_observations(observations().assign(paired_to_rna_replicate=True))
