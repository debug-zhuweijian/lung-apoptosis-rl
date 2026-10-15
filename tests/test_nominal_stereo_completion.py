from lung.data.drug_identity import review_stereo_omission


def test_missing_atom_stereo_can_reference_same_connectivity():
    r=review_stereo_omission('CC(O)F','C[C@H](O)F')
    assert r['eligible'] and not r['experimental_stereochemistry_verified']


def test_opposite_specified_isomer_not_replaced():
    r=review_stereo_omission('C[C@@H](O)F','C[C@H](O)F')
    assert not r['eligible'] and 'specified_stereo' in r['reason']


def test_geometric_stereo_omission_supported_but_conflict_not():
    assert review_stereo_omission('FC=CF','F/C=C/F')['eligible']
    assert not review_stereo_omission('F/C=C\\F','F/C=C/F')['eligible']


def test_salt_or_connectivity_difference_not_repaired():
    assert not review_stereo_omission('CC(O)F.Cl','C[C@H](O)F')['eligible']
    assert not review_stereo_omission('CC(O)Cl','C[C@H](O)F')['eligible']


def test_reference_unknown_stereo_remains_unknown():
    r=review_stereo_omission('CC(O)C(O)F','C[C@H](O)C(O)F')
    assert r['eligible'] and r['reference_unspecified_stereo_count']>0


def test_equal_or_invalid_structures_not_called_new_repair():
    assert not review_stereo_omission('CCO','CCO')['eligible']
    assert not review_stereo_omission(None,'CCO')['eligible']


def test_completion_requires_independent_source_name_resolution_at_caller():
    import pandas as pd
    import pytest
    from scripts.link_inde_tahoe_conditions import resolve
    with pytest.raises(ValueError,match='Independent name evidence'):
        resolve([],{}, {},pd.DataFrame(),complete_omitted_stereo=True)


def test_dexamethasone_literal_alias_conflict_is_not_missing_stereo():
    # Public PubChem CID5743 vs literal `Dexamethasone (DHAP)` CID40467075.
    # Both explicitly specify stereo; same connectivity must not trigger repair.
    from lung.data.drug_identity import compare_structures
    reference='C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4([C@]3([C@H](C[C@@]2([C@]1(C(=O)CO)O)C)O)F)C'
    alias='C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4([C@@]3([C@@H](C[C@@]2([C@@]1(C(=O)CO)O)C)O)F)C'
    assert compare_structures(reference,alias)=='same_connectivity_different_stereochemistry'
    review=review_stereo_omission(alias,reference)
    assert not review['eligible']
    assert review['reason']=='original_has_specified_stereo_requires_review'
