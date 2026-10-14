import pytest
from rdkit import Chem
from scripts.recover_em2_chemical_identity import validate_nominal_structure,KEY,CONNECTIVITY

GOOD='C=C(C)C(=O)O[C@H]1C/C(C)=C\\[C@@]2(OC)C=C(C)[C@H](O2)[C@H]2OC(=O)C(=C)[C@@H]21'


def test_full_nominal_structure_is_not_a_name_or_formula_match():
    d=validate_nominal_structure(GOOD)
    assert d['rdkit_inchikey']==KEY and d['fragment_count']==1
    assert d['unspecified_stereo_count']==0


@pytest.mark.parametrize('bad',[
    CONNECTIVITY,
    GOOD.replace('[C@@]2','[C@]2'),
    GOOD.replace('(OC)','(OCC)'),
    'C/C/1=C\\[C@@H]2[C@@H]([C@H](CC3=C[C@@H](C1)O[C@@H]3OC)OC(=O)C(=C)C)C(=C)C(=O)O2',
])
def test_unspecified_stereo_epimer_ethyl_and_wrong_isomer_rejected(bad):
    with pytest.raises(ValueError):validate_nominal_structure(bad)
