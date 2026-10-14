import gzip
import numpy as np
import pytest
from scripts.recover_scifate_time_reference import coordinate_rows,joint_sums


def test_measured_new_subset_and_time_sums():
    t=[(0,0,0,4),(1,1,0,2),(3,1,1,8)];n=[(1,1,0,1),(3,1,1,3)]
    a,b,c,nt,nn=joint_sums(t,n,2,2,np.array([0,1]))
    assert c.tolist()==[[6,1],[8,3]] and (nt,nn)==(3,2)
    assert a[0].tolist()==[4,2] and b[1].tolist()==[0,3]


@pytest.mark.parametrize('new', [[(0,0,0,5)],[(1,1,0,1)],[(2,0,1,1)]])
def test_impossible_new_counts_are_rejected(new):
    with pytest.raises(ValueError):joint_sums([(0,0,0,4)],new,2,2,np.array([0,1]))


@pytest.mark.parametrize('body',['1 1 2\n1 1 1\n','1 2 2\n1 1 1\n','0 1 2\n1 1 1\n','1 1 2\n'])
def test_coordinate_corruption_rejected(tmp_path,body):
    p=tmp_path/'x.gz'
    with gzip.open(p,'wt') as f:f.write('%%MatrixMarket matrix coordinate integer general\n2 2 2\n'+body)
    with pytest.raises(ValueError):list(coordinate_rows(p,2,2))
