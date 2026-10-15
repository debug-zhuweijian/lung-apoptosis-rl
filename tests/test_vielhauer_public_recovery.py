import pytest
from scripts.recover_vielhauer_public_figures import interval, LANDMARKS


def test_component_boundary_cancels_without_necrosis():
    args=(1521.5,1230.5,60)
    early=interval(1388,1489,*args)[0];late=interval(1338,1388,*args)[0]
    total=interval(1338,1489,*args)[0]
    assert early+late==pytest.approx(total)
    assert total==pytest.approx(31.1340206185567)
    assert total<interval(1338,1521.5,*args)[0]


def test_geometric_range_is_not_manufactured_biological_sem():
    mid,lo,hi=interval(750,759,760,615,40)
    assert 0<=lo<mid<hi<100
    assert interval(759,760,760,615,40)[1]==0


@pytest.mark.parametrize('args',[(5,4,10,0,40),(0,11,10,0,40),(-1,5,10,0,40),(0,1,0,10,40)])
def test_bad_or_broken_axis_coordinates_rejected(args):
    with pytest.raises(ValueError):interval(*args)


def test_all_reviewed_coordinates_are_finite_and_ordered():
    for land in LANDMARKS.values():
        assert len(land['x'])==len(land['boundaries'])==11
        for top,middle,bottom in land['boundaries']:
            assert land['axis_high_y']<=top<=middle<=bottom<=land['axis_zero_y']
