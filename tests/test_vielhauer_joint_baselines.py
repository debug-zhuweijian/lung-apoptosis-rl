from copy import deepcopy
import pytest
from scripts.audit_vielhauer_joint_baselines import freeze_predictions,role,evaluate_role
from test_vielhauer_physical_recovery import fixture


def inputs():
    rows=fixture()[0];fit={r['condition_id']:.1 if r['dose_nM']==.05 else .3 for r in rows if role(r)=='fit'}
    return rows,fit


def test_foreign_or_held_labels_cannot_be_used_for_fitting():
    r,y=inputs();y[r[2]['condition_id']]=.99
    with pytest.raises(ValueError,match='fit-role'):freeze_predictions(r,y)


def test_changing_held_endpoints_never_changes_predictions():
    r,y=inputs();a=freeze_predictions(r,y)
    for row in r:row['measured_endpoint']={'mean_fraction':.999 if role(row)!='fit' else .0}
    assert a==freeze_predictions(r,y)


def test_log_interpolation_reproduces_training_anchors_and_preserves_ties():
    r,y=inputs();p=freeze_predictions(r,y)
    for cid,v in y.items():assert p['drug_protocol_logdose_interpolation'][cid]==pytest.approx(v)
    labels={row['condition_id']:dict(label_fraction=.2,reading_lower_fraction=.19,reading_upper_fraction=.21) for row in r}
    e=evaluate_role(r,labels,p,'final_test')
    assert len(e['fit_mean']['joint_all']['selected_candidate_probabilities'])==8
    assert e['fit_mean']['joint_all']['measured_candidate_regret']==pytest.approx(0)


def test_missing_full_candidate_domain_cannot_be_scored():
    r,y=inputs()
    with pytest.raises(ValueError):freeze_predictions(r[:-1],y)


def test_duplicate_physical_intervention_is_not_an_independent_candidate():
    r,y=inputs();r[-3]['dose_nM']=r[-5]['dose_nM']
    with pytest.raises(ValueError,match='physical candidate'):freeze_predictions(r,y)
