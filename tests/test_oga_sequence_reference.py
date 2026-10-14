from scripts.recover_oga_sequence_reference import references, contrasts


def test_order_preserves_real_exposure_and_replacement():
    _,p=references()
    a=p['DDP12_then_OGA12'];b=p['OGA12_then_DDP12']
    assert a['stage_medium_replacement'] and b['stage_medium_replacement']
    assert a['stage_extra_wash_count'] is None
    assert [x['drug'] for x in a['events']]==list(reversed([x['drug'] for x in b['events']]))
    assert [(x['start_h'],x['end_h']) for x in a['events']]==[(0,12),(12,24)]
    assert p['DDP_OGA24']['events'][1]['end_h']==24
    assert a['events'][1]['dose_unit']=='ug/ml'
    assert a['events'][1]['unique_structure_verified'] is False


def test_no_fake_replicates_or_total_sd():
    t,_=references()
    assert len(t)==18 and t.protocol_id.nunique()==6
    assert t[t.endpoint=='Annexin_positive_total'].sd.isna().all()
    assert not t.raw_replicates_available.any() and not t.RNA_available.any()
    assert t.source_reported_independent_replications.eq(3).all()
    assert t.cells_counted_per_sample.eq(10000).all()


def test_endpoint_choice_reverses_order_preference():
    t,_=references();r=contrasts(t);d=r['same_exposure_order_difference_percentage_points']
    assert d['Annexin_positive_PI_negative']<0
    assert d['Annexin_positive_PI_positive']>0
    assert abs(d['Annexin_positive_total']-11.3)<1e-10
    assert r['total_Annexin_best_observed_protocol']=='DDP_OGA24'
    assert r['statistical_significance_of_derived_total'] is None
