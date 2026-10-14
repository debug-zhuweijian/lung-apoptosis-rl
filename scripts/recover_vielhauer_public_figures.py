"""Re-digitize pinned public figures into the existing duration challenge reader.

Coordinates are personally reviewed raster line centres, not fitted values or raw
replicates. New output directories keep the earlier local extraction untouched.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_vielhauer_duration_challenge as target

PINS = {'fig3.jpg': '41762f784511db99a069fdacec3a4ca13c24e8e50db5891ce0312f16faf139e6',
        'fig5.jpg': '61d3c614ebfb2414b7336596aa51e693c0187c2297cf6f7a111837a644b4cd25'}
# Each tuple: total top, late/early boundary, early/necrosis boundary.
LANDMARKS = {
 'Fig3C': {'image': 'fig3.jpg', 'axis_zero_y': 1521.5, 'axis_high_y': 1230.5, 'axis_high_percent': 60,
  'x': [215,260,305,350,394,438,483,528,573,618,663],
  'boundaries': [(1505,1510,1519),(1496,1501,1521),(1475,1483,1516),(1338,1388,1489),
                 (1411,1433,1506),(1481,1492,1517),(1498,1504,1520),(1489,1495,1520),
                 (1344,1392,1491),(1460,1470,1512),(1481,1495,1518)]},
 'Fig5C_left': {'image': 'fig5.jpg', 'axis_zero_y': 760, 'axis_high_y': 615, 'axis_high_percent': 40,
  'x': [109,131,153,175,197,219,241,263,285,307,329],
  'boundaries': [(749,755,759),(750,755,759),(750,755,759),(750,755,759),
                 (698,710,756),(722,730,758),(751,755,759),(751,755,759),
                 (750,755,759),(747,751,759),(723,731,758)]}}


def interval(upper, lower, zero, high, percent, edge_error=2, axis_error=.5):
    """Conservative geometric bounds, NOT a statistical confidence interval."""
    if not high < zero or not high <= upper <= lower <= zero:
        raise ValueError('Invalid raster order or broken-axis range')
    span = zero-high
    if span <= 2*axis_error or edge_error < 0 or axis_error < 0:
        raise ValueError('Invalid calibration uncertainty')
    delta = lower-upper
    return (delta/span*percent,
            max(0, delta-2*edge_error)/(span+2*axis_error)*percent,
            min(100, (delta+2*edge_error)/(span-2*axis_error)*percent))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def recover(cache, output):
    cache, output=Path(cache),Path(output)
    if output.exists(): raise ValueError('Use a new output directory')
    for name,pin in PINS.items():
        if target.sha(cache/name)!=pin: raise ValueError('Source image hash changed')
    if not (cache/'paper.txt').is_file(): raise ValueError('Reviewed paper text required')
    source=output/'source';source.mkdir(parents=True)
    for name in PINS:shutil.copyfile(cache/name,source/name)
    shutil.copyfile(cache/'paper.txt',source/'Vielhauer2013_PMC_fulltext.md')
    (source/'pixel_landmarks.json').write_text(json.dumps({'figures':LANDMARKS,
      'edge_error_pixels':2,'axis_error_pixels':.5,'review':'Dot self-review; no independent reviewer',
      'coordinate_space':'original downloaded raster; not historical enlarged crop',
      'source_urls':{'fig3.jpg':'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/dbbf/3813569/1f19bd903ee5/cbt-14-527-g3.jpg',
                     'fig5.jpg':'https://cdn.ncbi.nlm.nih.gov/pmc/blobs/dbbf/3813569/35fa8ed62aab/cbt-14-527-g5.jpg'}},indent=2))
    conditions=[];components=[];totals=[];comparisons=[];ids={}
    common=dict(protocol_id=target.PROTOCOL,denominator=target.DENOMINATOR,sampling_gate=target.GATE)
    specs=[('vehicle',0)]+[(d,q) for d in ['bocnho','cbi_indole2'] for q in [.05,.5,10,40,1000]]
    for fig,land in LANDMARKS.items():
        arm,duration,wash,recovery=target.ARMS[fig]
        for i,((drug,dose),(top,middle,bottom)) in enumerate(zip(specs,land['boundaries'])):
            cid=f'cloud_raster_v1:{drug}:{dose}:{duration}';ids[(drug,dose,duration)]=cid
            n,note='3','three_independent_experiments'
            if dose==1000:
                n,note=(('2','general_Annexin_1uM_each_drug_one_outlier_excluded') if duration==24
                        else ('','washout_section_n3_but_general_1uM_outlier_n2_scope_ambiguous'))
            conditions.append(dict(common,condition_id=cid,cell_line='A549',state_id='a549_parental_standard_culture',
                drug_id=drug,drug_name=drug,protocol_arm=arm,source_figure=fig,dose_nM=dose,
                exposure_duration_h=duration,washout_at_h='' if wash is None else wash,drug_free_recovery_h=recovery,
                terminal_time_h=24,is_vehicle=str(drug=='vehicle').lower(),initial_RNA_reference='not_available_in_this_source',
                cross_figure_replicate_pairing='not_established',published_n=n,n_status=note,
                control_schedule_verified=False,oxygen_percent='not_reported',seeded_cells=300000,
                medium='F-12K + 10% FBS',harvest='floating_and_adherent_at_endpoint',
                early_floating_cells_recovered_during_washout='not_reported'))
            base=dict(common,condition_id=cid,source_image=land['image'],source_figure=fig,
                value_status='digitized_published_group_mean',uncertainty_kind='pixel_reading_bounds_not_SEM')
            def values(a,b):return interval(a,b,land['axis_zero_y'],land['axis_high_y'],land['axis_high_percent'])
            for name,a,b in [('necrosis',bottom,land['axis_zero_y']),('early_apoptosis',middle,bottom),('late_apoptosis',top,middle)]:
                mid,lo,hi=values(a,b)
                components.append(dict(base,component=name,estimate_percent=mid,reading_lower_percent=lo,
                                       reading_upper_percent=hi,x=land['x'][i],upper_y=a,lower_y=b))
            mid,lo,hi=values(top,bottom)
            totals.append(dict(base,label_type='Annexin_positive_fraction',estimate_percent=mid,
                reading_lower_percent=lo,reading_upper_percent=hi,estimate_fraction=mid/100,
                reading_lower_fraction=lo/100,reading_upper_fraction=hi/100,sem=''))
    for drug,dose in specs[1:]:
        comparisons.append(dict(drug_id=drug,dose_nM=dose,terminal_time_h=24,
            continuous_condition_id=ids[(drug,dose,24)],pulse_condition_id=ids[(drug,dose,6)],duration_p_value=''))
    for filename,rows in zip(target.SOURCE_TABLES,[conditions,components,totals,comparisons]):write_csv(source/filename,rows)
    result=target.build(output/'challenge',source)
    sensitivity=[]
    for edge_error in [1,2,3,4]:
        for i,(drug,dose) in enumerate(specs[1:],start=1):
            values=[]
            for land in LANDMARKS.values():
                top,_,bottom=land['boundaries'][i]
                values.append(interval(top,bottom,land['axis_zero_y'],land['axis_high_y'],
                                       land['axis_high_percent'],edge_error=edge_error))
            c,w=values;lo=c[1]-w[2];hi=c[2]-w[1]
            sensitivity.append(dict(drug_id=drug,dose_nM=dose,edge_error_pixels=edge_error,
                difference_percentage_points=c[0]-w[0],pixel_lower_percentage_points=lo,
                pixel_upper_percentage_points=hi,reading_direction='continuous_higher' if lo>0
                else 'pulse_higher' if hi<0 else 'reading_overlap',biological_significance_test=False))
    (output/'pixel_sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    receipt=dict(source_image_sha256=PINS,source_paper_sha256=target.sha(cache/'paper.txt'),
       script_sha256=target.sha(__file__),contract=result['contract'],
       actual_original_figures_reviewed=True,old_local_source_tables_read=False,
       independent_digitization_review=False,synthetic=False,raw_replicates_recovered=False,
       comparison_to_historical_reading='Different original raster calibration; use geometric ranges, not exact equality')
    (output/'recovery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.cache,a.output),indent=2))
