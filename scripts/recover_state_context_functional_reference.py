"""Recover eight published A549 context/dose means, preserving assay and time limits.

The original Fig2B raster is checked by hash. Values are group-mean readings,
not raw repeats, cell survival probabilities, Annexin fractions or fitted labels.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.recover_public_expression import digest
from scripts.recover_a549_state_context import protocol

IMAGE_SHA='c9b27f15d3a421a127a02ba33b700b53f4fc7361ee21f0aa15c9db47dd2a47ca'
IMAGE_URL='https://cdn.ncbi.nlm.nih.gov/pmc/blobs/fe39/11694068/c1f522c0cd8e/can-24-0529_f2.webp'
# Coordinates in the original 1600x1975 WebP, manually reviewed, not enlarged crop.
ZERO_Y=873.5
HIGH_Y=669.5
HIGH_VALUE=150.
LANDMARKS=[(1,'ctr',449,711),(1,'pb',475,768.5),(1,'doxo',501,779),
           (1,'ink',527,788.5),(5,'ctr',571,744),(5,'pb',598,792),
           (5,'doxo',624,811.5),(5,'ink',650,826.5)]


def read_value(top, edge_error=2., axis_error=.5):
    if not np.isfinite([top,edge_error,axis_error]).all() or edge_error<0 or axis_error<0:
        raise ValueError('Invalid pixel uncertainty')
    span=ZERO_Y-HIGH_Y
    if span<=2*axis_error:raise ValueError('Invalid axis uncertainty')
    value=(ZERO_Y-top)/span*HIGH_VALUE
    low=(ZERO_Y-axis_error-top-edge_error)/(span+2*axis_error)*HIGH_VALUE
    high=(ZERO_Y+axis_error-top+edge_error)/(span-2*axis_error)*HIGH_VALUE
    # Relative biomass can exceed 100; never clip it or invert to death fraction.
    return value,low,high


def bind_contexts(samples):
    expected={'ctr','pb','doxo','ink'}
    if set(samples.condition)!=expected or samples.sample_id.duplicated().any():
        raise ValueError('Context identity differs')
    rows=[]
    for context in sorted(expected):
        g=samples[samples.condition.eq(context)]
        if len(g)!=4 or set(g.library_label.astype(str))!={'1','2','2-2','3'}:
            raise ValueError('RNA library coverage differs')
        if not g.background.eq('A549').all() or g.conservative_split_group.nunique()!=3:
            raise ValueError('Background or conservative replicate units differ')
        rows.append(dict(state_id='GSE246690:A549:'+context,context=context,
            RNA_sample_ids=json.dumps(g.sample_id.tolist()),RNA_libraries=4,conservative_RNA_split_groups=3,
            RNA_collection_h=168,RNA_role='pretreatment_context_reference',
            RNA_function_replicate_pairing=False,**protocol(context)))
    return pd.DataFrame(rows)


def recover(cache,state,output):
    cache,state,output=map(Path,[cache,state,output])
    if output.exists():raise FileExistsError('New output directory required')
    if digest(cache/'figure2.webp')!=IMAGE_SHA:raise ValueError('Original figure hash differs')
    m=json.loads((state/'manifest.json').read_text())
    for name,sha in m['outputs'].items():
        if digest(state/name)!=sha:raise ValueError('State package bytes changed')
    contexts=bind_contexts(pd.read_csv(state/'samples.csv'))
    rows=[]
    for dose,context,x,y in LANDMARKS:
        value,lo,hi=read_value(y)
        rows.append(dict(condition_id=f'A549_context_navitoclax:{context}:{dose}uM',
            state_id='GSE246690:A549:'+context,context=context,drug_name='navitoclax',dose_uM=dose,
            source_reagent='Eurodiagnostico HY-10087',historical_structure_verified=False,
            cell_background='A549',value=value,reading_lower=lo,reading_upper=hi,
            value_scale='crystal_violet_relative_to_own_context_untreated_percent',
            endpoint='relative_adherent_biomass_survival_assay_not_Annexin',
            denominator='each pretreatment context own parallel untreated reference',
            observation_unit='digitized_published_group_mean',author_reported_n=3,
            raw_replicate_values_recovered=False,RNA_function_replicate_pairing=False,
            exposure_duration_h=None,caption_duration_h=72,methods_duration_h=72,results_duration_h=48,
            duration_conflict=True,first_drug_continued_during_navitoclax_verified=False,
            source_panel='Fig2B_A549',x_pixel=x,bar_top_y_pixel=y,
            edge_error_pixels=2.,axis_error_pixels=.5,uncertainty='reading_bounds_not_SEM_or_CI'))
    observations=pd.DataFrame(rows)
    comparisons=[]
    for dose in [1,5]:
        g=observations[observations.dose_uM.eq(dose)].set_index('context');ref=g.loc['ctr']
        for context in ['pb','doxo','ink']:
            r=g.loc[context]
            comparisons.append(dict(comparison='context_minus_proliferating',dose_uM=dose,context=context,
                difference_percentage_points=r.value-ref.value,
                reading_lower=r.reading_lower-ref.reading_upper,reading_upper=r.reading_upper-ref.reading_lower,
                shared_physical_denominator=False,biological_significance_test=False))
    output.mkdir(parents=True)
    contexts.to_csv(output/'context_RNA_references.csv',index=False)
    observations.to_csv(output/'functional_group_means.csv',index=False)
    pd.DataFrame(comparisons).to_csv(output/'context_contrasts.csv',index=False)
    report=dict(source_doi='10.1158/0008-5472.CAN-24-0529',actual_functional_group_means=8,
        contexts=4,doses_uM=[1,5],author_reported_n_per_mean=3,raw_biological_replicates_recovered=False,
        original_raster_size=[1600,1975],image_sha256=IMAGE_SHA,image_url=IMAGE_URL,
        calibration=dict(zero_y=ZERO_Y,high_y=HIGH_Y,high_value=HIGH_VALUE),
        nominal_context_links=4,paired_RNA_functional_units=0,post_navitoclax_RNA_available=False,
        duration_conflict=True,independent_state_RNA_increment_validated=False,
        context_category_and_history_are_confounded=True,apoptosis_endpoint_validated=False,
        real_sequence_response_model_validated=False,full_scientific_qualification=False,
        independent_digitization_review=False,self_review='Dot self-review; no independent agent',
        source_hashes={'paper.xml':digest(cache/'paper.xml'),'state_manifest':digest(state/'manifest.json')},
        script_sha256=digest(__file__),outputs={p.name:digest(p) for p in output.iterdir()})
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','state','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.cache,a.state,a.output),indent=2))
