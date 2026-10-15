import numpy as np
import pandas as pd
import pytest
from scripts.recover_abraxane_state_reference import map_refseq,descriptive_interaction,bind_samples


def test_source_suffix_cannot_be_stripped_into_a_known_transcript():
    link=pd.DataFrame([['NM_1.2','','','','','','1']+['']*12])
    gi=pd.DataFrame({'GeneID':['1'],'dbXrefs':['Ensembl:ENSG1']})
    m=map_refseq(['NM_1','NM_1_2'],link,gi)
    assert m.ensembl_id.iloc[0]=='ENSG1' and pd.isna(m.ensembl_id.iloc[1])
    assert m.mapping_status.iloc[1]=='author_suffix_unresolved'


def test_ambiguous_current_transcript_gene_link_is_unresolved():
    link=pd.DataFrame([['NM_1.1','','','','','','1']+['']*12,['NM_1.2','','','','','','2']+['']*12])
    gi=pd.DataFrame({'GeneID':['1','2'],'dbXrefs':['Ensembl:ENSG1','Ensembl:ENSG2']})
    assert map_refseq(['NM_1'],link,gi).ensembl_id.isna().all()


def test_shared_response_has_no_descriptive_interaction():
    x=np.array([[1,2],[1,2],[3,5],[5,8],[11,17]],float)
    r=descriptive_interaction(x)
    assert r['descriptive_state_by_treatment_interaction_mean_square']<1e-25
    assert not r['RNA_increment_beyond_state_ID_identified']
    assert not r['statistical_significance_test']


def test_missing_or_negative_source_library_is_not_accepted():
    for x in [np.ones((4,3)),np.full((5,3),np.nan),-np.ones((5,3))]:
        with pytest.raises(ValueError):descriptive_interaction(x)


def test_unmatched_geo_samples_are_not_assigned_by_position():
    with pytest.raises(ValueError,match='coverage'):bind_samples([],[])


def test_functional_summaries_keep_material_and_different_endpoint(tmp_path):
    from scripts.recover_abraxane_state_reference import functional_table
    names=['Abraxane(nM)','PTX(nM)','DTX(nM)','DOX(nM)','RA-V(μg/mL)','CDDP(μM)','5-FU(μM)']
    rows=''.join('<tr>'+''.join('<td>'+v+'</td>' for v in [n,'1.0±0.1','2.0±0.2','2.0','0.01'])+'</tr>' for n in names)
    p=tmp_path/'paper.xml';p.write_text('<article><table-wrap id="pone.0131429.t001"><table><tbody>'+rows+'</tbody></table></table-wrap></article>')
    f=functional_table(p)
    assert len(f)==14 and not f.exact_RNA_endpoint_dose_time_pair.any()
    assert f.loc[f.source_drug.eq('Abraxane'),'material'].eq('albumin_bound_paclitaxel').all()
    assert f.loc[f.source_drug.eq('PTX'),'material'].eq('source_reported_drug').all()
    assert f.endpoint.str.contains('not_Annexin').all()
    assert f.loc[f.source_drug.eq('DTX'),'unit_context'].eq('table_nM_methods_ng_per_mL').all()
