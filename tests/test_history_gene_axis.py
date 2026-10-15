"""Synthetic invariants; actual GSE96779 readback is recorded separately."""
import gzip
import numpy as np
import pandas as pd
import pytest
from lung.data.history_gene_axis import align_history_fpkm, read_grch37_genes


def inputs():
    genes = pd.DataFrame([
        ('ENSG00000000001', 'OLD_A', '1', 100, 300),
        ('ENSG00000000002', 'B', '2', 100, 300),
        ('ENSG00000000003', 'C', '3', 100, 300),
    ], columns=['ensembl_id','gene_name','chromosome','start_1based','end_1based'])
    hgnc = pd.DataFrame([
        ('A', 'OLD_A', 'ENSG00000000001'),
        ('B', '', 'ENSG00000000002'),
        ('C', '', 'ENSG00000000003'),
    ], columns=['symbol','prev_symbol','ensembl_gene_id'])
    frame = pd.DataFrame({'source_feature_id':['f1','f2'], 'gene_id':['A','B'],
                          'locus':['chr1:99-300','chr2:99-300'], 'x':[0.,2.5], 'y':[1.25,3.]})
    axis=['ENSG00000000003','ENSG00000000002','ENSG00000000001']
    return frame, genes, hgnc, axis


def run(frame, genes, hgnc, axis):
    return align_history_fpkm(frame, genes, hgnc, axis, samples=['x','y'], source_build='hg19')


def test_exact_order_scale_zero_and_missing():
    frame,genes,hgnc,axis=inputs()
    x,mask,ledger=run(frame,genes,hgnc,axis)
    np.testing.assert_array_equal(x[:,1:], [[2.5,0.],[3.,1.25]])
    assert np.isnan(x[:,0]).all() and not mask[:,0].any()
    assert mask[0,2] and x[0,2]==0
    assert ledger.mapping_status.tolist()==['matched','matched']
    assert ledger.source_feature_id.tolist()==frame.source_feature_id.tolist()


def test_same_symbol_different_locus_may_disambiguate():
    frame,genes,hgnc,axis=inputs()
    genes.loc[1,'gene_name']='OLD_A'; frame.loc[1,'gene_id']='OLD_A'
    _,_,ledger=run(frame,genes,hgnc,axis)
    assert ledger.mapping_status.eq('matched').all()


def test_duplicate_measurements_not_summed_or_chosen():
    frame,genes,hgnc,axis=inputs()
    duplicate=frame.iloc[[0]].copy(); duplicate.source_feature_id='other_feature'
    frame=pd.concat([frame,duplicate],ignore_index=True)
    x,mask,ledger=run(frame,genes,hgnc,axis)
    assert ledger.mapping_status.eq('duplicate_source_target').sum()==2
    assert not mask[:,2].any() and np.isnan(x[:,2]).all()


def test_axis_not_used_to_resolve_biological_ambiguity():
    frame,genes,hgnc,axis=inputs()
    frame.loc[0,'gene_id']='OLD_A'
    other=genes.iloc[[0]].copy(); other.ensembl_id='ENSG99999999999'
    genes=pd.concat([genes,other],ignore_index=True)
    x,mask,ledger=run(frame,genes,hgnc,axis)
    assert ledger.iloc[0].mapping_status=='ambiguous_symbol_locus'
    assert not mask[:,2].any()


@pytest.mark.parametrize('locus',['chr8:99-300','chr1:900-1000','chrUn_gl0001:99-300'])
def test_wrong_or_unsupported_locus_does_not_map_by_name_alone(locus):
    frame,genes,hgnc,axis=inputs();frame.loc[0,'locus']=locus
    _,mask,ledger=run(frame,genes,hgnc,axis)
    assert ledger.iloc[0].mapping_status!='matched' and not mask[:,2].any()


def test_coordinate_boundary_dependence_remains_unknown():
    frame,genes,hgnc,axis=inputs();frame.loc[0,'locus']='chr1:300-400'
    _,_,ledger=run(frame,genes,hgnc,axis)
    assert ledger.iloc[0].mapping_status=='coordinate_convention_sensitive'


@pytest.mark.parametrize('value',[-1,np.nan,np.inf])
def test_invalid_fpkm_rejected(value):
    frame,genes,hgnc,axis=inputs();frame.loc[0,'x']=value
    with pytest.raises(ValueError,match='FPKM'):run(frame,genes,hgnc,axis)


@pytest.mark.parametrize('axis',['duplicate','empty','versioned'])
def test_invalid_target_axis_rejected(axis):
    frame,genes,hgnc,ids=inputs()
    ids={'duplicate':[ids[0],ids[0]],'empty':[], 'versioned':[ids[0]+'.2']}[axis]
    with pytest.raises(ValueError,match='axis'):run(frame,genes,hgnc,ids)


def test_wrong_build_rejected():
    frame,genes,hgnc,axis=inputs()
    with pytest.raises(ValueError,match='hg19'):
        align_history_fpkm(frame,genes,hgnc,axis,samples=['x','y'],source_build='hg38')


def test_source_feature_identity_must_be_unique():
    frame,genes,hgnc,axis=inputs();frame.source_feature_id='same'
    with pytest.raises(ValueError,match='feature'):run(frame,genes,hgnc,axis)


def test_gtf_build_is_read_not_guessed_from_filename(tmp_path):
    path=tmp_path/'GRCh37.gtf.gz'
    with gzip.open(path,'wt') as f:
        f.write('#!genome-build GRCh38.p14\n')
        f.write('1\tref\tgene\t100\t300\t.\t+\t.\tgene_id "ENSG00000000001"; gene_name "A";\n')
    with pytest.raises(ValueError,match='GRCh37'):read_grch37_genes(path)


def test_grch37_gtf_preserves_coordinates(tmp_path):
    path=tmp_path/'ref.gtf.gz'
    with gzip.open(path,'wt') as f:
        f.write('#!genome-build GRCh37.p13\n')
        f.write('1\tref\tgene\t100\t300\t.\t+\t.\tgene_id "ENSG00000000001"; gene_name "A";\n')
    genes=read_grch37_genes(path)
    assert genes.iloc[0].start_1based==100 and genes.iloc[0].end_1based==300


def test_empty_feature_table_is_a_clear_error():
    frame, genes, hgnc, axis = inputs()
    with pytest.raises(ValueError, match='nonempty'):
        run(frame.iloc[:0], genes, hgnc, axis)


def test_empty_source_feature_identity_is_rejected():
    frame, genes, hgnc, axis = inputs()
    frame.loc[0, 'source_feature_id'] = ' '
    with pytest.raises(ValueError, match='feature'):
        run(frame, genes, hgnc, axis)


def test_build_substring_is_not_assembly_evidence(tmp_path):
    path = tmp_path / 'ref.gtf.gz'
    with gzip.open(path, 'wt') as f:
        f.write('#!genome-build notGRCh37\n')
        f.write('1\tref\tgene\t100\t300\t.\t+\t.\tgene_id "ENSG00000000001"; gene_name "A";\n')
    with pytest.raises(ValueError, match='GRCh37'):
        read_grch37_genes(path)
