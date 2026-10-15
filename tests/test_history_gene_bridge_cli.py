"""End-to-end source adapter tests; values below are synthetic, not RNA evidence."""
from argparse import Namespace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.recover_history_gene_bridge import recover


def fixture(tmp_path):
    data = {'gene_id': ['A', 'B'], 'locus': ['chr1:99-300', 'chr2:99-300']}
    text = ''
    states = [
        ('Parental', 'A549 cells parental', 'DMSO'),
        ('MSR', 'MSR-A549 cells', '5uM momelotinib and 1uM selumetinib'),
        ('Withdrawn', 'MSR-A549 cells drugs withdrawal', 'DMSO'),
        ('JQ1', 'MSR-A549 cells with JQ1 treatment', '5uM momelotinib, 1uM selumetinib and 200nM JQ1'),
    ]
    for state, title, drugs in states:
        for rep in range(1, 4):
            number = (len(data) - 2) + 1
            data[f'{state}_{rep}_FPKM'] = [0., number + .25]
            text += (f'^SAMPLE = GSM{number}\n!Sample_geo_accession = GSM{number}\n'
                     f'!Sample_title = {title} {rep}\n'
                     '!Sample_organism_ch1 = Homo sapiens\n!Sample_taxid_ch1 = 9606\n'
                     '!Sample_series_id = GSE96779\n!Sample_molecule_ch1 = total RNA\n'
                     f'!Sample_characteristics_ch1 = drugs treated: {drugs}\n'
                     '!Sample_growth_protocol_ch1 = A549 and MSR-A549 cells were cultured in DMEM supplemented with 10% fetal bovine serum.\n'
                     '!Sample_treatment_protocol_ch1 = To establish MSR-A549 cells, A549 cells were cultured in the presence of 5 μM MMB and 1 μM selumetinib or in DMEM supplemented with 10% FBS alternately for 2 months. MSR-A549 cells were maintained with 5 μM momelotinib and 1 μM selumetinib continuously. Then MSR-A549 cells were treated with 200nM JQ1 for 24 hours. MSR-A549 withdrawal cells were established from MSR-A549 cells cultured without MMB and selumetinib treatment for 10 days.\n'
                     '!Sample_data_processing = hg19 UCSC RefGene Feb 2016 Cufflinks 2.2.1\n')
    matrix, soft, gtf, hgnc, axis = [tmp_path / p for p in ['matrix.tsv', 'meta.gz', 'ref.gtf.gz', 'hgnc.tsv', 'axis.npz']]
    pd.DataFrame(data).to_csv(matrix, sep='\t', index=False)
    with gzip.open(soft, 'wt') as h:
        h.write(text)
    with gzip.open(gtf, 'wt') as h:
        h.write('#!genome-build GRCh37.p13\n')
        for gene, name, chrom in [('ENSG00000000001','A','1'),('ENSG00000000002','B','2')]:
            h.write(f'{chrom}\tref\tgene\t100\t300\t.\t+\t.\tgene_id "{gene}"; gene_name "{name}";\n')
    pd.DataFrame({'symbol':['A','B'],'prev_symbol':['',''],
                  'ensembl_gene_id':['ENSG00000000001','ENSG00000000002']}).to_csv(hgnc,sep='\t',index=False)
    np.savez(axis, ensembl_ids=np.array(['ENSG00000000003','ENSG00000000002','ENSG00000000001']))
    return Namespace(matrix=matrix, soft=soft, gtf=gtf, hgnc=hgnc, axis_npz=axis, output=tmp_path/'out')


def test_consumer_preserves_source_ids_scale_and_cotreatment_without_pseudopairs(tmp_path):
    args=fixture(tmp_path)
    before={p:p.read_bytes() for p in [args.matrix,args.soft,args.gtf,args.hgnc,args.axis_npz]}
    report=recover(args)
    assert report['source_native']['measurement_values']==24
    assert report['observed_values_exactly_rechecked']==24
    assert report['observed_zero_values']==12 and report['unobserved_genes']==1
    assert not report['reference_axis_match_verified'] and not report['whole_project_qualified']
    with np.load(args.output/'common_axis_expression.npz') as z:
        assert z['sample_ids'].tolist()==[f'GSM{i}' for i in range(1,13)]
        np.testing.assert_array_equal(z['expression'][:,1],np.arange(1,13)+.25)
        assert not z['observed_mask'][:,0].any()
        assert z['observed_mask'][:,2].all() and (z['expression'][:,2]==0).all()
    links=json.loads((args.output/'condition_links.json').read_text())
    assert len(links['condition_members'])==4
    assert links['condition_members']['JQ1']==['GSM10','GSM11','GSM12']
    assert len(links['interventions']['JQ1'])==3
    assert links['interventions']['JQ1'][-1]['dose_uM']==.2
    assert links['interventions']['Withdrawn'][0]['withdrawal_h']==240
    assert not links['paired_replicates_or_cells']
    assert links['functional_endpoint_values_in_this_package']==0
    manifest=json.loads((args.output/'manifest.json').read_text())
    for name,digest in manifest['outputs'].items():
        assert hashlib.sha256((args.output/name).read_bytes()).hexdigest()==digest
    for path,contents in before.items():
        assert path.read_bytes()==contents


def test_existing_outputs_are_never_overwritten(tmp_path):
    args=fixture(tmp_path);args.output.mkdir(); marker=args.output/'marker';marker.write_text('keep')
    with pytest.raises(FileExistsError):recover(args)
    assert marker.read_text()=='keep'


@pytest.mark.parametrize('old,new',[
    ('!Sample_taxid_ch1 = 9606','!Sample_taxid_ch1 = 10090'),
    ('!Sample_series_id = GSE96779','!Sample_series_id = GSE999999'),
    ('alternately for 2 months','alternately for 2 days'),
    ('DMEM supplemented with 10% fetal bovine serum.','RPMI supplemented with 5% serum.'),
])
def test_background_and_history_cannot_be_inferred_from_title_alone(tmp_path,old,new):
    args=fixture(tmp_path)
    with gzip.open(args.soft,'rt') as h: text=h.read()
    with gzip.open(args.soft,'wt') as h: h.write(text.replace(old,new))
    with pytest.raises(ValueError,match='source|history|background'):recover(args)
    assert not args.output.exists()


def test_missing_condition_member_is_not_silent_partial_completion(tmp_path):
    args=fixture(tmp_path)
    with gzip.open(args.soft,'rt') as h: text=h.read()
    with gzip.open(args.soft,'wt') as h: h.write(text.rsplit('^SAMPLE',1)[0])
    frame=pd.read_csv(args.matrix,sep='\t');frame.drop(columns=['JQ1_3_FPKM']).to_csv(args.matrix,sep='\t',index=False)
    with pytest.raises(ValueError,match='three'):recover(args)
    assert not args.output.exists()
