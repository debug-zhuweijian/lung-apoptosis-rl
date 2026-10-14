"""Source-bound Bothe bulk observations. No normalization, model or endpoint.

Reuse the closed approved production/project readback; rehash small metadata and
new joint exports, never the already verified FASTQ/SAM/index assets here.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[3]
BASE = '08-datasets/Bothe2021/E-MTAB-9923'
VERSION = 'registered-rna-counts-v1'
RECOVERY = 'experiments/20261011-data-alignment-round22/bothe_recovery'
PHYSICAL = 'experiments/20261011-data-alignment-round19/full-protocol-inputs-v3'
SOURCE = 'experiments/20261011-data-alignment-round17/bothe_protocol_source'
OUTPUT = BASE + '/observation-inputs-v1'
SCHEMA = 'bothe-source-bound-bulk-observation-inputs-v1'
DOMAIN = 'bulk_total_RNA_unique_intron_fragment_counts_on_joint32461'
SAMPLES = tuple(f'RNA-seq_{history}_rep{rep}' for history in
                ('dex_dex', 'dex_etoh', 'etoh_dex', 'etoh_etoh') for rep in (1, 2, 3))
RUNS = ('ERR5000305', 'ERR5000296', 'ERR5000298', 'ERR5000299', 'ERR5000300', 'ERR5000302',
        'ERR5000301', 'ERR5000304', 'ERR5000297', 'ERR5000295', 'ERR5000303', 'ERR5000306')
FINALIZER_SHA = '76c585931395b3542e0c5fc3cfbe1f481df0700f2d4ac4b9fbfdbbe455490ebe'
PINS = MappingProxyType({
    SOURCE + '/PMC8403771.xml': '836ec5b9703faeb6cdf48489b81bd8e422457ea29149154ecd545039845f473e',
    SOURCE + '/E-MTAB-9923.sdrf.txt': 'cf7d41065c4f7ab2626014ffccfd57b08850360d1d262b605e03fa52e7175df6',
    SOURCE + '/raw_sample_mapping.json': 'eac6bced59a1c41e3ecea8311590edc89e6c288ba591cd9587e9bccc31e86ec0',
    PHYSICAL + '/manifest.json': '60ff10eecdb63e67a9df506e3ff022022474d70856d83933447f177cacb2abd5',
    PHYSICAL + '/records.json': '044625fdbe549dc79a0a61a8d0cb5348c2391056284b86003b48d779cbe9f55a',
    'experiments/20261011-data-alignment-round19/bothe_dexamethasone_source/manifest.json': 'dc22f8d7bd53c4b2b62d9edffeba9a60f2acc9a03b334075a0678f1d1d769f59',
    'experiments/20261011-data-alignment-round19/bothe_dexamethasone_source/morgan2048_chiral.npy': 'd7f284b91a8dbb5f1e0f88bf13472f0a89b9efb8efec512f056e5681ac2aaf26',
    'experiments/20261011-data-alignment-round19/bothe_dexamethasone_source/chemical_identity.json': '5e9a2c2ae3677752c4197b4d5e47c217f9d2288340b1a9d15a545a98a637f5b2',
    'src/lung/data/full_protocol_inputs.py': '6d067664e80cd9f7035387fc904c0c6271d6792b4d934b58aec2fe4854de6a37',
    'src/lung/data/full_protocol_inputs_v3.py': 'ae95a032b44ed021ea4fcd3ffbaaff37fc11f602ce3e8d1bea7b731c135c3a15',
    'src/lung/models/full_protocol.py': 'c6bdce150ed8a0cc53e3e26d3f438597805f918c6a4a4e1fa4f6304e1aa64e27',
    'src/lung/data/bothe_count_production.py': '16a71407526ab51fcc51d293d937d274f3b927cd6366a4de4093b12e9faed986',
    'src/lung/data/bothe_gene_axis_bridge.py': 'd8c0e03d023f5c6b0475a718a1a2bb070ef6ba073e28d6e303fc157015e41bf7',
    RECOVERY + '/finalize_joint_counts.py': FINALIZER_SHA,
    BASE + '/reference/gene-axis-bridge-v1/manifest.json': '99b58eab7ff39770e0fa844f77780d3d91fc01919ceb4cc2153630300e57bc57',
    BASE + '/reference/gene-axis-bridge-v1/joint_axis.tsv': '26fafb12fe60902d56da2ffe018258ad4d9815c8ab2622a34bf5f28328ca34db',
})
FILES = ('counts.json', 'observed_mask.json', 'gene_axis.json', 'sample_contexts.json', 'protocol_contexts.json')
_READBACK_SEAL = object()
LIMITS = {'normalization_fitted': False, 'same_cell_t0': False, 'T4_or_T28_RNA_observed': False,
          'split_roles_assigned': False, 'E_D_P_dependencies_resolved': False,
          'shared_latent_validated': False, 'raw_state_score_C_validated': False,
          'experiment_endpoint_G_validated': False, 'candidate_value_Q_validated': False,
          'model_eligible': False, 'function_eligible': False, 'Annexin_link_validated': False,
          'ML_PT_training': False, 'whole_project_feasibility_completed': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def compile_matrix(genes, ncbi, samples, values, masks):
    """Pure exact-schema transpose; this function grants no real-data evidence."""
    require(tuple(samples) == SAMPLES, 'EXACT_SOURCE_SAMPLE_AXIS_REQUIRED')
    require(len(genes) == len(ncbi) == len(values) == len(masks) > 0
            and len(set(genes)) == len(genes)
            and all(isinstance(g, str) and re.fullmatch(r'ENSG[0-9]{11}', g) for g in genes),
            'EXACT_UNIQUE_GENE_AXIS_REQUIRED')
    require(all(g is None or isinstance(g, str) and re.fullmatch(r'[1-9][0-9]*', g) for g in ncbi)
            and len({g for g in ncbi if g is not None}) == sum(g is not None for g in ncbi),
            'UNIQUE_OPTIONAL_NCBI_IDS_REQUIRED')
    for gene, row, flags in zip(ncbi, values, masks, strict=True):
        require(len(row) == len(flags) == 12, 'EXACT_TWELVE_SAMPLE_ROW_REQUIRED')
        for value, observed in zip(row, flags, strict=True):
            require(type(observed) is bool and observed is (value is not None)
                    and observed is (gene is not None), 'MISSING_COUNT_MAPPING_AND_MASK_DISAGREE')
            require(value is None or type(value) is int and 0 <= value <= 2**63 - 1,
                    'NONNEGATIVE_EXACT_INT64_OR_NONE_REQUIRED')
    return (tuple(tuple(values[g][s] for g in range(len(genes))) for s in range(12)),
            tuple(tuple(masks[g][s] for g in range(len(genes))) for s in range(12)))


def parse_joint_tables(count_path, mask_path, expected_axis):
    genes, ncbi, values, flags = [], [], [], []
    with Path(count_path).open(encoding='utf8', newline='') as counts, Path(mask_path).open(encoding='utf8', newline='') as masks:
        cr, mr = csv.reader(counts, delimiter='\t'), csv.reader(masks, delimiter='\t')
        require(next(cr) == ['ensembl_id', 'ncbi_gene_id', *SAMPLES]
                and next(mr) == ['ensembl_id', *SAMPLES], 'JOINT_EXPORT_HEADERS_CHANGED')
        for index, (row, mask) in enumerate(zip(cr, mr, strict=True)):
            require(len(row) == 14 and len(mask) == 13 and index < len(expected_axis), 'JOINT_EXPORT_SHAPE_CHANGED')
            require((row[0], row[1] or None) == expected_axis[index] and mask[0] == row[0], 'JOINT_ID_ORDER_CHANGED')
            require(all(re.fullmatch(r'0|[1-9][0-9]*', v) for v in row[2:] if v != ''), 'CANONICAL_INTEGER_TEXT_REQUIRED')
            require(all(v in ('0', '1') for v in mask[1:]), 'CANONICAL_BOOLEAN_MASK_REQUIRED')
            genes.append(row[0]); ncbi.append(row[1] or None)
            values.append(tuple(None if v == '' else int(v) for v in row[2:]))
            flags.append(tuple(v == '1' for v in mask[1:]))
    require(len(genes) == len(expected_axis), 'ALL_JOINT_GENES_REQUIRED')
    native, observed = compile_matrix(genes, ncbi, SAMPLES, values, flags)
    return tuple(genes), tuple(ncbi), native, observed


def _check_protocol(record):
    first, _, second = record['source_condition_id'].split('+')
    fields = ('dose', 'start_hours', 'end_hours', 'exposure_hours', 'wash_count',
              'prior_retention', 'media_change', 'recovery_hours')
    expected = [('interval', first, {'start_hours': 0, 'end_hours': 4, 'exposure_hours': 4}),
                ('wash', None, {'start_hours': 4, 'end_hours': 4, 'wash_count': 2}),
                ('media_change', None, {'start_hours': 4, 'end_hours': 4, 'media_change': 1}),
                ('recovery', None, {'start_hours': 4, 'end_hours': 28, 'recovery_hours': 24}),
                ('interval', second, {'start_hours': 28, 'end_hours': 32, 'exposure_hours': 4})]
    require(len(record['events']) == 5 and record['stock_form'] is None
            and record['medium_scope'] == '5percent_FBS_no_extra_hormone_not_absolute_hormone_free'
            and same_json(record['source_vehicle_dose'], {'value': .1, 'unit': 'percent_unspecified_basis',
                                                          'tensor_status': 'masked_registry_has_no_plain_percent'}),
            'SOURCE_MEDIUM_VEHICLE_BASIS_OR_EVENT_LIST_CHANGED')
    for event, (kind, compound, known) in zip(record['events'], expected, strict=True):
        drug = compound == 'dexamethasone'; vehicle = compound == 'ethanol'
        values = {field: known.get(field) for field in fields}
        values['dose'] = 100 if drug else None
        event_type = ('drug_interval' if drug else 'vehicle_interval') if kind == 'interval' else kind
        require(event['event_type'] == event_type and event['vehicle'] is vehicle
                and event['dose_unit'] == ('nM' if drug else 'unknown') and same_json(event['values'], values),
                'SOURCE_EXPOSURE_WASH_RECOVERY_UNITS_OR_MISSING_CHANGED')
        if drug:
            require(type(event['fingerprint']['row']) is int and event['fingerprint']['row'] == 0
                    and event['fingerprint']['identity_status'] == 'nominal_pubchem_reference_historical_batch_not_verified',
                    'NOMINAL_GRAPH_SCOPE_CHANGED')
        else:
            require(event['fingerprint'] is None, 'UNREPORTED_COMPOUND_GRAPH_ADDED')


def build_contexts(mapping, physical, sdrf_rows):
    """Join identities; replicate numbers never pair a treated cell with t0."""
    by_sample = {r['sample']: r for r in mapping}
    by_history = {r['source_condition_id']: r for r in physical}
    expected_histories = {a + '+washout+' + b for a in ('dexamethasone', 'ethanol') for b in ('dexamethasone', 'ethanol')}
    require(len(mapping) == len(by_sample) == 12 and set(by_sample) == set(SAMPLES)
            and len({r['run'] for r in mapping}) == 12 and {r['run'] for r in mapping} == set(RUNS),
            'TWELVE_SOURCE_SAMPLE_RUN_IDENTITIES_REQUIRED')
    require(len(physical) == len(by_history) == 4 and set(by_history) == expected_histories,
            'EXACT_FOUR_SOURCE_HISTORIES_REQUIRED')
    for record in physical:
        _check_protocol(record)
    controls = list(SAMPLES[-3:])
    contexts = []
    for sample in SAMPLES:
        row = by_sample[sample]; history = row['history']; p = by_history[history]
        expected_samples = sorted(r['sample'] for r in mapping if r['history'] == history)
        require(len(expected_samples) == 3 and p['source_samples'] == expected_samples
                and p['source_id'] == 'Bothe2021:E-MTAB-9923' and p['cell_line'] == 'A549'
                and p['role'] == 'unassigned_history_RNA_auxiliary' and type(p['endpoint_hours']) is int
                and p['endpoint_hours'] == 32 and p['time_origin'] == 'first_treatment_start',
                'SOURCE_GROUP_ROLE_OR_T32_CLOCK_CHANGED')
        rows = [r for r in sdrf_rows if r['Source Name'] == sample]
        require(len(rows) == 2 and all(r['Comment[ENA_RUN]'] == row['run']
                and r['Factor Value[stimulus]'] == history and r['Characteristics[cell line]'] == 'A549'
                and r['Characteristics[organism]'] == 'Homo sapiens' and r['Material Type'] == 'cell'
                and r['Comment[LIBRARY_LAYOUT]'] == 'PAIRED' for r in rows), 'SDRF_SAMPLE_RUN_BACKGROUND_CHANGED')
        contexts.append({'sample_id': sample, 'run_id': row['run'], 'ENA_sample': row['ena_sample'],
                         'history_id': history, 'physical_candidate_id': p['candidate_id'],
                         'replicate_index': int(sample.rsplit('rep', 1)[1]),
                         'replicate_type': 'source_triplicate_no_extra_independence_assumption',
                         'source_role': p['role'], 'source_state_id': p['state_id'], 'split_role': None,
                         'cell_line': 'A549', 'organism': rows[0]['Characteristics[organism]'],
                         'disease': rows[0]['Characteristics[disease]'], 'organ_part': rows[0]['Characteristics[organism part]'],
                         'observation_unit': 'bulk_total_RNA_extracted_from_one_million_A549_cells',
                         'count_unit': 'unique_intron_mapped_fragment_count', 'domain': DOMAIN,
                         'observation_hours': 32, 'observation_clock': 'nominal_first_treatment_relative_T32',
                         'reference_sample_ids': list(controls), 'reference_role': 'parallel_T32_double_vehicle_control',
                         'is_reference_control': history == 'ethanol+washout+ethanol',
                         'same_cell_t0': False, 'T4_or_T28_RNA_observed': False,
                         'paired_reference_target_identified': False, 'pre_experiment_history_observed': False,
                         'source_reported_terminal_factor': {name: rows[0][name] for name in
                             ('Factor Value[compound]', 'Factor Value[dose]', 'Unit[concentration unit]')},
                         'source_locator': {'paper': SOURCE + '/PMC8403771.xml', 'figure': 'fig3 A/E',
                                            'sections': ['s10', 's15', 's27'], 'SDRF': SOURCE + '/E-MTAB-9923.sdrf.txt'}})
    return contexts


def closed_count_publication(root=ROOT):
    """Fresh small closure metadata; a dead PID or driver alone grants nothing."""
    root = Path(root).resolve()
    directory = root / BASE / 'processed' / VERSION
    try:
        lifecycle = read_json(directory / 'driver.json')
        if lifecycle.get('status') != 'complete':
            return None
        require(lifecycle.get('runs') == list(RUNS) and lifecycle.get('completed_runs') == list(RUNS),
                'COMPLETE_FIXED_COUNT_LIFECYCLE_REQUIRED')
        provenance = Path(lifecycle['recovery_provenance']).resolve()
        require(provenance.is_relative_to((root / RECOVERY).resolve()) and provenance.name == 'provenance.json'
                and provenance.parent.name.startswith('count-resume-'), 'COUNT_PUBLICATION_PATH_OUTSIDE_SOURCE')
        publication = provenance.parent / 'publication.json'
        marker, production = read_json(publication), read_json(directory / 'manifest.json')
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    require(marker.get('counts_complete') is True and marker.get('runs') == list(RUNS)
            and marker.get('ML_PT_training') is False
            and sha(directory / 'manifest.json') == marker.get('manifest_sha256')
            and production.get('schema') == 'bothe-source-bound-counts-v1'
            and production.get('evidence') == 'actual_registered_RNA_native_execution'
            and production.get('domain') == 'bulk_total_RNA_unique_intron_fragment_counts'
            and production.get('version') == VERSION and production.get('counts_complete') is True
            and production.get('status') == 'counts_complete' and production.get('runs') == list(RUNS)
            and all(production.get(k) is False for k in ('readyRNA', 'model_eligible', 'function_eligible')),
            'CLOSED_COUNT_PUBLICATION_BINDING_CHANGED')
    return {'publication': str(publication), 'publication_sha256': sha(publication),
            'production_manifest_sha256': marker['manifest_sha256'], 'counts_complete': True,
            'ML_PT_training': False}


def _closed_export(root):
    directory = root / BASE / 'processed' / VERSION
    joint = directory / 'joint-axis-v1'
    try:
        report = read_json(joint / 'manifest.json')
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    require(report.get('schema') == 'bothe-actual-joint-count-export-v1'
            and report.get('status') == 'complete_actual_source_bound_count_projection'
            and report.get('version') == VERSION and report.get('domain') == DOMAIN
            and report.get('counts_complete') is True and report.get('sample_ids') == list(SAMPLES)
            and type(report.get('ensembl_genes')) is int and report['ensembl_genes'] == 32461
            and report.get('helper_sha256') == FINALIZER_SHA
            and report.get('fixed_projection_consumer_sha256') == PINS['src/lung/data/bothe_gene_axis_bridge.py']
            and report.get('fixed_count_source_consumer_sha256') == PINS['src/lung/data/bothe_count_production.py']
            and all(report.get(k) is False for k in ('model_eligible', 'function_eligible', 'ML_PT_training',
                'source_decoder_or_normalization_equivalence_validated', 'whole_project_feasibility_completed')),
            'CLOSED_APPROVED_ACTUAL_PROJECT_READBACK_REQUIRED')
    results = []
    for path in (root / RECOVERY).glob('joint-finalization-*/result.json'):
        try:
            if same_json(read_json(path), report):
                results.append(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    if not results:
        return None
    require(len(results) == 1, 'UNIQUE_CLOSED_PROJECT_READBACK_REQUIRED')
    publication = Path(report['actual_count_publication_receipt']).resolve()
    require(publication.is_relative_to((root / RECOVERY).resolve()) and publication.name == 'publication.json'
            and publication.parent.name.startswith('count-resume-'), 'COUNT_PUBLICATION_PATH_OUTSIDE_SOURCE')
    marker, production = read_json(publication), read_json(directory / 'manifest.json')
    require(marker.get('counts_complete') is True and marker.get('runs') == list(RUNS)
            and marker.get('ML_PT_training') is False
            and sha(publication) == report['actual_count_publication_receipt_sha256']
            and sha(directory / 'manifest.json') == marker.get('manifest_sha256') == report['production_manifest_sha256']
            and production.get('schema') == 'bothe-source-bound-counts-v1'
            and production.get('evidence') == 'actual_registered_RNA_native_execution'
            and production.get('domain') == 'bulk_total_RNA_unique_intron_fragment_counts'
            and production.get('version') == VERSION and production.get('counts_complete') is True
            and production.get('status') == 'counts_complete' and production.get('runs') == list(RUNS)
            and all(production.get(k) is False for k in ('readyRNA', 'model_eligible', 'function_eligible')),
            'CLOSED_COUNT_PUBLICATION_BINDING_CHANGED')
    require(set(report['outputs']) == {'counts', 'observed_mask'}, 'EXACT_JOINT_EXPORT_FILES_REQUIRED')
    bindings = {str(joint / 'manifest.json'): sha(joint / 'manifest.json'), str(results[0]): sha(results[0]),
                str(publication): sha(publication), str(directory / 'manifest.json'): sha(directory / 'manifest.json')}
    for key, filename in (('counts', 'joint_gene_counts.tsv'), ('observed_mask', 'observed_mask.tsv')):
        expected = (joint / filename).resolve(); entry = report['outputs'][key]
        require(Path(entry['path']).resolve() == expected and sha(expected) == entry['sha256'], 'JOINT_EXPORT_BYTES_CHANGED')
        bindings[str(expected)] = entry['sha256']
    return report, bindings


def preflight(root=ROOT):
    ready = _closed_export(Path(root).resolve())
    return {'schema': SCHEMA, 'status': 'ready_closed_joint_export' if ready else 'pending_source_bound_joint_count_export',
            'source_data_ready': ready is not None, 'sample_pack_created': False,
            'pending_reason': None if ready else '12_run_production_and_closed_actual_joint_readback_required',
            'normalization_fitted': False, 'ML_PT_training': False, 'whole_project_feasibility_completed': False}


def _snapshot(root):
    closed = _closed_export(root)
    require(closed is not None, 'PENDING_COMPLETE_ACTUAL_JOINT_EXPORT_NO_PLACEHOLDER')
    report, bindings = closed
    for name, expected in PINS.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and sha(path) == expected, 'FIXED_OBSERVATION_SOURCE_CHANGED: ' + name)
        bindings[str(path)] = expected
    physical_manifest = read_json(root / PHYSICAL / 'manifest.json')
    require(physical_manifest['schema'] == 'source_bound_full_physical_protocol_inputs_v3'
            and physical_manifest['feature_schema'] == 'full_physical_protocol_v1'
            and physical_manifest['files_sha256']['records.json'] == PINS[PHYSICAL + '/records.json'],
            'FIXED_FULL_PHYSICAL_SOURCE_IDENTITY_REQUIRED')
    physical = [r for r in read_json(root / PHYSICAL / 'records.json') if r['source_id'] == 'Bothe2021:E-MTAB-9923']
    mapping = read_json(root / SOURCE / 'raw_sample_mapping.json')
    with (root / SOURCE / 'E-MTAB-9923.sdrf.txt').open(encoding='utf8', newline='') as stream:
        contexts = build_contexts(mapping, physical, list(csv.DictReader(stream, delimiter='\t')))
    with (root / BASE / 'reference/gene-axis-bridge-v1/joint_axis.tsv').open(encoding='utf8', newline='') as stream:
        axis = list(csv.DictReader(stream, delimiter='\t'))
    require(len(axis) == 32461 and [r['common_gene_index'] for r in axis] == [str(i) for i in range(32461)],
            'REGISTERED_JOINT_GENE_AXIS_REQUIRED')
    expected_axis = [(r['ensembl_id'], r['ncbi_gene_id'] or None) for r in axis]
    joint = root / BASE / 'processed' / VERSION / 'joint-axis-v1'
    genes, ncbi, counts, observed = parse_joint_tables(joint / 'joint_gene_counts.tsv', joint / 'observed_mask.tsv', expected_axis)
    coverage = {'source_mapped_measurable_genes': sum(g is not None for g in ncbi),
                'actual_observed_gene_sample_cells': sum(sum(row) for row in observed),
                'unobserved_gene_sample_cells': sum(sum(not value for value in row) for row in observed),
                'observed_zero_cells': sum(sum(value == 0 for value in row if value is not None) for row in counts)}
    require(coverage['source_mapped_measurable_genes'] == 17107 and coverage['actual_observed_gene_sample_cells'] == 205284
            and all(type(report[k]) is int and report[k] == value for k, value in coverage.items()), 'ACTUAL_PROJECT_COVERAGE_CHANGED')
    code = 'src/lung/data/bothe_observation_inputs.py'; builder = 'scripts/prepare_bothe_observation_inputs.py'
    require(sha(root / code) == sha(__file__), 'EXECUTING_OBSERVATION_CONSUMER_COPY_CHANGED')
    bindings[str(root / code)] = sha(__file__)
    require(sha(root / builder) == sha(ROOT / builder), 'EXECUTING_OBSERVATION_BUILDER_COPY_CHANGED')
    bindings[str(root / builder)] = sha(ROOT / builder)
    data = {'counts.json': counts, 'observed_mask.json': observed,
            'gene_axis.json': {'ensembl_ids': genes, 'ncbi_gene_ids': ncbi},
            'sample_contexts.json': contexts, 'protocol_contexts.json': physical}
    metadata = {'schema': SCHEMA, 'status': 'complete_actual_source_bound_bulk_observation_data',
                'version': VERSION, 'domain': DOMAIN, 'shape': [12, 32461], 'sample_ids': list(SAMPLES),
                'count_unit': 'unique_intron_mapped_fragment_count', 'missing_representation': 'JSON_null_with_false_mask',
                'data_layout': 'sample_by_registered_joint_gene', 'physical_schema': 'full_physical_protocol_v1',
                'historical_physical_record_pending_fields_preserved': True,
                'reference_role': 'parallel_T32_double_vehicle_control', 'reference_clock_hours': 32,
                'bulk_not_UMI_or_mature_mRNA': True, 'coverage': coverage,
                'qualification': 'actual bulk observation data only; E/D/P and function validation pending',
                'raw_SAM_index_rehashed_here': False, 'closed_source_bound_project_readback_reused': True,
                'source_bindings': bindings, **LIMITS}
    return data, metadata


def prepare(root=ROOT, output=None):
    root = Path(root).resolve(); output = (root / OUTPUT if output is None else Path(output)).resolve()
    require(output.parent == (root / BASE).resolve() and re.fullmatch(r'observation-inputs-v[1-9][0-9]*', output.name),
            'NEW_BOTHE_OBSERVATION_VERSION_WITHIN_SOURCE_REQUIRED')
    require(not output.exists(), 'NEW_OBSERVATION_OUTPUT_REQUIRED_NO_OVERWRITE')
    data, metadata = _snapshot(root)
    output.mkdir()
    for name in FILES:
        write_json(output / name, data[name])
    manifest = {**metadata, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                'files_sha256': {name: sha(output / name) for name in FILES}}
    write_json(output / 'manifest.json', manifest)
    reader = BotheObservationInputs.load(root, output)
    require(same_json(reader._data, data), 'ACTUAL_OBSERVATION_SERIALIZATION_READBACK_CHANGED')
    return manifest


class BotheObservationInputs:
    """Immutable exact native matrices; context access returns independent copies."""
    def __init__(self, data, manifest, *, _seal=None):
        require(_seal is _READBACK_SEAL, 'SOURCE_BOUND_LOADER_REQUIRED_NO_DIRECT_QUALIFICATION')
        self._data = json.loads(json.dumps(data, allow_nan=False))
        self._manifest = json.loads(json.dumps(manifest, allow_nan=False))
        self._ensembl_ids = tuple(data['gene_axis.json']['ensembl_ids'])
        self._counts = tuple(tuple(row) for row in data['counts.json'])
        self._observed_mask = tuple(tuple(row) for row in data['observed_mask.json'])

    @classmethod
    def load(cls, root=ROOT, directory=None):
        root = Path(root).resolve(); directory = (root / OUTPUT if directory is None else Path(directory)).resolve()
        require(directory.parent == (root / BASE).resolve(), 'OBSERVATION_PATH_OUTSIDE_SOURCE')
        manifest = read_json(directory / 'manifest.json')
        expected, metadata = _snapshot(root)
        require(set(manifest) == set(metadata) | {'prepared_at_utc', 'files_sha256'}
                and same_json({k: manifest[k] for k in metadata}, metadata)
                and set(manifest['files_sha256']) == set(FILES), 'OBSERVATION_SOURCE_OR_TRUTH_BOUNDARY_CHANGED')
        data = {}
        for name in FILES:
            require(sha(directory / name) == manifest['files_sha256'][name], 'OBSERVATION_BYTES_CHANGED')
            data[name] = read_json(directory / name)
        require(same_json(data, expected), 'OBSERVATION_VALUES_OR_CONTEXT_DIFFER_FROM_SOURCE')
        return cls(data, manifest, _seal=_READBACK_SEAL)

    @property
    def manifest(self):
        return json.loads(json.dumps(self._manifest))

    @property
    def sample_ids(self):
        return SAMPLES

    @property
    def ensembl_ids(self):
        return self._ensembl_ids

    @property
    def counts(self):
        return self._counts

    @property
    def observed_mask(self):
        return self._observed_mask

    @property
    def sample_contexts(self):
        return json.loads(json.dumps(self._data['sample_contexts.json']))

    @property
    def protocol_contexts(self):
        return json.loads(json.dumps(self._data['protocol_contexts.json']))
