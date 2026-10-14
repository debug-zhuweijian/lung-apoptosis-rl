"""Recover the complete archived Annexin panel and join nominal chemical identities.

This is a source-specific data preparation path, NOT a fitted endpoint model.
No RNA, individual-cell trajectory, biological replicate or training role is invented.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .chemistry import build_chemical_catalog

SOURCE_BLOB = 'f4627a1775c7d311894b81f491244aad31bcdf21'
BASELINE = '4a78560bdeb2e0d98332d5a9520764a120abf612'
REGISTRY_SHA256 = '60ee9be2044289be6354331be42d0cefa45be9ea6aafbbd303cf22e10fb4bba5'
SUPPLEMENT_SHA256 = '6ac58a4129072e6a7956438a9bb8c2d091a85c76472c74bbc2d0db20c0a7e7fb'
SUPPLEMENT_BYTES = 3231707
ACTIVE_IDS = tuple('3' + letter for letter in 'abcdefghijk')
EVENT_IDS = (*ACTIVE_IDS, 'untreated')


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_archived_events(source_script: Path) -> pd.DataFrame:
    """Read a pinned literal with AST; never import/execute the network collector."""
    data = Path(source_script).read_bytes().replace(b'\r\n', b'\n')
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError('Archived source script differs from the pinned baseline blob')
    tree = ast.parse(data.decode('utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'transcribe_mphahlele')
    assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'source' for t in node.targets))
    values = ast.literal_eval(assignment.value)
    frame = pd.DataFrame(values, columns=['source_compound_id', 'dose_um', 'Q2_events', 'Q4_events'])
    frame['all_events'] = 10000
    frame['time_hours'] = 24
    return frame


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {'source_compound_id', 'dose_um', 'Q2_events', 'Q4_events', 'all_events', 'time_hours'}
    if not required.issubset(events.columns):
        raise ValueError('Missing event-table columns')
    if (len(events) != 12 or not events.source_compound_id.is_unique
            or set(events.source_compound_id) != set(EVENT_IDS)):
        raise ValueError('Expected exactly the twelve archived source panels, without duplicate IDs')
    result = events.copy()
    for key in required - {'source_compound_id'}:
        if events[key].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise ValueError(f'Boolean is not a numeric measurement: {key}')
        result[key] = pd.to_numeric(events[key], errors='raise')
        if not np.isfinite(result[key].to_numpy(dtype=float)).all():
            raise ValueError(f'Non-finite observation: {key}')
    for key in ('Q2_events', 'Q4_events', 'all_events'):
        if (result[key] < 0).any() or (result[key] != np.floor(result[key])).any():
            raise ValueError(f'Expected nonnegative integer events: {key}')
        # Bound before casting: oversized floating values can wrap int64.
        if (result[key] > 10000).any():
            raise ValueError(f'Event count exceeds the reported source denominator: {key}')
        result[key] = result[key].astype(np.int64)
    if not result.all_events.eq(10000).all():
        raise ValueError('The archived panel denominator is exactly 10000 acquired events')
    if ((result.Q2_events + result.Q4_events) > result.all_events).any():
        raise ValueError('Annexin event sum exceeds its denominator')
    if not result.time_hours.eq(24).all():
        raise ValueError('This source contract is the reported 24-hour Annexin assay only')
    active = result.source_compound_id.isin(ACTIVE_IDS)
    if (result.loc[active, 'dose_um'] <= 0).any() or not result.loc[~active, 'dose_um'].eq(0).all():
        raise ValueError('Invalid active or untreated nominal dose')
    result['annexin_fraction'] = (result.Q2_events + result.Q4_events) / result.all_events
    result['annexin_percent'] = result.annexin_fraction * 100
    return result.sort_values('source_compound_id', kind='stable').reset_index(drop=True)


def load_nominal_registry(registry: Path) -> dict:
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    data = Path(registry).read_bytes().replace(b'\r\n', b'\n')
    if _sha(data) != REGISTRY_SHA256:
        raise ValueError('Nominal registry changed; a revised source review/version is required')
    result = json.loads(data)
    records = result['compounds']
    if [r['source_compound_id'] for r in records] != list(ACTIVE_IDS):
        raise ValueError('Nominal registry requires all eleven compounds in source order')
    canonical = []
    for row in records:
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol is None or CalcMolFormula(mol) != row['neutral_formula']:
            raise ValueError('Invalid nominal registry structure or neutral formula')
        if sum(b.GetStereo() == Chem.BondStereo.STEREOE for b in mol.GetBonds()) != 1:
            raise ValueError('Nominal registry must retain the reported E alkene')
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    if len(set(canonical)) != 11:
        raise ValueError('Nominal registry collapses distinct source compounds')
    return result


def prepare_alignment(source_script: Path, registry: Path, output: Path,
                      *, supplement: Path | None = None) -> dict:
    """Publish a new immutable audit bundle; never promote it to training-ready."""
    source_script, registry, output = map(Path, (source_script, registry, output))
    if output.exists():
        raise FileExistsError(f'Output must be a new directory: {output}')
    events = validate_events(load_archived_events(source_script))
    specification = load_nominal_registry(registry)
    original_bytes_verified = False
    supplement_receipt = None
    if supplement is not None:
        raw = Path(supplement).read_bytes()
        if len(raw) != SUPPLEMENT_BYTES or _sha(raw) != SUPPLEMENT_SHA256:
            raise ValueError('Original supplement size or SHA256 does not match its source review')
        original_bytes_verified = True
        supplement_receipt = {'sha256': _sha(raw), 'bytes': len(raw)}
    nominal = pd.DataFrame(specification['compounds'])
    expected = nominal.set_index('source_compound_id').source_dose_um
    active = events.source_compound_id.isin(ACTIVE_IDS)
    if not np.allclose(events.loc[active, 'dose_um'], events.loc[active, 'source_compound_id'].map(expected),
                       atol=1e-10, rtol=0):
        raise ValueError('Archived dose disagrees with the primary Table 2 A549 column')
    nominal['drug_id'] = 'mphahlele2023:' + nominal.source_compound_id
    nominal['drug_name'] = 'Mphahlele2023 ' + nominal.source_compound_id
    nominal['component_role'] = 'active'
    # Use the existing real catalog/fingerprint path, not a placeholder vector.
    chemicals, fingerprints, chemistry = build_chemical_catalog(nominal)
    chemicals = chemicals.merge(nominal.drop(columns=['drug_name', 'canonical_smiles']),
                                on='drug_id', how='left', validate='one_to_one')
    identity_columns = ['source_compound_id', 'drug_id', 'chemical_or_experiment_group_id',
                        'fingerprint_row', 'source_identity_conflict']
    observations = events.merge(chemicals[identity_columns], on='source_compound_id',
                                 how='left', validate='one_to_one')
    observations['fingerprint_row'] = observations.fingerprint_row.astype('Int64')
    observations['observation_id'] = 'mphahlele2023:TableS3:' + observations.source_compound_id
    observations['source_id'] = 'Mphahlele2023_Annexin'
    observations['source_doi'] = specification['doi']
    observations['experiment_group_id'] = 'mphahlele2023:TableS3'
    observations['cell_line'] = 'A549'
    observations['background_id'] = 'mphahlele2023:A549_as_reported'
    observations['control_observation_id'] = 'mphahlele2023:TableS3:untreated'
    observations['control_type'] = 'untreated_as_reported'
    observations['vehicle_identity'] = None
    observations['source_dose_value'] = observations.dose_um
    observations['source_dose_unit'] = 'uM'
    observations['dose_basis'] = np.where(observations.source_compound_id.eq('untreated'),
                                          'untreated', '1x individual A549 MTT IC50')
    observations['exposure_duration_h'] = 24
    observations['sampling_elapsed_h'] = 24
    observations['clock_origin'] = 'drug_addition_or_parallel_untreated'
    observations['prior_treatment_history_known'] = False
    observations['washout_schedule_known'] = False
    observations['endpoint_type'] = 'Annexin_Q2_plus_Q4_fraction'
    observations['endpoint_unit'] = 'fraction_of_10000_acquired_events'
    observations['statistical_unit'] = 'reported_condition_panel'
    observations['biological_replicate_id'] = None
    observations['batch_id'] = None
    observations['rna_observation_id'] = None
    observations['gene_axis_id'] = None
    observations['rna_is_paired'] = False
    observations['data_role'] = 'unassigned_source_review'
    observations['use_for_training'] = False
    observations['material_identity_authenticated'] = False
    outputs = {'observations.csv': observations, 'nominal_chemicals.csv': chemicals}
    # Every input/semantic check occurs before creating the publication directory.
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False, encoding='utf-8', lineterminator='\n')
    np.save(output / 'morgan2048_chiral.npy', fingerprints, allow_pickle=False)
    hashes = {name: {'sha256': _sha((output / name).read_bytes()), 'bytes': (output / name).stat().st_size}
              for name in [*outputs, 'morgan2048_chiral.npy']}
    # Readback is mandatory; processing exit success alone is insufficient.
    readback = pd.read_csv(output / 'observations.csv')
    np.testing.assert_allclose(readback.annexin_fraction, observations.annexin_fraction,
                               rtol=0, atol=1e-14)
    np.testing.assert_array_equal(np.load(output / 'morgan2048_chiral.npy', allow_pickle=False), fingerprints)
    receipt = {
        'status': 'archived_transcription_aligned_to_nominal_structures_not_training_ready',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_commit': BASELINE, 'source_script_git_blob': SOURCE_BLOB,
        'source_script_sha256': _sha(source_script.read_bytes()),
        'registry_sha256': _sha(registry.read_bytes()),
        'implementation_sha256': _sha(Path(__file__).read_bytes()),
        'chemistry_implementation_sha256': _sha(Path(sys.modules[build_chemical_catalog.__module__].__file__).read_bytes()),
        'rows': 12, 'nominal_compounds': 11, 'reported_control_panels': 1,
        'source_identity_conflicts': ['3k'],
        'nominal_identity_is_material_authentication': False,
        'original_supplement_bytes_verified': original_bytes_verified,
        'original_supplement': supplement_receipt,
        'original_events_reverified_this_run': False,
        'numeric_readback_passed': True,
        'rna_observations_available': 0, 'biological_replicate_series_available': False,
        'training_ready': False, 'whole_project_qualified': False,
        'chemistry': chemistry, 'outputs': hashes,
        'limitations': ['No matched RNA or RNA bridge is created.',
                        'Acquired flow events are not independent treatment replicates.',
                        'IC50 dose selection is assay-adaptive; this is not common-dose potency.',
                        '3k has conflicting published HRMS identity evidence.',
                        'No frozen roles, reward, existing data or model parameters were changed.'],
    }
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return receipt
