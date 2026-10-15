"""Tests distinguish source arithmetic, nominal structures and scientific eligibility."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from lung.data.mphahlele_alignment import (
    load_archived_events, load_nominal_registry, prepare_alignment, validate_events,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/collect_a549_endpoint_review_sources_20261009.py'
REGISTRY = ROOT / 'configs/mphahlele2023_nominal_structures_v1.json'


def test_archived_source_exact_ids_units_and_arithmetic():
    events = load_archived_events(SCRIPT)
    assert len(events) == 12 and events.source_compound_id.is_unique
    checked = validate_events(events)
    assert checked.loc[checked.source_compound_id == '3c', 'annexin_fraction'].item() == .8458
    assert checked.loc[checked.source_compound_id == 'untreated', 'annexin_fraction'].item() == .11


def test_source_script_is_data_not_executed(tmp_path):
    text = SCRIPT.read_text(encoding='utf-8')
    other = tmp_path / 'source.py'
    other.write_text(text + '\nraise RuntimeError("never execute")\n')
    with pytest.raises(ValueError, match='source script'):
        load_archived_events(other)


@pytest.mark.parametrize('field,value', [
    ('Q2_events', -1), ('Q2_events', 1.5), ('Q2_events', float('nan')),
    ('Q2_events', 10001), ('Q2_events', 1e20), ('Q4_events', 1e20),
    ('all_events', 0), ('all_events', 9999),
    ('dose_um', 0), ('dose_um', float('inf')), ('time_hours', 48),
])
def test_bad_numeric_observations_rejected(field, value):
    events = load_archived_events(SCRIPT)
    events[field] = events[field].astype(float)
    events.loc[0, field] = value
    with pytest.raises(ValueError):
        validate_events(events)


def test_duplicate_missing_and_unknown_compounds_rejected():
    events = load_archived_events(SCRIPT)
    for bad in [events.iloc[:-1], pd.concat([events, events.iloc[:1]]),
                events.assign(source_compound_id=['X'] + events.source_compound_id.tolist()[1:])]:
        with pytest.raises(ValueError):
            validate_events(bad)


def test_registry_captures_isomers_geometry_formula_and_source_conflict():
    reg = load_nominal_registry(REGISTRY)
    assert len(reg['compounds']) == 11
    mols = {r['source_compound_id']: Chem.MolFromSmiles(r['canonical_smiles']) for r in reg['compounds']}
    for record in reg['compounds']:
        mol = mols[record['source_compound_id']]
        assert rdMolDescriptors.CalcMolFormula(mol) == record['neutral_formula']
        assert sum(b.GetStereo() == Chem.BondStereo.STEREOE for b in mol.GetBonds()) == 1
    assert Chem.MolToSmiles(mols['3b']) != Chem.MolToSmiles(mols['3c'])
    assert Chem.MolToSmiles(mols['3d']) != Chem.MolToSmiles(mols['3e'])
    assert Chem.MolToSmiles(mols['3g']) != Chem.MolToSmiles(mols['3h'])
    fluorines = [a for a in mols['3k'].GetAtoms() if a.GetAtomicNum() == 9]
    assert len(fluorines) == 4
    assert rdMolDescriptors.CalcMolFormula(mols['3k']) == 'C17H11F4NO4'
    assert reg['compounds'][-1]['source_identity_conflict'] is True


def test_altered_registry_is_not_silently_accepted(tmp_path):
    reg = json.loads(REGISTRY.read_text())
    reg['compounds'][0]['canonical_smiles'] = 'CCO'
    path = tmp_path / 'altered.json'
    path.write_text(json.dumps(reg))
    with pytest.raises(ValueError, match='registry'):
        load_nominal_registry(path)


def test_real_archived_bundle_and_fingerprint_readback(tmp_path):
    output = tmp_path / 'aligned'
    receipt = prepare_alignment(SCRIPT, REGISTRY, output)
    assert receipt['rows'] == 12 and receipt['nominal_compounds'] == 11
    assert receipt['whole_project_qualified'] is False
    assert receipt['training_ready'] is False
    assert receipt['original_supplement_bytes_verified'] is False
    assert receipt['original_events_reverified_this_run'] is False
    events = pd.read_csv(output / 'observations.csv')
    chems = pd.read_csv(output / 'nominal_chemicals.csv')
    fp = np.load(output / 'morgan2048_chiral.npy', allow_pickle=False)
    assert fp.shape == (11, 2048) and fp.dtype == np.uint8
    assert events.rna_observation_id.isna().all()
    assert events.biological_replicate_id.isna().all()
    assert not events.use_for_training.any()
    untreated = events[events.source_compound_id == 'untreated'].iloc[0]
    assert untreated.control_type == 'untreated_as_reported'
    assert pd.isna(untreated.fingerprint_row)
    assert pd.isna(untreated.chemical_or_experiment_group_id)
    assert (events.statistical_unit == 'reported_condition_panel').all()
    assert len(events.experiment_group_id.unique()) == 1
    # Recompute every bit independently of the production catalog function.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
        includeChirality=True, useBondTypes=True, includeRedundantEnvironments=False)
    for row in chems.itertuples():
        expected = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(row.canonical_full_smiles)), expected)
        np.testing.assert_array_equal(fp[row.fingerprint_row], expected)
    prior = (output / 'receipt.json').read_bytes()
    with pytest.raises(FileExistsError):
        prepare_alignment(SCRIPT, REGISTRY, output)
    assert prior == (output / 'receipt.json').read_bytes()


def test_bad_original_supplement_is_rejected_before_output(tmp_path):
    bad = tmp_path / 'supplement.docx'
    bad.write_bytes(b'not the expected public supplement')
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='supplement'):
        prepare_alignment(SCRIPT, REGISTRY, output, supplement=bad)
    assert not output.exists()


def test_cli_runs_actual_path_and_produces_readable_receipt(tmp_path):
    output = tmp_path / 'cli'
    command = [sys.executable, str(ROOT / 'scripts/align_mphahlele_review.py'),
               '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['training_ready'] is False
    assert 'archived_transcription' in receipt['status']


def test_windows_checkout_line_endings_preserve_pinned_source_identity(tmp_path):
    script = tmp_path / 'source.py'
    registry = tmp_path / 'registry.json'
    script.write_bytes(SCRIPT.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    registry.write_bytes(REGISTRY.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    pd.testing.assert_frame_equal(load_archived_events(script), load_archived_events(SCRIPT))
    assert load_nominal_registry(registry) == load_nominal_registry(REGISTRY)
