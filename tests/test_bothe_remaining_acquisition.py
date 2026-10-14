"""Coordinator boundary tests with subprocess mocks; no data is downloaded."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/download_bothe_remaining.py'
    spec = importlib.util.spec_from_file_location('bothe_remaining_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'OUT', tmp_path / 'jobs')
    monkeypatch.setattr(module, 'RAW', tmp_path / 'raw')
    rows = {r['run']: r for r in json.loads(module.MAPPING.read_text(encoding='utf-8'))}
    calls = []

    def mock_run(argv, **kwargs):
        run = argv[argv.index('--run') + 1]
        assert run != 'ERR5000305'
        calls.append(run)
        receipt = Path(argv[argv.index('--receipt') + 1])
        value = {
            'unit_test_mock_only': True,
            'source_mapping_sha256': module.PINNED_MAPPING_SHA,
            'downloader_source_sha256': module.sha(module.DOWNLOADER),
            'files': [{'read_end': end, 'run': run, 'sample': rows[run]['sample'],
                       'status': 'size_and_publisher_MD5_verified'} for end in (1, 2)],
        }
        receipt.write_text(json.dumps(value), encoding='utf-8')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', mock_run)
    return module, calls, mock_run


def test_eleven_unique_runs_exclude_existing_pilot(coordinator):
    module, calls, _ = coordinator
    module.main()
    assert len(calls) == len(set(calls)) == 11
    plan = json.loads((module.OUT / 'jobs.json').read_text(encoding='utf-8'))
    final = json.loads((module.OUT / 'completion.json').read_text(encoding='utf-8'))
    assert plan['parallel_download_workers'] == 2
    assert plan['expected_remaining_bytes'] == 56754947648
    assert {r['run'] for r in plan['runs']} == set(calls)
    assert final['status'] == 'complete'
    assert final['RNA_counts_recovered'] is False
    assert final['model_forward_training_or_PT_load'] is False


def test_existing_run_ownership_rejected_before_any_download(coordinator):
    module, calls, _ = coordinator
    (module.RAW / 'ERR5000296').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='RUN_DIRECTORY_ALREADY_EXISTS'):
        module.main()
    assert not calls and not module.OUT.exists()


def test_existing_coordinator_directory_is_not_overwritten(coordinator):
    module, calls, _ = coordinator
    module.OUT.mkdir()
    marker = module.OUT / 'keep.txt'
    marker.write_text('preserve', encoding='utf-8')
    with pytest.raises(FileExistsError):
        module.main()
    assert not calls and marker.read_text(encoding='utf-8') == 'preserve'


def test_failed_run_not_restarted_and_other_runs_continue(coordinator, monkeypatch):
    module, calls, success = coordinator

    def fail_one(argv, **kwargs):
        if argv[argv.index('--run') + 1] == 'ERR5000296':
            calls.append('ERR5000296')
            return SimpleNamespace(returncode=1)
        return success(argv, **kwargs)

    monkeypatch.setattr(module.subprocess, 'run', fail_one)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    assert len(calls) == 11 and calls.count('ERR5000296') == 1
    final = json.loads((module.OUT / 'completion.json').read_text(encoding='utf-8'))
    assert final['status'] == 'incomplete_failed_runs_preserved'
    assert final['results']['ERR5000296']['status'] == 'failed_preserved_partial_no_restart'


def test_claimed_success_with_wrong_sample_is_rejected(coordinator, monkeypatch):
    module, calls, success = coordinator

    def corrupt_one(argv, **kwargs):
        value = success(argv, **kwargs)
        if argv[argv.index('--run') + 1] == 'ERR5000296':
            receipt = Path(argv[argv.index('--receipt') + 1])
            body = json.loads(receipt.read_text(encoding='utf-8'))
            body['files'][0]['sample'] = 'caller_wrong_sample'
            receipt.write_text(json.dumps(body), encoding='utf-8')
        return value

    monkeypatch.setattr(module.subprocess, 'run', corrupt_one)
    with pytest.raises(SystemExit):
        module.main()
    final = json.loads((module.OUT / 'completion.json').read_text(encoding='utf-8'))
    assert final['results']['ERR5000296']['status'] == 'failed_preserved_partial_no_restart'
    assert final['results']['ERR5000296']['error_type'] == 'ValueError'


def test_changed_mapping_stops_before_download(coordinator, monkeypatch, tmp_path):
    module, calls, _ = coordinator
    changed = tmp_path / 'mapping.json'
    changed.write_text('[]', encoding='utf-8')
    monkeypatch.setattr(module, 'MAPPING', changed)
    with pytest.raises(ValueError, match='FROZEN_SAMPLE_MAPPING_CHANGED'):
        module.main()
    assert not calls and not module.OUT.exists()
