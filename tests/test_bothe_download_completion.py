"""Adversarial receipt checks; tiny local bytes are unit fixtures, not RNA."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def verifier(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'scripts/verify_bothe_downloads.py'
    spec = importlib.util.spec_from_file_location('bothe_completion_test_module', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    row = {'run': 'audit_mock', 'sample': 'audit_mock_sample', 'history': 'audit_mock_history',
           'R1_bytes': 3, 'R2_bytes': 4, 'fastq_ftp': 'audit.invalid/1;audit.invalid/2'}
    records = []
    for end, content in ((1, b'abc'), (2, b'defg')):
        path = module.DATA / 'raw/audit_mock' / f'audit_mock_{end}.fastq.gz'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = module.hashes(path)
        records.append({'run': row['run'], 'sample': row['sample'], 'history': row['history'],
                        'read_end': end, 'url': 'https://' + row['fastq_ftp'].split(';')[end-1],
                        'file': path.name, 'path': path.relative_to(tmp_path).as_posix(),
                        'expected_bytes': len(content), 'actual_bytes': len(content),
                        'expected_md5': digest['md5'], **digest, 'status': 'size_and_publisher_MD5_verified'})
    row['md5'] = ';'.join(e['md5'] for e in records)
    receipt = {'source_mapping_sha256': module.PINNED_MAPPING,
               'downloader_source_sha256': module.PINNED_DOWNLOADER, 'files': records}
    return module, row, receipt


def test_actual_bytes_and_hashes_checked_no_counts_claim(verifier):
    module, row, receipt = verifier
    result = module.verify_pair(row, receipt)
    assert result['actual_file_bytes_MD5_SHA256_verified'] is True
    assert result['RNA_counts_recovered'] is False
    assert result['FASTQ_content_or_pair_sync_verified'] is False


@pytest.mark.parametrize('field,value', [('actual_bytes', 0), ('expected_bytes', 999),
                                         ('md5', '0'*32), ('expected_md5', '0'*32)])
def test_size_and_md5_claims_cannot_override_fixed_declaration(verifier, field, value):
    module, row, receipt = verifier
    receipt = deepcopy(receipt)
    receipt['files'][0][field] = value
    with pytest.raises(ValueError, match='RECEIPT_FIELD_MISMATCH'):
        module.verify_pair(row, receipt, verify_files=False)


def test_changed_actual_file_is_rejected_even_valid_receipt(verifier):
    module, row, receipt = verifier
    (module.DATA / 'raw/audit_mock/audit_mock_1.fastq.gz').write_bytes(b'bad')
    with pytest.raises(ValueError, match='ACTUAL_FILE_MD5_OR_SHA256_MISMATCH'):
        module.verify_pair(row, receipt)


def test_metadata_only_does_not_certify_files(verifier):
    module, row, receipt = verifier
    result = module.verify_pair(row, receipt, verify_files=False)
    assert result['actual_file_bytes_MD5_SHA256_verified'] is False
    assert result['verification_level'] == 'receipt_fields_only'


def test_bool_read_end_not_integer_identity(verifier):
    module, row, receipt = verifier
    receipt['files'][0]['read_end'] = True
    with pytest.raises(ValueError, match='EXACT_TWO_READ_ENDS'):
        module.verify_pair(row, receipt)


@pytest.mark.parametrize('payload', [[], {'files': {}}, {'files': [None]}])
def test_bad_payload_shape_is_a_recordable_rejection(verifier, payload):
    module, row, _ = verifier
    with pytest.raises(ValueError, match='RECEIPT_OBJECT_AND_FILE_RECORDS_REQUIRED'):
        module.verify_pair(row, payload)
