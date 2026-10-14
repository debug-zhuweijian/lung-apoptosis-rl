from types import SimpleNamespace

import pytest

from scripts.download_bothe_inputs import validate_response


def test_exact_resume():
    validate_response(SimpleNamespace(status_code=206, headers={
        'Content-Range': 'bytes 4-9/10', 'Content-Length': '6'}), 4, 10)


@pytest.mark.parametrize('status,range_value,length', [
    (200, '', '10'), (206, 'bytes 0-9/10', '10'),
    (206, 'bytes 4-8/10', '5'), (206, 'bytes 4-9/11', '6')])
def test_inexact_resume_rejected(status, range_value, length):
    with pytest.raises(ValueError):
        validate_response(SimpleNamespace(status_code=status, headers={
            'Content-Range': range_value, 'Content-Length': length}), 4, 10)


def test_full_length_and_encoding():
    validate_response(SimpleNamespace(status_code=200, headers={'Content-Length': '10'}), 0, 10)
    with pytest.raises(ValueError):
        validate_response(SimpleNamespace(status_code=200, headers={'Content-Length': '9'}), 0, 10)
    with pytest.raises(ValueError):
        validate_response(SimpleNamespace(status_code=200, headers={
            'Content-Length': '10', 'Content-Encoding': 'gzip'}), 0, 10)
