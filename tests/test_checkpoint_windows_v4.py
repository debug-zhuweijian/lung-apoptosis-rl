"""CPU checks for bounded Windows sharing retries; never replace before close."""
import ctypes
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lung.engine import checkpoint


def win_error(code):
    error = OSError('injected Windows replacement failure')
    error.winerror = code
    return error


@pytest.mark.parametrize('code', [5, 32, 33])
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows retry policy')
def test_transient_replace_preserves_old_checkpoint_until_commit(tmp_path, code):
    dst = tmp_path / 'last.pt'
    checkpoint.atomic_save(dst, {'generation': 1})
    old = dst.read_bytes()
    replace = checkpoint.os.replace
    attempts = []
    def fail_twice(src, target):
        attempts.append(str(src))
        assert dst.read_bytes() == old
        if len(attempts) < 3:
            raise win_error(code)
        replace(src, target)
    with patch.object(checkpoint.os, 'replace', side_effect=fail_twice), patch.object(checkpoint.time, 'sleep') as sleep:
        checkpoint.atomic_save(dst, {'generation': 2})
    assert len(attempts) == 3 and sleep.call_count == 2
    assert torch.load(dst, weights_only=True)['generation'] == 2
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows retry policy')
def test_persistent_denial_and_nontransient_errors_propagate(tmp_path):
    dst = tmp_path / 'last.pt'
    checkpoint.atomic_save(dst, {'generation': 1})
    old = dst.read_bytes()
    for code, expected_attempts in ((5, 6), (112, 1)):
        error = win_error(code)
        with patch.object(checkpoint.os, 'replace', side_effect=error) as replace, patch.object(checkpoint.time, 'sleep') as sleep:
            with pytest.raises(OSError) as caught:
                checkpoint.atomic_save(dst, {'generation': 2})
        assert caught.value is error
        assert replace.call_count == expected_attempts
        assert sleep.call_count == expected_attempts - 1
        assert dst.read_bytes() == old
        assert not list(tmp_path.glob('*.tmp'))


def kernel():
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    k.CreateFileW.restype = ctypes.c_void_p
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    return k


@pytest.mark.skipif(sys.platform != 'win32', reason='Real Windows sharing semantics')
def test_real_sharing_lock_releases_and_handles_do_not_leak(tmp_path):
    dst = tmp_path / 'last.pt'
    checkpoint.atomic_save(dst, {'generation': 1})
    k = kernel()
    locked = k.CreateFileW(str(dst), 0x80000000, 1, None, 3, 0, None)
    assert locked not in (None, ctypes.c_void_p(-1).value)
    unlocked = threading.Event()
    def release():
        time.sleep(.08)
        assert k.CloseHandle(locked)
        unlocked.set()
    thread = threading.Thread(target=release)
    thread.start()
    try:
        checkpoint.atomic_save(dst, {'generation': 2})
    finally:
        thread.join()
    assert unlocked.is_set() and torch.load(dst, weights_only=True)['generation'] == 2
    count = ctypes.c_uint32()
    assert k.GetProcessHandleCount(k.GetCurrentProcess(), ctypes.byref(count))
    before = count.value
    for value in range(50):
        checkpoint.atomic_save(dst, {'generation': value})
        assert torch.load(dst, weights_only=True)['generation'] == value
    assert k.GetProcessHandleCount(k.GetCurrentProcess(), ctypes.byref(count))
    assert count.value == before
    assert not list(tmp_path.glob('*.tmp'))
    (tmp_path / 'handle_observables.txt').write_text(f'50 save/load cycles; handles before={before}, after={count.value}; real sharing lock released, replacement succeeded', encoding='utf-8')
