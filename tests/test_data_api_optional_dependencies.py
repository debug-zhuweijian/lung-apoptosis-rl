"""Source-native recovery must not import the Parquet training stack eagerly."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_source_recovery_imports_without_pyarrow():
    code = '''
import importlib.abc
import sys
class BlockArrow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            raise ModuleNotFoundError('Parquet deliberately unavailable', name='pyarrow')
sys.meta_path.insert(0, BlockArrow())
from lung.data.external_gene_axis import align_gene_axis
from scripts.recover_public_expression import history_expression
assert 'lung.data.prepare' not in sys.modules
assert 'lung.data.reader' not in sys.modules
assert callable(align_gene_axis) and callable(history_expression)
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'src')]))
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_data_api_attribute_has_standard_error():
    import lung.data as data
    assert data.__all__ == ['TrainingData', 'prepare_training']
    try:
        data.this_api_does_not_exist
    except AttributeError as exc:
        assert 'this_api_does_not_exist' in str(exc)
    else:
        raise AssertionError('Unknown API attribute was silently accepted')
