"""The inspector must observe constructor bytes without executing them."""
import importlib.util
import pickle
import sys
import zipfile
import struct
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("checkpoint_inspector", Path(__file__).parents[1] / "scripts/inspect_checkpoint_metadata.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def archive(path, data):
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr("archive/data.pkl", data)
    return path


def write_sentinel(path):
    Path(path).write_text("executed", encoding="utf-8")


def test_dangerous_reduce_stays_opaque(tmp_path):
    sentinel = tmp_path / "sentinel"

    class Dangerous:
        def __reduce__(self):
            return write_sentinel, (str(sentinel),)

    data = pickle.dumps({"schema_version": 1, "epoch": 7, "identity": "id",
                         "metadata": {"stage": "ae", "synthetic": False, "data_identity": "d"},
                         "counters": {"optimizer_step": 12}, "rng": Dangerous()}, protocol=2)
    result = module.inspect(archive(tmp_path / "x.pt", data))
    assert result["fields"]["metadata"]["data_identity"] == "d"
    assert result["fields"]["epoch"] == 7
    assert result["fields"]["counters"]["optimizer_step"] == 12
    root, _ = module.parse_symbols(data)
    assert isinstance(root["rng"], module.Symbol)
    assert not sentinel.exists()


@pytest.mark.parametrize("data", [b"\x80\x02}q\x00(U\x01xK\x01u.",
                                      b"\x80\x03}.", b"\x80\x02}(X\x01\x00\x00\x00xK\x01X\x01\x00\x00\x00xK\x02u."])
def test_reject_unsupported_protocol_and_duplicate(data):
    with pytest.raises(ValueError):
        module.parse_symbols(data)


def test_cap_nonzip_and_trailing_bytes(tmp_path):
    path = archive(tmp_path / "x.pt", pickle.dumps({"epoch": 1}, protocol=2))
    with pytest.raises(ValueError, match="byte cap"):
        module.inspect(path, max_bytes=2)
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not zip")
    with pytest.raises(zipfile.BadZipFile):
        module.inspect(bad)
    with pytest.raises(ValueError, match="trailing"):
        module.parse_symbols(pickle.dumps({}, protocol=2) + b"x")


def test_small_tensor_checkpoint_is_static(tmp_path):
    import torch

    path = tmp_path / "tensor.pt"
    torch.save({"schema_version": 1, "epoch": 2, "model": {"weight": torch.ones(2, 3)},
                "metadata": {"stage": "fixture", "synthetic": True}}, path)
    result = module.inspect(path)
    assert result["fields"]["metadata"]["synthetic"] is True
    assert result["model_tensor_shapes"].startswith("unknown")
    assert "BINPERSID" in result["opcodes"]


@pytest.mark.parametrize("payload", [
    b"\x80\x02}(K\x01\x86.",  # TUPLE2 cannot consume MARK.
    b"\x80\x02}(q\x00h\x00u.",  # Memo cannot contain MARK.
    b"\x80\x02}cunused\nconstructor\nK\x01Rs.",
    b"\x80\x02}cunused\nconstructor\nK\x01\x81s.",
    b"\x80\x02}cunused\nconstructor\n(Rs.",
    b"\x80\x02}(Qs.",
    b"\x80\x02}cunused\nconstructor\n)R(bs.",
    b"\x80\x02q\x00.",
])
def test_reject_malformed_operands(payload):
    with pytest.raises(ValueError):
        module.parse_symbols(payload)


def test_unsupported_zip_compression_is_rejected(tmp_path):
    path = archive(tmp_path / "unsupported.pt", pickle.dumps({"epoch": 1}, protocol=2))
    data = bytearray(path.read_bytes())
    local, central = data.index(b"PK\x03\x04"), data.index(b"PK\x01\x02")
    struct.pack_into("<H", data, local + 8, 99)
    struct.pack_into("<H", data, central + 10, 99)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="ZIP payload"):
        module.inspect(path)
