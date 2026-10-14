from dataclasses import replace
import numpy as np
import pytest
from lung.data.normalization_bridge import NormalizationCoordinates, bridge_standardized


def coordinate(**kwargs):
    return NormalizationCoordinates(("a", "b"), np.array([1., 2.]), np.array([2., 3.]),
                                    "common_library_log1p", "same_counts", "a" * 64, **kwargs)


def test_affine_and_roundtrip():
    source = coordinate()
    target = replace(source, mean=np.array([4., 5.]), scale=np.array([1., 2.]))
    raw = np.array([[6., 8.]])
    z = ((raw - source.mean) / source.scale).astype(np.float32)
    converted = bridge_standardized(z, source, target)
    np.testing.assert_allclose(converted, (raw - target.mean) / target.scale)
    np.testing.assert_allclose(bridge_standardized(converted, target, source), z)
    assert converted.dtype == np.float32 and not source.mean.flags.writeable


@pytest.mark.parametrize("change", [dict(gene_ids=("b", "a")), dict(base_transform="different_library"), dict(units="TPM")])
def test_reject_incompatible_coordinates(change):
    source = coordinate()
    with pytest.raises(ValueError):
        bridge_standardized(np.zeros((1, 2)), source, replace(source, **change))


@pytest.mark.parametrize("change", [dict(scale=np.array([0., 1.])), dict(scale=np.array([np.inf, 1.])),
    dict(mean=np.array([np.nan, 1.])), dict(clipping="[-10,10]"), dict(source_sha256=""),
    dict(gene_ids=("a", "a")), dict(mean=np.array([1.])),
    dict(mean=np.array([1.+2j, 1.])), dict(scale=np.array([1., 2.+3j]))])
def test_reject_invalid_contract(change):
    with pytest.raises(ValueError):
        replace(coordinate(), **change)


@pytest.mark.parametrize("values", [np.array([np.nan, 1.]), np.zeros(3), np.array(1.), np.array([1.+2j, 3.])])
def test_reject_invalid_values(values):
    with pytest.raises(ValueError):
        bridge_standardized(values, coordinate(), coordinate())


def test_load_requires_exact_asset_hash(tmp_path):
    path = tmp_path / "normalization.npz"
    np.savez(path, gene_ids=np.array(["a", "b"]), mean=np.zeros(2), scale=np.ones(2))
    with pytest.raises(ValueError, match="SHA256"):
        NormalizationCoordinates.load(path, expected_sha256="a" * 64, base_transform="log1p", units="count")
