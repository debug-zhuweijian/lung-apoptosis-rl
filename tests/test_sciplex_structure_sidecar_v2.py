"""Regression checks for literal dummy atoms and relative stereo provenance."""
import numpy as np
import pytest

from scripts import build_sciplex_structure_sidecar_v2 as v2
from test_sciplex_structure_sidecar import inputs


def test_unknown_dummy_cannot_silently_enter_fingerprint():
    with pytest.raises(ValueError, match="Unreviewed dummy"):
        v2.build_records(*inputs("*.CCO"))


def test_normal_structure_still_has_usable_binary_fingerprint():
    rows, fp, _, candidates, variants = v2.build_records(*inputs())
    assert rows.iloc[0].eligible_for_structural_features
    assert rows.iloc[0].fingerprint_row == 0
    assert fp.shape == (1, 2048) and np.isin(fp, [0, 1]).all()
    assert candidates.empty and variants.shape == (0, 2048)


@pytest.mark.skipif(not v2.SDF.exists(), reason="Local reviewed primary SDF is not distributed in Git")
def test_reviewed_relative_structure_is_not_a_unique_absolute_input():
    source, reference, conditions = inputs("*.Cl.N[C@@H]1C[C@H]1c1ccccc1")
    source = source.assign(catalog_number="S4246")
    rows, fp, _, candidates, variants = v2.build_records(source, reference, conditions)
    assert not rows.iloc[0].eligible_for_structural_features
    assert rows.iloc[0].fingerprint_row == -1 and fp.shape == (0, 2048)
    assert rows.iloc[0].identity_id == v2.RELATIVE_ID
    assert set(candidates.rdkit_inchikey) == {"ZPEFMSTTZXJOTM-OULXEKPRSA-N", "ZPEFMSTTZXJOTM-RJUBDTSPSA-N"}
    assert candidates.mixture_weight.isna().all() and not candidates.observed_material_identity.any()
    assert variants.shape == (2, 2048)


@pytest.mark.skipif(not v2.SDF.exists(), reason="Local reviewed primary SDF is not distributed in Git")
def test_relative_evidence_cannot_be_reused_for_a_different_source_structure():
    with pytest.raises(ValueError, match="evidence differ"):
        v2.relative_candidates("*.Cl.N[C@H]1C[C@@H]1c1ccccc1", v2.SDF)
