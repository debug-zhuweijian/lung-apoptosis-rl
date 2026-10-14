"""Real-source metadata isolation, multi-background domains, fixed blocks."""
import json
from pathlib import Path

import numpy as np
import pytest

from lung.data.martin_state_function import DOMAIN_IDS, MartinStateFunctionInputs

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/bridge_v2"


@pytest.fixture
def inputs():
    return MartinStateFunctionInputs(BRIDGE, project_root=ROOT)


def test_metadata_serving_does_not_deserialize_function_labels(monkeypatch):
    original = Path.read_text

    def guard(path, *args, **kwargs):
        if path.name in ("observations.json", "published_means.csv"):
            raise AssertionError("functional labels read by metadata serving")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard)
    data = MartinStateFunctionInputs(BRIDGE, project_root=ROOT)
    assert data.manifest["counts"]["core_observations"] == 20
    for cid in data.candidates:
        packet = data.model_input(cid)
        assert "value" not in json.dumps({k: v for k, v in packet.items() if k != "chemical_fingerprint"})
        assert packet["chemical_fingerprint"].shape == (2048,)
        assert not packet["chemical_fingerprint"].flags.writeable
        assert packet["response_request"]["target_observation_h"] == 72
        assert packet["response_request"]["standard_Tahoe_24h_descriptor_eligible"] is False
    for block in data.blocks.values():
        assert data.decision_inputs(block["candidate_ids"])["response_features_status"].startswith("pending")


def test_training_domain_allows_three_backgrounds_but_decision_requires_one(inputs):
    fold_id = "leave_background_out:PC9"
    for domain_id in DOMAIN_IDS:
        domain = inputs.training_domain(fold_id, domain_id)
        assert len(domain["fit_state_references"]) == 3
        assert len(domain["fit_candidate_ids"]) == (9 if domain_id.endswith("S2F") else 6)
        blocks = [b for b in inputs.blocks.values() if b["measurement_domain_id"] == domain_id]
        inputs.decision_inputs(blocks[0]["candidate_ids"])
        with pytest.raises(ValueError, match="MIXED"):
            inputs.decision_inputs(blocks[0]["candidate_ids"] + blocks[1]["candidate_ids"])


def test_core_blocks_cannot_mix_panels_or_subset_outcomes(inputs):
    first = next(iter(inputs.blocks.values()))
    ids = first["candidate_ids"]
    with pytest.raises(ValueError, match="COMPLETE"):
        inputs.decision_inputs(ids[:1])
    with pytest.raises(ValueError, match="DUPLICATE"):
        inputs.decision_inputs(ids + ids)
    other = next(b for b in inputs.blocks.values() if b["cell_background"] == first["cell_background"]
                 and b["measurement_domain_id"] != first["measurement_domain_id"])
    with pytest.raises(ValueError, match="MIXED"):
        inputs.decision_inputs(ids + other["candidate_ids"])


def test_real_bulk_reference_and_missing_axis_boundaries(inputs):
    for ref_id, reference in inputs.references.items():
        data = inputs.reference_expression(ref_id)
        assert data["values"].shape == (19712, 3)
        assert np.isfinite(data["values"]).all() and not data["values"].flags.writeable
        assert data["measurement_level"] == "bulk"
        assert reference["same_well_function_pairing"] is False
        assert reference["functional_Y_per_RNA_replicate"] is False


def test_roles_are_complete_background_holds_not_cell_pseudorepeats(inputs):
    for fold_id, fold in inputs.folds.items():
        assert len(fold["fit_candidate_ids"]) == 15
        assert len(fold["final_test_candidate_ids"]) == 5
        assert not set(fold["fit_candidate_ids"]) & set(fold["final_test_candidate_ids"])
        for domain in DOMAIN_IDS:
            fit = inputs.observation_units(fold_id, "fit", domain)
            final = inputs.observation_units(fold_id, "final_test", domain)
            assert len(fit) == (9 if domain.endswith("S2F") else 6)
            assert len(final) == (3 if domain.endswith("S2F") else 2)
            for unit in fit + final:
                assert unit["weight"] == 1 and not unit["biological_function_replicate_ids"]
                assert unit["value_scale"].startswith("published_relative_caspase")
        with pytest.raises(ValueError, match="NO_SUCH"):
            inputs.observation_units(fold_id, "selection", DOMAIN_IDS[0])
