"""Independent tiny integration checks; no biological model training."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lung.models import ModelConfig, ResponseModel, rbf_mmd
from lung.engine.response import exact_mmd_backward
from lung.engine.data import EngineData, open_data

fixture_spec = importlib.util.spec_from_file_location("review_data_fixture", Path(__file__).with_name("test_training_data_v1.py"))
fixture_module = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)


class IntegrationReviewTests(unittest.TestCase):
    def test_endpoint_replay_gradient_matches_full_batch_autograd(self):
        torch.manual_seed(141)
        shape = ModelConfig(n_genes=6, d_model=8, n_heads=2, encoder_layers=1,
                            pad_layers=1, query_chunk=2, context_dim=2)
        graph = torch.tensor([[i, (i + 1) % 6, (i + 2) % 6] for i in range(6)])
        first = ResponseModel(shape, graph)
        # Exercise attention, E_X and time paths beyond their zero-gate startup.
        for block in first.velocity_model.blocks:
            for modulation in block.modulations:
                torch.nn.init.normal_(modulation[-1].bias, std=.1)
        replay = copy.deepcopy(first)
        x, y, noise = torch.randn(3, 6), torch.randn(3, 6), torch.randn(3, 6)
        fingerprint = torch.randn(1, 2048)

        def encode(model, n):
            return model.encode_condition(fingerprint.expand(n, -1), torch.ones(n),
                 torch.full((n,), 24.), torch.zeros(n, dtype=torch.long),
                 torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long),
                 torch.zeros(n), torch.full((n,), 24.))

        full = torch.cat([first.sample(x[i:i+1], encode(first, 1), noise=noise[i:i+1],
                                       steps=3, checkpoint=False) for i in range(len(x))])
        objective = rbf_mmd(full, y, bandwidth=1.3) * .37
        objective.backward()
        loss = exact_mmd_backward(replay, x, y, lambda n: encode(replay, n),
                                  bandwidth=1.3, steps=3, scale=.37, noise=noise)
        self.assertAlmostEqual(float(objective.detach()) / .37, loss, places=5)
        nonzero = 0
        for (name, left), (_, right) in zip(first.named_parameters(), replay.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=4e-5, atol=2e-6, msg=name)
                nonzero += int(torch.count_nonzero(left.grad) > 0)
        self.assertGreater(nonzero, 30)

    def test_state_stage_reader_permits_separate_temperature_calibration(self):
        from lung.data import prepare_training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_module.tiny_bundle(root)
            prepare_training(root)
            config = {"data": str(root / "configs/dataset.yaml"), "synthetic": False}
            with patch("lung.engine.data.ROOT", root):
                data = open_data(config, "state", prepare=False)
            arrays = data.state_arrays("1_to_8", "calibration")
            self.assertEqual(set(arrays["labels"]), {0, 1, 2})

    def test_engine_real_metadata_bridge_fields_without_preprocessing(self):
        # This test is completely synthetic, but calls the actual production reader.
        from lung.data import prepare_training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_module.tiny_bundle(root)
            prepare_training(root, build_preprocessing=False)
            config = {"data": str(root / "configs/dataset.yaml")}
            with patch("lung.engine.data.ROOT", root):
                data = EngineData(config, require_preprocessing=False)
            metadata = data.action_features(data.condition_ids("T")[0])
            self.assertEqual(metadata["fingerprint"].shape, (2048,))
            self.assertEqual(metadata["duration"], 24.)
            self.assertIsNone(data.mu)
            self.assertIsNone(data.bandwidth)
            self.assertEqual(data.manifest["status"], "metadata_ready")
            with self.assertRaises(PermissionError):
                data.condition_ids("E")

    def test_world_rejects_auxiliary_after_upstream_weights_change(self):
        from lung.engine.auxiliary import dependency_record
        from lung.engine.world import BiologicalWorld
        from lung.engine.config import StageUnavailable
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, auxiliary = root / "response.pt", root / "auxiliary.pt"
            source.write_bytes(b"synthetic-original-model")
            dependencies = dependency_record({"response": [str(source)]}, ("response",))
            torch.save({"schema_version": 1, "metadata": {"dependencies": dependencies}}, auxiliary)
            source.write_bytes(b"synthetic-changed-model")
            pipeline = {"response": [str(source)], "pool": str(auxiliary), "state": str(auxiliary)}
            with patch("lung.engine.world.load_response") as loader:
                with self.assertRaisesRegex(StageUnavailable, "AUX_DEPENDENCY_MISMATCH"):
                    BiologicalWorld(SimpleNamespace(identity="test"),
                                    {"device": "cpu", "task": "PROXY-A549-V1"},
                                    pipeline)
                loader.assert_not_called()




    def test_pipeline_rejects_changed_selected_weights(self):
        from lung.engine.train import write_pipeline, read_pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "best.pt"
            best.write_bytes(b"synthetic-selected-version-one")
            pipeline = {"response": [str(best)]}
            monitor = SimpleNamespace(experiment=root, config={"synthetic": True, "task": "RESPONSE-A549-V1"})
            write_pipeline(monitor, SimpleNamespace(identity="synthetic-data"), pipeline)
            restored = read_pipeline({"pipeline": str(root / "pipeline.json")})
            self.assertEqual(restored["response"], [str(best)])
            best.write_bytes(b"synthetic-changed-selected-weights")
            with self.assertRaisesRegex(ValueError, "FROZEN_PIPELINE_ARTIFACT_CHANGED"):
                read_pipeline({"pipeline": str(root / "pipeline.json")})


if __name__ == "__main__":
    unittest.main()
