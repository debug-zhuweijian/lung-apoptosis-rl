"""CPU equivalence, bounded compact data and restart tests for auxiliary execution."""

import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.engine.auxiliary import fit_epochs, state_batch_loss, encode_vectors
from lung.engine.checkpoint import load_checkpoint, atomic_save
from lung.engine.residuals import BackwardPredictionCache, compact_prediction, BACKWARD_FIELDS
from lung.models import StateClassifier
from lung.models.state_objectives import state_loss
from test_auxiliary_v1 import setup as auxiliary_setup


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("objective", ["ce", "weighted_ce", "ce_aggregate"])
@pytest.mark.parametrize("count", [1, 7])
def test_state_losses_gradients_and_adam_update(objective, count):
    torch.manual_seed(31)
    serial = StateClassifier(16)
    batched = copy.deepcopy(serial)
    x = torch.randn(count, 16)
    labels = torch.tensor([0, 0, 0, 0, 1, 2, 0])[:count]
    weights = torch.tensor([.5, 2., 3.])
    options = {"state_loss": objective, "state_aggregate_lambda": .7}
    opt_serial = torch.optim.Adam(serial.parameters(), lr=.003)
    opt_batch = torch.optim.Adam(batched.parameters(), lr=.003)
    losses = []
    rng = torch.get_rng_state().clone()
    for vector, label in zip(x, labels):
        value = state_loss(serial(vector).reshape(1, 3), label.reshape(1),
                           objective=objective, aggregate_lambda=.7, weights=weights)[0]
        (value / count).backward()
        losses.append(value.detach())
    calls = []
    handle = batched.register_forward_hook(lambda *args: calls.append(1))
    values = state_batch_loss(batched, list(zip(x, labels)), "cpu", options, weights)
    values.mean().backward()
    handle.remove()
    torch.testing.assert_close(values, torch.stack(losses), atol=2e-6, rtol=2e-6)
    gradient_error = 0.
    for before, after in zip(serial.parameters(), batched.parameters()):
        gradient_error = max(gradient_error, float((before.grad - after.grad).abs().max()))
        torch.testing.assert_close(before.grad, after.grad, atol=2e-6, rtol=2e-6)
    opt_serial.step()
    opt_batch.step()
    for before, after in zip(serial.parameters(), batched.parameters()):
        torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)
    assert torch.equal(rng, torch.get_rng_state())
    assert calls == [1]
    print({"objective": objective, "units": count, "forward_calls": len(calls),
           "gradient_max_abs": gradient_error})


class Monitor:
    def __init__(self, path):
        self.weights = path

    def progress(self, values, **kwargs):
        return values

    def record(self, *args):
        pass


def test_fit_epochs_tail_and_resume(tmp_path):
    torch.manual_seed(41)
    initial = StateClassifier(8)
    x, labels = torch.randn(11, 8), torch.arange(11) % 3
    rows = list(zip(x, labels))
    objective = {"state_loss": "weighted_ce", "state_aggregate_lambda": 0.}
    weights = torch.tensor([.7, 1.1, 2.])
    data = SimpleNamespace(identity="fixture")
    config = {"epochs": 2, "seed": 7, "aux_lr": .001}
    def scalar(model, row):
        return state_loss(model(row[0]).reshape(1, 3), row[1].reshape(1), weights=weights)[0]
    def batch(model, selected):
        return state_batch_loss(model, selected, "cpu", objective, weights)
    old, full, resumed = (copy.deepcopy(initial) for _ in range(3))
    for name, model, epochs, batched, resume in [
        ("serial", old, 2, None, False), ("full", full, 2, batch, False),
        ("restart", resumed, 1, batch, False), ("restart", resumed, 2, batch, True),
    ]:
        fit_epochs("state", model, rows, scalar, data, {**config, "epochs": epochs},
                   Monitor(tmp_path / name), "fixture-v2", {}, resume=resume,
                   batch_size=4, batch_loss=batched)
    for name in full.state_dict():
        torch.testing.assert_close(old.state_dict()[name], full.state_dict()[name], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(full.state_dict()[name], resumed.state_dict()[name], atol=0, rtol=0)
    saved = load_checkpoint(tmp_path / "restart/state/last.pt")
    assert saved["counters"]["optimizer_step"] == 6
    assert saved["epoch"] == 2
    print({"tail_units": 3, "optimizer_steps": 6, "resume_bitwise_equal": True})


def prediction(genes=32461):
    return {"raw": torch.randn(32, genes), "observed": torch.randn(32, genes),
            "observed_population": torch.randn(16), "condition": torch.randn(16),
            "reference_mean": torch.randn(genes)}


def test_compact_prediction_io_bounded_cache_and_mutation(tmp_path):
    original = prediction()
    source = tmp_path / "prediction.pt"
    atomic_save(source, original)
    compact = compact_prediction(source, original)
    descriptor = {"path": str(source), "compact_path": str(compact)}
    cache = BackwardPredictionCache(max_bytes=140000)
    loads = []
    real_load = torch.load
    def tracked(path, *args, **kwargs):
        loads.append(str(path))
        return real_load(path, *args, **kwargs)
    with patch.object(torch, "load", tracked):
        first = cache.read(descriptor, "cpu")
        second = cache.read(descriptor, "cpu")
    assert loads == [str(compact)]
    assert set(first) == set(BACKWARD_FIELDS)
    for name in BACKWARD_FIELDS:
        torch.testing.assert_close(first[name], original[name], rtol=0, atol=0)
        torch.testing.assert_close(second[name], original[name], rtol=0, atol=0)
    assert cache.nbytes <= cache.max_bytes
    assert compact.stat().st_size < source.stat().st_size / 10
    tiny = BackwardPredictionCache(max_bytes=4)
    tiny.read(descriptor, "cpu")
    assert tiny.nbytes == 0 and not tiny.entries
    changed = {**original, "reference_mean": original["reference_mean"] + 1}
    atomic_save(source, changed)
    with pytest.raises(ValueError, match="RESIDUAL_COMPACT_SOURCE_MISMATCH"):
        cache.read(descriptor, "cpu")
    compact_prediction(source, changed)
    torch.testing.assert_close(cache.read(descriptor, "cpu")["reference_mean"], changed["reference_mean"])
    print({"full_genes": 32461, "compact_bytes": compact.stat().st_size,
           "source_bytes": source.stat().st_size, "hot_load_count": len(loads),
           "cache_tensor_bytes": cache.nbytes})


def test_compact_cache_interruption_leaves_previous_file(tmp_path):
    source = tmp_path / "p.pt"
    original = prediction(11)
    atomic_save(source, original)
    compact = compact_prediction(source, original)
    before = compact.read_bytes()
    changed = {**original, "reference_mean": original["reference_mean"] + 2}
    atomic_save(source, changed)
    with patch("lung.engine.checkpoint.os.replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            compact_prediction(source, changed)
    assert compact.read_bytes() == before
    compact_prediction(source, changed)
    record = BackwardPredictionCache().read({"path": str(source), "compact_path": str(compact)}, "cpu")
    torch.testing.assert_close(record["reference_mean"], changed["reference_mean"])


def test_encode_vectors_tensor_path_matches_numpy():
    class Encoder:
        def encode_cells(self, x):
            return None, x.square() + .2
    x = torch.randn(7, 11)
    legacy = encode_vectors(Encoder(), x.numpy(), "cpu")
    direct = encode_vectors(Encoder(), x, "cpu", output_device="cpu")
    torch.testing.assert_close(legacy, direct, rtol=0, atol=0)


def test_prediction_generation_matches_legacy_and_rng(auxiliary_setup):
    from lung.engine.auxiliary import train_pool, pipeline_identity, load_auxiliary
    from lung.engine.residuals import prediction_rows, residual_dependencies
    from lung.engine.response import load_response, tensor, condition_kwargs, projected
    data, config, monitor, pipeline = auxiliary_setup
    pipeline = train_pool(data, config, monitor, pipeline)
    # Two members exercise constructor RNG consumption between generated populations.
    pipeline = {**pipeline, "response": pipeline["response"] * 2}
    identity = pipeline_identity("residual_predictions", data, config, residual_dependencies(pipeline))
    torch.manual_seed(531)
    reference = load_response(pipeline["response"][0], data, "cpu")
    pool = load_auxiliary(pipeline["pool"], data, "cpu")
    expected = {}
    for key in data.condition_ids("R"):
        seed = int(hashlib.sha256((identity + str(key)).encode()).hexdigest()[:8], 16)
        batch = data.sample_response(key, config["population"], np.random.default_rng(seed), role="R")
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            raw_members, means = [], []
            for index, checkpoint in enumerate(pipeline["response"]):
                member = reference if index == 0 else load_response(checkpoint, data, "cpu")
                generated = []
                for cell in range(len(batch["control"])):
                    source = tensor(batch["control"][cell:cell+1], "cpu")
                    condition = member.encode_condition(**condition_kwargs(batch, 1, "cpu"))
                    generated.append(member.sample(source, condition, steps=1, checkpoint=False).cpu())
                raw = torch.cat(generated)
                means.append(projected(raw, data).mean(0).cpu())
                raw_members.append(raw)
            raw = torch.cat(raw_members)
            valid = projected(raw, data).cpu()
            vectors = encode_vectors(reference, valid.numpy(), "cpu")
            observed_vectors = encode_vectors(reference, batch["target"], "cpu")
            expected[key] = {"raw": raw, "population": pool(vectors).cpu(),
                             "observed_population": pool(observed_vectors).cpu(),
                             "ensemble_variance": torch.stack(means).var(0, correction=0).mean(),
                             "condition": reference.encode_condition(**condition_kwargs(batch, 1, "cpu"))[0].cpu()}
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(531)
    actual = prediction_rows("R", data, config, monitor, pipeline)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for descriptor in actual:
        saved = torch.load(descriptor["path"], weights_only=False)
        for key, value in expected[descriptor["condition_id"]].items():
            torch.testing.assert_close(saved[key], value, atol=0, rtol=0)
    print({"legacy_prediction_bitwise_equal": True, "members": 2, "rng_equal": True})
