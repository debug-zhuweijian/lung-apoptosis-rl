"""Response noise and convergence-aware artifacts."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lung.engine.inference_noise import response_noise


def test_noise_is_drawn_in_declared_space_and_legacy_rng_is_unchanged():
    reference = torch.zeros(3, 8)
    seed = 81
    actual = response_noise(object(), reference, torch.Generator().manual_seed(seed))
    assert torch.equal(actual, torch.randn(reference.shape, generator=torch.Generator().manual_seed(seed)))
    class Latent:
        def sample_noise(self, ref, generator=None):
            return torch.randn((len(ref), 2), generator=generator, device=ref.device, dtype=ref.dtype)
    assert response_noise(Latent(), reference, torch.Generator().manual_seed(seed)).shape == (3, 2)


def test_auxiliary_stops_on_validation_and_saves_honest_completion(tmp_path):
    from types import SimpleNamespace
    from lung.engine.auxiliary import fit_epochs
    from lung.engine.checkpoint import load_checkpoint
    from lung.engine.training_control import checkpoint_completion
    torch.set_num_threads(1)
    model = torch.nn.Linear(1, 1)
    config = {"epochs": 10, "aux_lr": 0.01, "aux_batch": 1,
              "early_stopping": {"enabled": True, "patience": 2}}
    monitor = SimpleNamespace(weights=tmp_path, progress=lambda it, **kwargs: it, record=lambda *a, **kw: None)
    loss = lambda net, value: net(torch.tensor([[value]])).square().mean()
    selected = fit_epochs("pool", model, [1.], loss, SimpleNamespace(identity="data"), config,
                          monitor, "identity", {}, validation=lambda net: {"val/loss": 1.})
    saved = load_checkpoint(Path(selected).parent / "final.pt")
    completion = checkpoint_completion(saved, 10)
    assert completion["actual_epochs"] == 3
    assert completion["reason"] == "validation_plateau"
    assert load_checkpoint(selected)["epoch"] == 1


def test_auxiliary_without_selection_does_not_stop_using_train_loss(tmp_path):
    from types import SimpleNamespace
    from lung.engine.auxiliary import fit_epochs
    from lung.engine.checkpoint import load_checkpoint
    model = torch.nn.Linear(1, 1)
    config = {"epochs": 4, "early_stopping": {"enabled": True, "patience": 1}}
    monitor = SimpleNamespace(weights=tmp_path, progress=lambda it, **kwargs: it, record=lambda *a, **kw: None)
    selected = fit_epochs("backward", model, [1.], lambda net, v: net(torch.tensor([[v]])).square().mean(),
                          SimpleNamespace(identity="data"), config, monitor, "identity", {})
    saved = load_checkpoint(selected)
    assert saved["epoch"] == 4
    assert saved["extra"]["selection_available"] is False
