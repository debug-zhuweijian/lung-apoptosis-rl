#!/usr/bin/env python3
"""Check reward V2 identities on synthetic neural inputs, not biological data.

This is not a DA-AC reproduction, policy training, or efficacy experiment.
Run: python scripts/verify_reward_v2_equivalence.py
Requires the existing PyTorch installation; writes no files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Batch:
    states: Tensor
    next_states: Tensor
    interpolated_actions: Tensor
    reward: Tensor
    continuing: Tensor
    potential: Tensor
    next_potential: Tensor


@dataclass(frozen=True)
class Result:
    residual: Tensor
    critic_gradients: tuple[Tensor, ...]
    actor_gradients: tuple[Tensor, ...]


def actor_parameters(actor: nn.Module, states: Tensor) -> Tensor:
    raw = actor(states)
    # Six synthetic distribution coordinates: two category probabilities
    # and two bounded center / log-scale parameters; no actual drug labels.
    return torch.cat(
        (raw[:, :2].softmax(dim=1), raw[:, 2:].tanh()), dim=1
    )


def critic_value(critic: nn.Module, states: Tensor, actions: Tensor) -> Tensor:
    return critic(torch.cat((states, actions), dim=1)).squeeze(-1)


def potential(states: Tensor) -> Tensor:
    return (states[:, 0] + 0.2 * states[:, 1] - states[:, 2]).sigmoid()


def maximum_difference(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def evaluate(
    batch: Batch,
    actor: nn.Module,
    critics: tuple[nn.Module, nn.Module],
    targets: tuple[nn.Module, nn.Module],
    *,
    shaped: bool,
    offset: int,
    omit_terminal_mask: bool = False,
) -> Result:
    reward = batch.reward
    if shaped:
        multiplier = 1 if omit_terminal_mask else batch.continuing
        reward = reward + multiplier * batch.next_potential - batch.potential

    with torch.no_grad():
        next_actions = actor_parameters(actor, batch.next_states)
        target_values = torch.stack(
            [
                critic_value(target, batch.next_states, next_actions)
                + offset * batch.next_potential
                for target in targets
            ],
            dim=1,
        )
        target = reward + batch.continuing * target_values.min(dim=1).values

    predictions = torch.stack(
        [
            critic_value(critic, batch.states, batch.interpolated_actions)
            + offset * batch.potential
            for critic in critics
        ],
        dim=1,
    )
    residual = predictions - target[:, None]
    critic_loss = residual.square().sum(dim=1).mean()
    critic_params = tuple(p for model in critics for p in model.parameters())
    critic_gradients = torch.autograd.grad(critic_loss, critic_params)

    actions = actor_parameters(actor, batch.states)
    actor_loss = -(
        critic_value(critics[0], batch.states, actions)
        + offset * batch.potential
    ).mean()
    actor_gradients = torch.autograd.grad(actor_loss, tuple(actor.parameters()))
    return Result(residual.detach(), critic_gradients, actor_gradients)


def main() -> None:
    torch.manual_seed(20260918)
    dtype = torch.float64
    actor = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 6)).to(dtype)
    critics = tuple(
        nn.Sequential(nn.Linear(9, 12), nn.Tanh(), nn.Linear(12, 1)).to(dtype)
        for _ in range(2)
    )
    targets = tuple(
        nn.Sequential(nn.Linear(9, 12), nn.Tanh(), nn.Linear(12, 1)).to(dtype)
        for _ in range(2)
    )
    states = torch.randn(24, 3, dtype=dtype)
    next_states = torch.randn(24, 3, dtype=dtype)
    continuing = torch.tensor([0, 1] * 12, dtype=dtype)
    with torch.no_grad():
        original = actor_parameters(actor, states)
        delta = torch.cat(
            (
                torch.tensor([[1.0, 0.0]], dtype=dtype).expand(24, -1),
                torch.rand(24, 2, dtype=dtype) * 2 - 1,
                -torch.ones(24, 2, dtype=dtype),
            ),
            dim=1,
        )
        omega = torch.rand(24, 1, dtype=dtype)
        interpolated = omega * original + (1 - omega) * delta
        batch = Batch(
            states,
            next_states,
            interpolated,
            (1 - continuing) * torch.rand(24, dtype=dtype),
            continuing,
            potential(states),
            potential(next_states),
        )

    cases = {
        "C0_terminal_plain": evaluate(batch, actor, critics, targets, shaped=False, offset=0),
        "C1_shaped_minus_phi": evaluate(batch, actor, critics, targets, shaped=True, offset=-1),
        "C2_shaped_plain": evaluate(batch, actor, critics, targets, shaped=True, offset=0),
        "C3_terminal_plus_phi": evaluate(batch, actor, critics, targets, shaped=False, offset=1),
    }
    tolerance = 1e-12
    all_params = tuple(p for c in critics for p in c.parameters()) + tuple(actor.parameters())
    checks = {}
    for name, left, right in [
        ("C0_equals_C1", cases["C0_terminal_plain"], cases["C1_shaped_minus_phi"]),
        ("C2_equals_C3", cases["C2_shaped_plain"], cases["C3_terminal_plus_phi"]),
    ]:
        critic_error = maximum_difference(left.critic_gradients, right.critic_gradients)
        actor_error = maximum_difference(left.actor_gradients, right.actor_gradients)
        residual_error = float((left.residual - right.residual).abs().max())
        left_gradients = left.critic_gradients + left.actor_gradients
        right_gradients = right.critic_gradients + right.actor_gradients
        left_step = tuple(p.detach() - 0.01 * g for p, g in zip(all_params, left_gradients))
        right_step = tuple(p.detach() - 0.01 * g for p, g in zip(all_params, right_gradients))
        step_error = maximum_difference(left_step, right_step)
        if max(critic_error, actor_error, residual_error, step_error) > tolerance:
            raise AssertionError(f"{name} failed")
        checks[name] = {
            "max_td_residual_difference": residual_error,
            "max_critic_gradient_difference": critic_error,
            "max_actor_gradient_difference": actor_error,
            "max_virtual_sgd_step_difference": step_error,
        }

    broken = evaluate(
        batch, actor, critics, targets,
        shaped=True, offset=-1, omit_terminal_mask=True,
    )
    terminal_error = float((broken.residual - cases["C0_terminal_plain"].residual).abs().max())
    if terminal_error <= 1e-6:
        raise AssertionError("Negative control did not detect the missing terminal mask")
    checks["missing_terminal_mask_detected"] = {"max_residual_difference": terminal_error}

    print(json.dumps({
        "scope": "synthetic algebra and autograd checks only; no biological model or policy training",
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": str(dtype),
        "batch_size": 24,
        "terminal_samples": 12,
        "continuing_samples": 12,
        "same_icl_inputs": True,
        "tolerance": tolerance,
        "checks": checks,
        "passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()

