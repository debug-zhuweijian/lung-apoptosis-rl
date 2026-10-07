"""Read-only native choices; independent world re-ranking is reported separately."""
import torch

from lung.rl import ConditionalActor, JointActor, stream_seed
from lung.rl.networks import AffineCritic, DistributionCritic
from .config import StageUnavailable


def load_policy_models(saved, registry, device, kind):
    """Load only inference parameters, using the architecture stored in the checkpoint."""
    cfg = saved.get("learner", {}).get("config", {})
    actor_type = cfg.get("actor_type", "conditional")
    if actor_type not in {"conditional", "joint_control"}:
        raise ValueError("UNKNOWN_SAVED_ACTOR_ARCHITECTURE")
    critic_type = cfg.get("critic_type", "affine")
    with torch.random.fork_rng(devices=[]):
        actor = (JointActor if actor_type == "joint_control" else ConditionalActor)(registry)
        actor.load_state_dict(saved["model"])
        critics = None
        if kind == "daac":
            if "critics" not in saved.get("learner", {}):
                raise StageUnavailable("SAVED_CRITICS_REQUIRED_FOR_NATIVE_Q_EVALUATION")
            if critic_type not in {"affine", "icl_control"}:
                raise ValueError("UNKNOWN_SAVED_CRITIC_ARCHITECTURE")
            cls = AffineCritic if critic_type == "affine" else DistributionCritic
            critics = torch.nn.ModuleList([cls(registry), cls(registry)])
            critics.load_state_dict(saved["learner"]["critics"])
    actor = actor.to(device).eval().requires_grad_(False)
    if critics is not None:
        critics = critics.to(device).eval().requires_grad_(False)
    return actor, critics, critic_type


def candidate_q(critic, state, count, critic_type):
    if critic_type == "affine":
        return critic([state])[0]
    # ICL is a different architecture: evaluate complete-action simplex vertices.
    device = next(critic.parameters()).device
    rows = []
    for start in range(0, count, 64):
        ids = torch.arange(start, min(start + 64, count), device=device)
        vertices = torch.nn.functional.one_hot(ids, count).float()
        rows.append(critic.distribution_value([state] * len(ids), vertices))
    return torch.cat(rows)


@torch.no_grad()
def native_sequence(actor, adapter, root, seed, *, critics=None, critic_type="affine"):
    """Greedy native Actor or Q1 choices with explicit ties, without outcome access."""
    prefix, trace = (), []
    while not adapter.protocol.position(adapter.registry, prefix)[2]:
        state = adapter.observe(root, prefix, stream_seed(seed, "native-planning"))
        mask = adapter.protocol.legal_mask(adapter.registry, prefix)
        probabilities = actor([state], mask[None])[0].detach().cpu()
        q = [candidate_q(c, state, len(adapter.registry), critic_type).detach().cpu()
             for c in critics] if critics is not None else None
        values = q[0] if q is not None else probabilities
        if not torch.isfinite(values[mask]).all():
            raise FloatingPointError("NONFINITE_NATIVE_ACTION_SCORE")
        ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        ranked = sorted(ids, key=lambda i: (-float(values[i]), adapter.registry.candidates[i].action_id))
        best = ranked[0]
        ties = [adapter.registry.candidates[i].action_id for i in ranked if values[i] == values[best]]
        ranking = []
        for i in ranked:
            candidate = adapter.registry.candidates[i]
            row = {"action_id": candidate.action_id, "drug": candidate.drug_id,
                   "dose_id": candidate.dose_id, "time_id": candidate.time_id,
                   "duration_hours": candidate.duration_ticks / adapter.protocol.ticks_per_hour,
                   "actor_probability": float(probabilities[i])}
            if q is not None:
                row.update(q1=float(q[0][i]), q2=float(q[1][i]))
            ranking.append(row)
        trace.append({"prefix": list(prefix), "ranking": ranking, "top_ties": ties,
                      "selection": "q1_greedy" if q is not None else "actor_greedy",
                      "tie_rule": "action_id_order_only_no_dose_or_time_preference",
                      "value_semantics": ("shaped_frozen_model_value_not_apoptosis_rate" if q is not None
                                          else "actor_action_probability_not_outcome_probability")})
        prefix = (*prefix, adapter.registry.candidates[best].action_id)
    adapter.protocol.validate_complete(adapter.registry, prefix)
    return prefix, trace
