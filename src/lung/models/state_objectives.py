"""Scores for original TIDYI sorting labels, not biological apoptosis outcomes."""

import torch
import torch.nn.functional as F


def aggregate_nll(logits, labels):
    """Per-cell binary NLL for classes {1, 2} from the SAME 3-way softmax.

    Computing in log space avoids clipping saturated probabilities (and gradients).
    """
    log_probabilities = F.log_softmax(logits, dim=-1)
    positive_log_probability = torch.logsumexp(log_probabilities[..., 1:], dim=-1)
    return -torch.where(labels > 0, positive_log_probability, log_probabilities[..., 0])


def state_loss(logits, labels, *, objective="weighted_ce", aggregate_lambda=0., weights=None):
    """Per-cell loss; fixed fit-set weights preserve the legacy weighted CE."""
    ce = F.cross_entropy(logits, labels, reduction="none")
    if objective == "weighted_ce":
        if weights is None:
            raise ValueError("weighted_ce requires fixed fit-set class weights")
        return ce * weights[labels]
    if objective == "ce":
        return ce
    if objective == "ce_aggregate":
        # Exact lambda=0 equivalence, including gradients and extreme logits.
        return ce if aggregate_lambda == 0 else ce + aggregate_lambda * aggregate_nll(logits, labels)
    raise ValueError(f"Unknown state loss: {objective}")
