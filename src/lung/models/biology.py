"""Explicitly masked RNA rank-program covariates, not biological rewards."""

import torch


def average_gene_ranks(expression):
    """Descending average ranks (one-based), exact handling of tied expression."""
    if expression.ndim != 2 or not torch.isfinite(expression).all():
        raise ValueError("Finite [cells,genes] log expression required for ranks.")
    order = torch.argsort(expression, dim=1, descending=True, stable=True)
    ordered = expression.gather(1, order)
    batch, genes = expression.shape
    starts = torch.ones_like(ordered, dtype=torch.bool)
    starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    groups = starts.cumsum(1) - 1
    ranks = torch.arange(1, genes + 1, device=expression.device, dtype=expression.dtype).expand(batch, -1)
    sums = torch.zeros_like(expression).scatter_add(1, groups, ranks)
    counts = torch.zeros_like(expression).scatter_add(1, groups, torch.ones_like(expression))
    average = (sums / counts.clamp_min(1)).gather(1, groups)
    return torch.zeros_like(expression).scatter(1, order, average)


def biological_features(expression, gene_sets, original_set_sizes, control_scores=None,
                        control_scale=None):
    """Return [27] features and [cells,3] scores for a single population.

    Gene sets are three already verified panel-index lists, or None per missing
    asset. No fallback gene list is fabricated. The caller records source hashes.
    """
    if len(gene_sets) != 3 or len(original_set_sizes) != 3 or expression.shape[1] < 2 or not len(expression):
        raise ValueError("Exactly three program assets and a nonempty gene panel are required.")
    ranks = average_gene_ranks(expression)
    values, absent = [], []
    for indices, full_size in zip(gene_sets, original_set_sizes):
        full_size = int(full_size or 0)
        if indices is None or not full_size:
            indices = []
        indices = torch.as_tensor(indices, device=expression.device, dtype=torch.long)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= expression.shape[1]).any():
            raise ValueError("Program member indices outside the gene panel.")
        if indices.unique().numel() != indices.numel():
            raise ValueError("Duplicated gene program members.")
        missing = len(indices) < 5 or len(indices) < 0.5 * full_size
        absent.append(missing)
        score = expression.new_zeros(len(expression)) if missing else (1 - (ranks[:, indices] - 1) / (expression.shape[1] - 1)).mean(1)
        values.append(score)
    scores = torch.stack(values, dim=-1)
    program_missing = expression.new_tensor(absent)
    means = scores.mean(0)
    if control_scores is None or control_scale is None:
        deltas, control_missing = torch.zeros_like(means), torch.ones_like(means)
    else:
        if control_scores.ndim != 2 or not len(control_scores) or control_scores.shape[1] != 3 or control_scale.shape != (3,):
            raise ValueError("Matched control scores [N,3] and frozen T scales [3] required.")
        available = torch.isfinite(control_scores).all(0) & torch.isfinite(control_scale) & ~program_missing.bool()
        deltas = torch.where(available, (means - control_scores.mean(0)) / control_scale.clamp_min(.05), torch.zeros_like(means))
        control_missing = (~available).to(expression.dtype)
    quantiles = torch.quantile(scores, expression.new_tensor([.1, .5, .9]), dim=0, interpolation="linear").T.flatten()
    centered = scores - means
    covariance = centered.T @ centered / len(scores)
    upper = torch.triu_indices(3, 3, device=expression.device)
    features = torch.cat((means, deltas, quantiles, covariance[upper[0], upper[1]], program_missing, control_missing))
    return features, scores
