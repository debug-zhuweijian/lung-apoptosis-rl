"""Frozen-coordinate numerical objectives, no data fitting or file access."""

from dataclasses import dataclass
import math

import torch


def mean_squared_distance(left, right):
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Distance inputs require the same ordered gene axis.")
    return ((left.square().sum(-1)[:, None] + right.square().sum(-1)[None]
             - 2 * left @ right.T) / left.shape[1]).clamp_min(0)


@torch.no_grad()
def entropic_ot(source, target, epsilon=0.05, max_iter=10000, tolerance=1e-6,
                return_info=False):
    """Minibatch coupling prior with verified uniform marginals; not pairing truth.

    After 3000 scaling updates, use guarded dual Newton steps for nearly sparse
    kernels. Each outer iteration has at most one Newton solve and ten backtracks.
    Explicit max_iter remains a strict outer-iteration cap; the Gibbs kernel,
    FP64 precision and marginal tolerance are unchanged.
    """
    if epsilon <= 0 or max_iter < 1 or tolerance <= 0:
        raise ValueError("Invalid Sinkhorn numerical parameters.")
    if not len(source) or not len(target):
        raise ValueError("Empty OT marginal.")
    cost = mean_squared_distance(source.detach().double(), target.detach().double())
    if not torch.isfinite(cost).all():
        raise ValueError("Non-finite OT input.")
    rows, columns = cost.shape
    if not torch.count_nonzero(cost):
        coupling = torch.full_like(cost, 1 / (rows * columns))
        info = {"iterations": 0, "row_l1": 0.0, "column_l1": 0.0,
                "coupling_type": "minibatch_entropic_ot_prior"}
        return (coupling, info) if return_info else coupling
    # Linear 0.5 quantile is the conventional median for even sample counts.
    log_kernel = -cost / cost.quantile(.5).clamp_min(1e-6) / epsilon
    log_u, log_v = cost.new_zeros(rows), cost.new_zeros(columns)
    row_error = column_error = math.inf
    for iteration in range(1, max_iter + 1):
        if iteration > 3000 and rows > 1:
            # Eliminate the column potential: P_ij = softmax_i(K_ij + u_i)/m.
            # The reduced dual has gradient P1-a and Hessian diag(P1)-m PP^T.
            # This is the Schur complement of Brauer et al. (2018), Eq. (2.4),
            # https://arxiv.org/abs/1710.06635 . Fix the last potential's gauge;
            # the diagonal shift regularizes only the solve, never the OT cost.
            marginal = coupling.sum(1)
            hessian = torch.diag(marginal) - columns * (coupling @ coupling.T)
            reduced = hessian[:-1, :-1] + 1e-12 * torch.eye(
                rows - 1, dtype=cost.dtype, device=cost.device)
            direction, status = torch.linalg.solve_ex(reduced, (marginal - 1 / rows)[:-1])
            if status.item() == 0 and torch.isfinite(direction).all():
                direction = torch.cat((direction, direction.new_zeros(1)))
                for backtrack in range(10):
                    trial_u = log_u - (0.5 ** backtrack) * direction
                    trial_v = -math.log(columns) - torch.logsumexp(
                        log_kernel + trial_u[:, None], dim=0)
                    trial = (log_kernel + trial_u[:, None] + trial_v[None]).exp()
                    if ((trial.sum(1) - 1 / rows).abs().sum().item() < row_error
                            and (trial.sum(0) - 1 / columns).abs().sum().item() <= tolerance):
                        log_u, log_v = trial_u, trial_v
                        break
        log_u = -math.log(rows) - torch.logsumexp(log_kernel + log_v[None], dim=1)
        log_v = -math.log(columns) - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        coupling = (log_kernel + log_u[:, None] + log_v[None]).exp()
        row_error = (coupling.sum(1) - 1 / rows).abs().sum().item()
        column_error = (coupling.sum(0) - 1 / columns).abs().sum().item()
        if row_error <= tolerance and column_error <= tolerance:
            info = {"iterations": iteration, "row_l1": row_error, "column_l1": column_error,
                    "coupling_type": "minibatch_entropic_ot_prior"}
            return (coupling, info) if return_info else coupling
    raise RuntimeError(f"OT_NUMERICAL_FAILURE: row L1={row_error:.3g}, column L1={column_error:.3g}")


def sample_ot_pairs(coupling, n_pairs=128, generator=None):
    if coupling.ndim != 2 or n_pairs < 1 or not torch.isfinite(coupling).all():
        raise ValueError("Invalid OT coupling or pair count.")
    if (coupling < 0).any() or coupling.sum() <= 0:
        raise ValueError("OT coupling has invalid probability mass.")
    indices = torch.multinomial(coupling.reshape(-1), n_pairs, replacement=True, generator=generator)
    return indices // coupling.shape[1], indices % coupling.shape[1]


def rbf_kernel(left, right, bandwidth):
    bandwidth = torch.as_tensor(bandwidth, dtype=left.dtype, device=left.device)
    if bandwidth.numel() != 1 or not torch.isfinite(bandwidth).all() or (bandwidth <= 0).any():
        raise ValueError("A positive frozen training bandwidth is required.")
    scales = left.new_tensor([0.25, 0.5, 1.0, 2.0, 4.0]) * bandwidth
    distance = mean_squared_distance(left, right)
    return (-distance[..., None] / (2 * scales)).exp().mean(-1)


def rbf_mmd(generated, observed, bandwidth, unbiased=False):
    """Complete three-term multiscale MMD; U statistic only for final reporting."""
    n, m = len(generated), len(observed)
    if not n or not m or (unbiased and min(n, m) < 2):
        raise ValueError("Insufficient samples for requested MMD statistic.")
    xx, yy = rbf_kernel(generated, generated, bandwidth), rbf_kernel(observed, observed, bandwidth)
    xy = rbf_kernel(generated, observed, bandwidth)
    if unbiased:
        first = (xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
        second = (yy.sum() - yy.diagonal().sum()) / (m * (m - 1))
        return first + second - 2 * xy.mean()
    # V statistic is nonnegative algebraically; clamp floating cancellation only.
    return (xx.mean() + yy.mean() - 2 * xy.mean()).clamp_min(0)


def kernel_sum(left, right, bandwidth, *, block_size=128, exclude_block_diagonal=False):
    """Full-gene kernels with bounded pair matrices; diagonal means row position."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError("Positive kernel block size required")
    total = left.new_zeros(())
    for i in range(0, len(left), block_size):
        for j in range(0, len(right), block_size):
            kernel = rbf_kernel(left[i:i+block_size], right[j:j+block_size], bandwidth)
            if exclude_block_diagonal:
                rows = torch.arange(i, i + len(kernel), device=left.device)
                columns = torch.arange(j, j + kernel.shape[1], device=left.device)
                kernel = kernel.masked_fill(rows[:, None] == columns[None, :], 0.)
            total = total + kernel.sum()
    return total


def shared_control_mmd(populations, observed, bandwidth, *, block_size=128, progress=None):
    """Equal-member mixture U estimate, excluding all same-control-position pairs.

    Each member is [control draws, genes] in the same control order. Different
    positions must be iid control/noise blocks, independent of observed iid rows;
    dependence across members within a position is allowed. Frozen member models
    are conditioned on. Whole-population correction violates these assumptions.
    Repeated values from independent replacement draws remain separate blocks.
    Precision/device follow the input tensors. Negative estimates are retained.
    """
    if not populations or observed.ndim != 2 or observed.shape[1] == 0:
        raise ValueError("Nonempty equal-size member populations and observed [rows, genes] required")
    first = populations[0]
    if first.ndim != 2 or first.shape[1] != observed.shape[1] or any(
            value.shape != first.shape or value.device != observed.device or value.dtype != observed.dtype
            for value in populations):
        raise ValueError("Members require matching shape, ordered genes, device and dtype")
    n, m, members = len(first), len(observed), len(populations)
    if min(n, m) < 2:
        raise ValueError("At least two control blocks and observed rows required")
    if not torch.isfinite(observed).all() or any(not torch.isfinite(value).all() for value in populations):
        raise ValueError("Finite populations required; invalid rows must not be dropped")
    options = {"bandwidth": bandwidth, "block_size": block_size}
    pp = observed.new_zeros(())
    for a_index, a in enumerate(populations):
        for b_index, b in enumerate(populations):
            if progress:
                progress("mixture kernel member pairs", members*a_index + b_index, members**2)
            pp = pp + kernel_sum(a, b, exclude_block_diagonal=True, **options)
    if progress:
        progress("mixture kernel member pairs", members**2, members**2)
    yy = kernel_sum(observed, observed, exclude_block_diagonal=True, **options)
    py = sum(kernel_sum(part, observed, **options) for part in populations)
    return pp / (members**2*n*(n-1)) + yy / (m*(m-1)) - 2*py / (members*n*m)


@dataclass(frozen=True)
class ProjectionResult:
    x: torch.Tensor
    y: torch.Tensor
    valid: torch.Tensor


def physical_projection(raw, mean, scale):
    """Full common panel only. Invalid rows remain flagged and NaN, never zeros."""
    if raw.ndim != 2 or mean.shape != raw.shape[1:] or scale.shape != mean.shape:
        raise ValueError("Projection coordinates require raw [batch,G] and frozen mean/scale [G].")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid frozen normalization coordinates.")
    clipped = (mean + scale * raw).clamp(0, math.log(10001))
    counts = clipped.expm1()
    mass = counts.sum(-1, keepdim=True)
    valid = torch.isfinite(raw).all(-1) & torch.isfinite(mass[:, 0]) & (mass[:, 0] > 0)
    log_expression = (10000 * counts / mass.clamp_min(torch.finfo(raw.dtype).tiny)).log1p()
    normalized = (log_expression - mean) / scale
    invalid = torch.full_like(raw, float("nan"))
    return ProjectionResult(torch.where(valid[:, None], normalized, invalid),
                            torch.where(valid[:, None], log_expression, invalid), valid)


def group_expression_error(generated, observed, bandwidth):
    """Condition-level residual objective; no subtraction of unpaired cell rows."""
    return rbf_mmd(generated, observed, bandwidth) + (generated.mean(0) - observed.mean(0)).square().mean()
