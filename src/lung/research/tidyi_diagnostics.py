"""Recompute checkpoint sections 6.2--6.4 from frozen historical predictions."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

LABELS = ("PI-APC-", "PI-APC+", "PI+APC+")
EXPRESSION_MODELS = ("expression_uncalibrated", "expression_sigmoid_calibrated")
MODELS = ("prior", "well_position_only", "qc_only", *EXPRESSION_MODELS)


def validate_gene_axis(frame, expected_count=32461):
    if (len(frame) != expected_count or frame.ensembl_id.duplicated().any()
            or not np.array_equal(frame.common_gene_index, np.arange(expected_count))):
        raise ValueError("Common gene axis size/order/identity mismatch")


def require_gene_order(actual, expected):
    if not np.array_equal(actual, expected):
        raise ValueError("Gene identity/order mismatch")


def normalize_counts(counts):
    """Full common-axis library sum, float64 log1p(10000 * counts / sum)."""
    from scipy import sparse

    if sparse.issparse(counts):
        result = counts.astype(np.float64).tocsr(copy=True)
        if not np.isfinite(result.data).all() or (result.data < 0).any():
            raise ValueError("Invalid counts")
        totals = np.asarray(result.sum(axis=1)).ravel()
        if (totals <= 0).any():
            raise ValueError("Zero common-gene library")
        result = result.multiply((10000 / totals)[:, None]).tocsr()
        result.data = np.log1p(result.data)
        return result
    result = np.asarray(counts, dtype=np.float64)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Invalid counts")
    totals = result.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Zero common-gene library")
    return np.log1p(10000 * result / totals[:, None])


def ordered_probabilities(frame):
    """CSV header order is not the biological class order."""
    if not frame.true_state.isin(LABELS).all():
        raise ValueError("Unknown state label")
    probabilities = frame[["p_" + label for label in LABELS]].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Invalid class probabilities")
    return probabilities


def binary_metrics(frame):
    probability = ordered_probabilities(frame)[:, 1:].sum(axis=1)
    positive = frame.true_state.ne(LABELS[0]).to_numpy()
    n_positive, n_negative = positive.sum(), (~positive).sum()
    if not n_positive or not n_negative:
        raise ValueError("Binary AUROC requires both labels")
    auc = (rankdata(probability)[positive].sum() - n_positive * (n_positive + 1) / 2)
    auc /= n_positive * n_negative
    predicted = np.argmax(ordered_probabilities(frame), axis=1)
    return {"n": len(frame), "positive_n": int(n_positive), "auroc": float(auc),
            "brier": float(np.mean((probability - positive) ** 2)),
            "mean_positive_score": float(probability.mean()),
            "observed_positive_fraction": float(positive.mean()),
            "mean_score_bias": float(probability.mean() - positive.mean()),
            "class_recall": {label: float(np.mean(predicted[frame.true_state.eq(label)] == i))
                             for i, label in enumerate(LABELS)}}


def constrained_moment(matrix, target):
    """Exact small simplex least squares by enumeration of all nonempty faces."""
    matrix, target = np.asarray(matrix, float), np.asarray(target, float)
    if (matrix.ndim != 2 or target.shape != (matrix.shape[0],)
            or not np.isfinite(matrix).all() or not np.isfinite(target).all()):
        raise ValueError("Invalid moment inputs")
    width = matrix.shape[1]
    best, error = None, np.inf
    for size in range(1, width + 1):
        for face in combinations(range(width), size):
            columns = matrix[:, face]
            kkt = np.block([[columns.T @ columns, np.ones((size, 1))],
                            [np.ones((1, size)), np.zeros((1, 1))]])
            solution = np.linalg.lstsq(kkt, np.r_[columns.T @ target, 1.], rcond=None)[0][:size]
            if (solution < -1e-10).any():
                continue
            solution = np.maximum(solution, 0)
            solution /= solution.sum()
            candidate = np.zeros(width)
            candidate[list(face)] = solution
            residual = float(np.sum((matrix @ candidate - target) ** 2))
            if residual < error:
                best, error = candidate, residual
    if best is None:
        raise ValueError("No feasible simplex solution")
    return best


def mixtures(frame, metadata):
    probabilities = ordered_probabilities(frame)
    wells = metadata.set_index("cell_id").loc[frame.cell_id, "Well"]
    columns = wells.str.extract(r"^[A-Za-z]+(\d+)$", expand=False)
    if columns.isna().any():
        raise ValueError("Malformed well coordinate")
    even = columns.astype(int).to_numpy() % 2 == 0
    labels = frame.true_state.to_numpy()
    matrices = []
    for pool in (even, ~even):
        if any(not (pool & (labels == label)).any() for label in LABELS):
            raise ValueError("Every class must occur in each disjoint pool")
        matrices.append(np.column_stack([probabilities[pool & (labels == label)].mean(axis=0)
                                         for label in LABELS]))
    estimate, target = matrices
    prior_positive = float(np.mean(labels[even] != LABELS[0]))
    binary_a = np.column_stack([probabilities[even & (labels == LABELS[0])].mean(axis=0),
                               probabilities[even & (labels != LABELS[0])].mean(axis=0)])
    binary_a = np.vstack([binary_a[0], binary_a[1:].sum(axis=0)])
    rows = []
    for first in range(11):
        for second in range(11 - first):
            proportions = np.array([first, second, 10 - first - second]) / 10
            moment = target @ proportions
            recovered = constrained_moment(estimate, moment)
            binary_recovered = constrained_moment(binary_a, [moment[0], moment[1:].sum()])
            rows.append({"pi_negative": proportions[0], "pi_middle": proportions[1],
                         "pi_last": proportions[2], "true_positive": float(proportions[1:].sum()),
                         "direct_positive": float(moment[1:].sum()),
                         "moment_positive": float(recovered[1:].sum()),
                         "binary_moment_positive": float(binary_recovered[1]),
                         "fixed_prior_positive": prior_positive,
                         **{f"recovered_{i}": float(v) for i, v in enumerate(recovered)}})
    table = pd.DataFrame(rows)
    summary = {"estimation_n": int(even.sum()), "mixture_pool_n": int((~even).sum()),
               "mixtures": len(table), "condition_number": float(np.linalg.cond(estimate)),
               "A_rows_predicted_columns_true": estimate.tolist(),
               "target_class_conditional_outputs": target.tolist(),
               "disjoint_cell_pools": True}
    for name in ("direct", "moment", "binary_moment", "fixed_prior"):
        summary[name + "_positive_mae"] = float(abs(table[name + "_positive"] - table.true_positive).mean())
    return summary, table


def mean_decomposition(expression, plates, labels):
    if not set(labels).issubset(LABELS) or set(plates) != {1, 8}:
        raise ValueError("Unexpected plate or label")
    means, priors = {}, {}
    for plate in (1, 8):
        mask = plates == plate
        if any(not (mask & (labels == label)).any() for label in LABELS):
            raise ValueError("Missing class in plate")
        means[plate] = np.stack([expression[mask & (labels == label)].mean(axis=0) for label in LABELS])
        priors[plate] = np.array([np.mean(labels[mask] == label) for label in LABELS])
    total = expression[plates == 1].mean(axis=0) - expression[plates == 8].mean(axis=0)
    composition = (priors[1] - priors[8]) @ means[8]
    residual = priors[1] @ (means[1] - means[8])
    summary = {name + "_rms": float(np.sqrt(np.mean(vector ** 2)))
               for name, vector in [("total", total), ("composition", composition), ("residual", residual)]}
    summary.update(max_identity_residual=float(np.max(abs(total - composition - residual))),
                   source_plate=8, target_plate=1, source_priors=priors[8].tolist(),
                   target_priors=priors[1].tolist())
    return summary, pd.DataFrame({"total": total, "composition": composition, "residual": residual})


def run_tidyi(root, tracker):
    base = root / "08-datasets/TIDYI/E-MTAB-16354"
    gene_map = pd.read_parquet(tracker(root / "08-datasets/LungCancer_A549_Integration/prepared/v1/gene_map.parquet"))
    validate_gene_axis(gene_map)
    metadata = pd.read_csv(tracker(base / "audit_sources/human_ss3_A549_metadata.csv"))
    predictions = pd.read_csv(tracker(base / "A549_cross_plate_out_of_sample_predictions.csv"))
    with np.load(tracker(base / "audit_sources/A549_author_collection_counts.npz"), allow_pickle=False) as data:
        ids, genes = data["cell_ids"], data["gene_ids"]
        if metadata.cell_id.duplicated().any() or len(set(ids)) != len(ids) or set(ids) != set(metadata.cell_id):
            raise ValueError("TIDYI cell IDs mismatch")
        aligned = metadata.set_index("cell_id").loc[ids]
        if (not np.array_equal(aligned.Plate, data["plates"])
                or not np.array_equal(aligned.cell_state, data["states"])):
            raise ValueError("TIDYI metadata mismatch")
        require_gene_order(genes[gene_map.tidyi_column], gene_map.ensembl_id)
        expression = normalize_counts(data["expression"][:, gene_map.tidyi_column])
        decomposition, per_gene = mean_decomposition(expression, data["plates"], data["states"])
    per_gene.insert(0, "ensembl_id", gene_map.ensembl_id)
    tables, metrics, mix_summaries = {"tidyi_gene_decomposition": per_gene}, [], []
    for direction, train, test in [("primary", 8, 1), ("reverse", 1, 8)]:
        # Historical reverse direction uses its own frozen classifier.
        direction_values = predictions.loc[predictions.test_plate.eq(test), "direction"].unique()
        if len(direction_values) != 1:
            raise ValueError("Ambiguous historical direction")
        for model in MODELS:
            frame = predictions[predictions.test_plate.eq(test) & predictions.model.eq(model)].copy()
            expected = metadata[metadata.Plate.eq(test)]
            if (frame.cell_id.duplicated().any() or set(frame.cell_id) != set(expected.cell_id)
                    or not frame.train_plate.eq(train).all()
                    or not np.array_equal(frame.true_state, metadata.set_index("cell_id").loc[frame.cell_id, "cell_state"])):
                raise ValueError("Historical prediction coverage, label or plate mismatch")
            metrics.append({"direction": direction, "model": model, **binary_metrics(frame)})
            if model in EXPRESSION_MODELS:
                summary, table = mixtures(frame, metadata)
                mix_summaries.append({"direction": direction, "model": model, **summary})
                tables[f"tidyi_mixtures_{direction}_{model}"] = table
    tables["tidyi_binary_metrics"] = pd.json_normalize(metrics)
    return {"binary_metrics": metrics, "mean_decomposition": decomposition, "mixture_diagnostics": mix_summaries,
            "prediction_provenance": "reevaluation of five frozen historical models; not new neural-head results or new fitting",
            "binary_metric_models": list(MODELS), "mixture_models": list(EXPRESSION_MODELS),
            "gene_count": len(gene_map), "cell_count": len(metadata), "label_order": list(LABELS),
            "positive_definition": "original last two sorted-cell states; not natural apoptosis prevalence",
            "mixture_definition": "66 exact empirical mixtures; step 0.1; even well columns estimate A, odd columns target",
            "normalization": "float64 log1p(10000 * counts / full_common_gene_sum)",
            "biological_replicates": "two plates; cells and 66 reused empirical mixtures are not independent biological replicates",
            "exploratory": True, "no_model_fit": True}, tables
