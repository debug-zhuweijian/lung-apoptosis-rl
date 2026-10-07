"""Fixed exploratory source-internal classifier of original PI/APC sorting labels.

This is not an apoptosis-fraction predictor, cross-source validation, or a
replication of the TIDYI paper. All expression selection/calibration is nested.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08-datasets" / "TIDYI" / "E-MTAB-16354"
TAHOE = ROOT / "08-datasets" / "Tahoe-100M" / "2dc57900b7981cfcf5e211527169a0b006546a95" / "metadata" / "gene_metadata.parquet"
CLASSES = np.array(["PI+APC+", "PI-APC+", "PI-APC-"])
SEED = 20260918


class TrainOnlyExpression(BaseEstimator, TransformerMixin):
    """Cell-local library normalization; prevalence/variance use training rows only."""

    def __init__(self, n_genes=500, min_cells=5):
        self.n_genes = n_genes
        self.min_cells = min_cells

    @staticmethod
    def normalize(x):
        totals = x.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("Empty cells cannot be normalized")
        return np.log1p(x * (10000.0 / totals))

    def fit(self, x, y=None):
        z = self.normalize(x)
        valid = (x > 0).sum(axis=0) >= self.min_cells
        variance = z.var(axis=0)
        indices = np.flatnonzero(valid & (variance > 0))
        self.selected_ = indices[np.argsort(-variance[indices], kind="stable")[:self.n_genes]]
        if len(self.selected_) == 0:
            raise ValueError("No genes pass training-fold filtering")
        return self

    def transform(self, x):
        return self.normalize(x)[:, self.selected_]


def expression_model():
    return Pipeline([
        ("expression", TrainOnlyExpression()),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000, random_state=SEED)),
    ])


def summarize(y, probability):
    encoded = np.column_stack([y == cls for cls in CLASSES]).astype(float)
    predicted = CLASSES[np.argmax(probability, axis=1)]
    confidence = probability.max(axis=1)
    rows = []
    ece = 0.0
    for bin_id in range(10):
        mask = (confidence >= bin_id / 10) & ((confidence < (bin_id + 1) / 10) if bin_id < 9 else (confidence <= 1))
        if mask.any():
            conf = float(confidence[mask].mean())
            acc = float((predicted[mask] == y[mask]).mean())
            ece += mask.mean() * abs(conf - acc)
            rows.append({"bin_left": bin_id / 10, "bin_right": (bin_id + 1) / 10, "n": int(mask.sum()), "mean_confidence": conf, "accuracy": acc})
    metrics = {
        "n": len(y), "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
        "log_loss": log_loss(y, probability, labels=CLASSES),
        "multiclass_brier_sum": float(np.mean(np.sum((probability - encoded) ** 2, axis=1))),
        "macro_ovr_roc_auc": roc_auc_score(encoded, probability, average="macro"),
        "macro_ovr_average_precision": average_precision_score(encoded, probability, average="macro"),
        "top_label_ece_10_equal_width_bins": float(ece),
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(y, predicted, labels=CLASSES).tolist(),
    }
    return metrics, rows


def main():
    warnings.filterwarnings("error", category=ConvergenceWarning)
    bundle = np.load(OUT / "A549_expression_audited.npz", allow_pickle=False)
    meta = pd.read_csv(OUT / "SS3_Human_A549_metadata.csv", index_col=0).loc[bundle["cell_ids"]]
    tahoe_genes = set(pd.read_parquet(TAHOE)["ensembl_id"].astype(str))
    common = np.array([gene in tahoe_genes for gene in bundle["gene_ids"]])
    assert common.sum() == 32461, "Common dictionary changed; inspect before proceeding"
    eligible = bundle["eligible"]
    x = bundle["expression"][eligible][:, common].astype(np.float64)
    meta = meta.loc[eligible].copy()
    y = meta["cell_state"].to_numpy(dtype=str)
    groups = meta["Well"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    outer = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(x, y, groups))
    protocol = {
        "frozen_before_first_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "source": "TIDYI E-MTAB-16354 A549 only; all Plate=8",
        "classes_in_order": CLASSES.tolist(), "seed": SEED,
        "common_dictionary_genes": int(common.sum()), "cells": len(y),
        "outer": "4-fold StratifiedGroupKFold grouping well COLUMN; disjoint columns, same plate",
        "inner": "3-fold StratifiedGroupKFold on each outer training partition; sigmoid CalibratedClassifierCV wraps entire pipeline; ensemble=True",
        "expression": "per-cell log1p(10000*counts/common-panel-total), >=5 training cells detected, top500 training variance, train-only StandardScaler, multinomial LogisticRegression C=0.1",
        "hyperparameter_selection": "none; all parameters fixed before evaluating held-out data",
        "comparators": ["training-fold class prior", "uncalibrated fixed expression pipeline", "calibrated expression pipeline", "position-only max-depth3 tree negative control", "QC-only logtotal/logdetected/mitochondrialpercentage logistic diagnostic"],
        "normalization_panel": "32461 common IDs; identical panel defines both numerator and library total",
        "limitations": ["one plate; column grouping cannot remove state/column-range confounding", "PI/APC raw sorting label names not relabelled as verified BOBO3 states", "no cross-source, cross-drug, across-plate or real apoptosis-fraction validation"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "source_sha256": json.loads((OUT / "download_manifest.json").read_text())["SS3_Human_A549_counts.csv"]["sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    protocol_path = OUT / "A549_baseline_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert previous["script_sha256"] == protocol["script_sha256"], "Protocol/script changed after evaluation; use an explicitly versioned new experiment"
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    names = ["prior", "expression_uncalibrated", "expression_sigmoid_calibrated", "well_position_only", "qc_only"]
    predictions = {name: np.full((len(y), len(CLASSES)), np.nan) for name in names}
    fold_ids = np.full(len(y), -1)
    fold_records = []
    qc = np.column_stack([np.log1p(x.sum(axis=1)), np.log1p((x > 0).sum(axis=1)), meta["pct_counts_mt"].to_numpy(dtype=float)])
    assert np.isfinite(qc).all()
    with threadpool_limits(limits=2):
        for fold, (train, test) in enumerate(outer):
            assert not set(groups[train]) & set(groups[test])
            assert set(y[train]) == set(CLASSES) and set(y[test]) == set(CLASSES)
            inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold).split(x[train], y[train], groups[train]))
            for fit, cal in inner:
                assert not set(groups[train][fit]) & set(groups[train][cal])
                assert set(y[train][fit]) == set(CLASSES) and set(y[train][cal]) == set(CLASSES)
            models = {
                "prior": (DummyClassifier(strategy="prior"), x),
                "expression_uncalibrated": (expression_model(), x),
                "expression_sigmoid_calibrated": (CalibratedClassifierCV(expression_model(), method="sigmoid", cv=inner, ensemble=True, n_jobs=1), x),
                "well_position_only": (DecisionTreeClassifier(max_depth=3, random_state=SEED), groups[:, None]),
                "qc_only": (Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=0.1, max_iter=3000))]), qc),
            }
            for name, (model, features) in models.items():
                model.fit(features[train], y[train])
                assert np.array_equal(model.classes_, CLASSES)
                predictions[name][test] = model.predict_proba(features[test])
            fold_ids[test] = fold
            fold_records.append({"outer_fold": fold, "n_train": len(train), "n_test": len(test), "train_columns": sorted(set(groups[train].tolist())), "test_columns": sorted(set(groups[test].tolist())), "test_states": pd.Series(y[test]).value_counts().to_dict(), "inner_validation_columns": [sorted(set(groups[train][cal].tolist())) for _, cal in inner]})
            print(f"Completed outer fold {fold}: {len(train)} train / {len(test)} test", flush=True)
    results = {"classes_in_order": CLASSES.tolist(), "protocol": protocol, "folds": fold_records, "metrics": {}}
    all_bins = []
    table = pd.DataFrame({"cell_id": meta.index, "state_original": y, "well_column": groups, "outer_fold": fold_ids})
    for name, probability in predictions.items():
        assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
        metrics, bins = summarize(y, probability)
        results["metrics"][name] = metrics
        all_bins.extend([{"model": name, **row} for row in bins])
        for j, cls in enumerate(CLASSES):
            table[f"{name}__p_{cls}"] = probability[:, j]
    table.to_csv(OUT / "A549_baseline_out_of_fold_predictions.csv", index=False)
    pd.DataFrame(all_bins).to_csv(OUT / "A549_baseline_calibration_bins.csv", index=False)
    (OUT / "A549_baseline_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
