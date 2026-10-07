"""Fixed plate-8 -> unseen plate-1 diagnostic; reverse direction is sensitivity only."""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from tidyi_a549_baseline import CLASSES, OUT, SEED, TAHOE, expression_model, summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    warnings.filterwarnings("error", category=ConvergenceWarning)
    source = OUT / "audit_sources"
    data = np.load(source / "A549_author_collection_counts.npz", allow_pickle=False)
    metadata = pd.read_csv(source / "human_ss3_A549_metadata.csv", index_col=0).loc[data["cell_ids"]]
    annotations = pd.read_csv(source / "A549_author_gene_annotations.csv").set_index("ensembl_id").loc[data["gene_ids"]]
    source_audit = json.loads((source / "human_ss3_metadata_audit.json").read_text())
    assert source_audit["old_count_max_abs_difference"] == 0
    assert source_audit["A549_common_state_mismatches"] == 0
    assert source_audit["A549_ids_shared_with_official_CSV"] == 368
    assert len(set(data["cell_ids"])) == len(data["cell_ids"]) == 743
    tahoe = set(pd.read_parquet(TAHOE)["ensembl_id"].astype(str))
    panel = np.array([gene in tahoe for gene in data["gene_ids"]])
    assert panel.sum() == 32461
    x = data["expression"][:, panel].astype(np.float64)
    y = data["states"]
    plates = data["plates"]
    groups = metadata["Well"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    source_mt = annotations["source_mt"].to_numpy(dtype=bool)
    formula_mt = annotations["gene_symbol"].astype(str).str.upper().str.startswith("MT-").to_numpy()
    assert np.array_equal(source_mt, formula_mt), "Mito annotation differs from symbol rule"
    mt = source_mt[panel]
    totals = x.sum(axis=1)
    assert (totals > 0).all()
    qc = np.column_stack([np.log1p(totals), np.log1p((x > 0).sum(axis=1)), 100 * x[:, mt].sum(axis=1) / totals])
    protocol = {
        "frozen_before_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "primary_direction": "plate8_train_368_to_plate1_test_375",
        "reverse_direction": "plate1_train_375_to_plate8_test_368; sensitivity analysis, not second independent validation",
        "seed": SEED, "classes": CLASSES.tolist(), "panel_size": int(panel.sum()),
        "expression_model": "Same as frozen source baseline: log1p10000, >=5 train-cell detection, top500 train variance, train StandardScaler, LR L2 C=0.1",
        "calibration": "3-fold StratifiedGroupKFold within TRAIN PLATE only, grouped by well column; full pipeline inside sigmoid CalibratedClassifierCV ensemble=True",
        "qc_only": "All recomputed from raw counts on common32461 panel: log1p total, log1p detected genes, mitochondrial percent; source var.mt checked equal to gene_symbol startswith MT-",
        "mito_gene_count_full": int(source_mt.sum()), "mito_gene_count_panel": int(mt.sum()),
        "comparators": ["prior", "expression_uncalibrated", "expression_sigmoid_calibrated", "well_position_only", "qc_only"],
        "hyperparameter_search": "none; neither direction used for tuning or feature choice",
        "limitations": ["Same study and shared plate-position sorting design", "PI/APC source labels, no relabelling", "No Tahoe cross-source or drug-apoptosis outcome validation"],
        "cross_plate_script_sha256": sha(Path(__file__)),
        "reused_baseline_script_sha256": sha(Path(__file__).with_name("tidyi_a549_baseline.py")),
        "source_manifest": json.loads((source / "human_ss3_download_manifest.json").read_text()),
    }
    protocol_file = OUT / "A549_cross_plate_protocol.json"
    if protocol_file.exists():
        previous = json.loads(protocol_file.read_text())
        assert previous["cross_plate_script_sha256"] == protocol["cross_plate_script_sha256"]
        assert previous["reused_baseline_script_sha256"] == protocol["reused_baseline_script_sha256"]
    else:
        protocol_file.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    results = {"protocol": protocol, "directions": {}}
    predictions = []
    bins = []
    with threadpool_limits(limits=2):
        for train_plate, test_plate, designation in [(8, 1, "primary"), (1, 8, "reverse_sensitivity")]:
            train = np.flatnonzero(plates == train_plate)
            test = np.flatnonzero(plates == test_plate)
            assert not set(data["cell_ids"][train]) & set(data["cell_ids"][test])
            assert set(y[train]) == set(CLASSES) and set(y[test]) == set(CLASSES)
            inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(x[train], y[train], groups[train]))
            valid_inner = all(set(y[train][fit]) == set(CLASSES) and set(y[train][cal]) == set(CLASSES) and not set(groups[train][fit]) & set(groups[train][cal]) for fit, cal in inner)
            models = {
                "prior": (DummyClassifier(strategy="prior"), x),
                "expression_uncalibrated": (expression_model(), x),
                "well_position_only": (DecisionTreeClassifier(max_depth=3, random_state=SEED), groups[:, None]),
                "qc_only": (Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=0.1, max_iter=3000))]), qc),
            }
            if valid_inner:
                models["expression_sigmoid_calibrated"] = (CalibratedClassifierCV(expression_model(), method="sigmoid", cv=inner, ensemble=True, n_jobs=1), x)
            record = {"n_train": len(train), "n_test": len(test), "train_plate": train_plate, "test_plate": test_plate, "calibration_inner_all_classes_present": valid_inner, "calibration_skipped_reason": None if valid_inner else "Fixed inner split lacked a class", "inner_validation_columns": [sorted(set(groups[train][cal].tolist())) for _, cal in inner], "metrics": {}}
            for name, (model, features) in models.items():
                model.fit(features[train], y[train])
                assert np.array_equal(model.classes_, CLASSES)
                probability = model.predict_proba(features[test])
                assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
                metrics, calibration_bins = summarize(y[test], probability)
                recalls = recall_score(y[test], CLASSES[probability.argmax(axis=1)], labels=CLASSES, average=None)
                metrics["per_class_recall"] = dict(zip(CLASSES, recalls))
                record["metrics"][name] = metrics
                frame = pd.DataFrame({"direction": designation, "model": name, "cell_id": data["cell_ids"][test], "train_plate": train_plate, "test_plate": test_plate, "true_state": y[test]})
                for j, cls in enumerate(CLASSES):
                    frame[f"p_{cls}"] = probability[:, j]
                predictions.append(frame)
                bins.extend([{"direction": designation, "model": name, **row} for row in calibration_bins])
            results["directions"][designation] = record
            print(designation, json.dumps(record["metrics"], ensure_ascii=False), flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "A549_cross_plate_out_of_sample_predictions.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "A549_cross_plate_calibration_bins.csv", index=False)
    (OUT / "A549_cross_plate_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
