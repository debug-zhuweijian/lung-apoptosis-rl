"""Manual, bounded GPU forward/backward check; no optimizer or formal training."""

from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
from lung.data import TrainingData
from lung.data.preprocess import log_normalize
from lung.models import ModelConfig, ResponseModel
from lung.engine.checkpoint import atomic_json, seed_all
from lung.engine.data import EngineData
from lung.engine.config import load_config
from lung.engine.response import tensor, condition_kwargs


def main():
    config = load_config()
    shape = ModelConfig(**config["response"]["model"])
    data = EngineData(config, require_preprocessing=False)
    reader = data.reader
    key = data.condition_ids("T")[0]
    rng = np.random.default_rng(20260924)
    control_rows = reader._control_rows(key, 1, rng, "fit")
    target_rows = reader.cell_metadata(key, "fit").iloc[:1]
    x, ok_x = log_normalize(reader.rows.read(control_rows))
    y, ok_y = log_normalize(reader.rows.read(target_rows))
    assert ok_x.all() and ok_y.all()
    # Training-fitted graph/norm are not ready. This is a full-dimensional runtime
    # test with real counts and an explicitly synthetic ring graph, not a fit run.
    neighbors = (np.arange(shape.n_genes)[:, None] + np.arange(33)[None]) % shape.n_genes
    seed_all(20260924)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model = ResponseModel(shape, torch.from_numpy(neighbors)).to("cuda:0")
    metadata = data.action_features(key)
    condition = model.encode_condition(**condition_kwargs(metadata, 1, "cuda:0"))
    loss = model.cfm_loss(tensor(x.toarray(), "cuda:0"), tensor(y.toarray(), "cuda:0"), condition)
    loss.backward()
    torch.cuda.synchronize()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    report = {"status": "passed", "check": "full_panel_real_count_shape_forward_backward_only",
              "genes": shape.n_genes, "width": shape.d_model, "encoder_layers": shape.encoder_layers,
              "pad_layers": shape.pad_layers, "device": torch.cuda.get_device_name(0),
              "finite_loss": bool(torch.isfinite(loss)),
              "finite_gradients": all(bool(torch.isfinite(g).all()) for g in gradients),
              "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
              "elapsed_seconds": time.monotonic()-start, "optimizer_steps": 0,
              "formal_training": False, "graph": "synthetic_ring_33_neighbors",
              "normalization": "log1p_common_library_10000_only_no_fitted_T_stats"}
    if not report["finite_loss"] or not report["finite_gradients"]:
        raise FloatingPointError("Full-panel runtime gradient failure")
    if "--ode" in sys.argv:
        model.zero_grad(set_to_none=True)
        condition = model.encode_condition(**condition_kwargs(metadata, 1, "cuda:0"))
        generated = model.sample(tensor(x.toarray(), "cuda:0"), condition, steps=32, checkpoint=True)
        generated.square().mean().backward()
        torch.cuda.synchronize()
        report.update(ode_steps=32, full_ode_backward=True,
                      ode_finite_gradients=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
                      peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                      elapsed_seconds=time.monotonic()-start)
    output = ROOT / "logs/environment_checks/implementation/full_panel_runtime.json"
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
