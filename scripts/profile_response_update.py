"""One bounded real-T optimizer update, plus one V condition; not a training run."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lung.models import ModelConfig, build_response_model
from lung.engine.checkpoint import seed_all, save_training
from lung.engine.config import load_config, config_hash, implementation_identity
from lung.engine.data import open_data
from lung.engine.response import train_condition, response_rngs, learning_rate, evaluate_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    cfg = config["response"]
    if cfg["model"]["backbone"] != "compact_full_gene":
        raise ValueError("This bounded probe requires the compact candidate")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.set_num_threads(4)
    seed_all(cfg["seed"])
    started = time.perf_counter()
    data = open_data(config, "response", prepare=False)
    load_seconds = time.perf_counter() - started
    keys = list(data.iter_response_epoch("T", cfg["seed"]))
    selected = keys[:cfg["conditions_per_update"]]
    shape = ModelConfig(**cfg["model"])
    model = build_response_model(shape, torch.as_tensor(data.neighbors, device=device)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    print("Real-data diagnostic: warmup; no research suite will be started", flush=True)
    warm_start = time.perf_counter()
    with torch.random.fork_rng(devices=[device.index or 0]):
        warm_rng, warm_mmd_rng, warm_generator = response_rngs(cfg["seed"] + 777, device)
        train_condition(model, data, selected[0], warm_rng, cfg, device, 1,
                        mmd_rng=warm_mmd_rng, mmd_generator=warm_generator)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    rng, mmd_rng, generator = response_rngs(cfg["seed"], device)
    updates = math.ceil(len(keys) / cfg["conditions_per_update"])
    lr = learning_rate(0, config["epochs"] * updates,
                       min(cfg["warmup_epochs"], config["epochs"]) * updates, cfg["lr"], cfg["min_lr"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    torch.cuda.reset_peak_memory_stats()
    rows = []
    update_start = time.perf_counter()
    for index, key in enumerate(selected):
        condition_start = time.perf_counter()
        values = train_condition(model, data, key, rng, cfg, device, len(selected),
                                 mmd_rng=mmd_rng, mmd_generator=generator)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - condition_start
        rows.append({"condition": str(key), "seconds": seconds, "metrics": values})
        print(f"Condition {index + 1}/{len(selected)}: {seconds:.3f}s, loss={values['loss']:.6f}", flush=True)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError("The probe did not produce a finite nonzero gradient")
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_start
    peak = torch.cuda.max_memory_allocated()
    args.output.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    save_training(args.output / "diagnostic-update.pt", model, optimizer, epoch=0,
                  identity=config_hash({"data": data.identity, "response": cfg}),
                  metadata={"stage": "diagnostic_real_T_only", "model_config": asdict(shape)},
                  counters={"optimizer_step": 1}, extra={"not_a_research_checkpoint": True})
    save_seconds = time.perf_counter() - save_start

    class OneValidationCondition:
        def condition_ids(self, role):
            if role != "V":
                raise ValueError("This diagnostic only exposes a validation condition")
            return data.condition_ids("V")[:1]

        def __getattr__(self, name):
            return getattr(data, name)

    validation_start = time.perf_counter()
    validation, _ = evaluate_response(model, OneValidationCondition(), cfg, device)
    torch.cuda.synchronize()
    report = {"scope": "bounded_real_T_update_and_one_V_condition_not_model_quality_evaluation",
              "source_identity": implementation_identity(), "data_identity": data.identity,
              "device": torch.cuda.get_device_name(), "dtype": str(next(model.parameters()).dtype),
              "response_config": cfg, "parameters": sum(p.numel() for p in model.parameters()),
              "data_open_seconds": load_seconds, "warmup_seconds": warm_seconds,
              "conditions_per_update": len(selected), "conditions": rows,
              "update_seconds": update_seconds, "checkpoint_seconds": save_seconds,
              "validation_condition_seconds": time.perf_counter() - validation_start,
              "validation_diagnostic": validation, "gradient_norm_before_clip": float(norm),
              "peak_allocated_bytes": peak, "total_conditions_in_training_epoch": len(keys),
              "learning_rate": lr, "formal_training_started": False}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"conditions", "response_config"}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
