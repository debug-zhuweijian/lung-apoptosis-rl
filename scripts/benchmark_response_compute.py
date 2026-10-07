"""Bounded, synthetic-input CUDA timings; never launches a research training run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("RESPONSE_BENCH_SOURCE", str(ROOT / "src")))
from lung.models import ModelConfig, ResponseModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", choices=("velocity", "cfm", "ode", "mmd"), default="velocity")
    parser.add_argument("--backbone", choices=("gene_token", "compact_full_gene"), default="gene_token")
    parser.add_argument("--genes", type=int, default=32461)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-values", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if min(args.genes, args.batch, args.steps, args.repeats) < 1 or args.warmup < 1:
        raise ValueError("Use positive dimensions, repeats, and at least one warmup")
    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    torch.cuda.manual_seed_all(20260927)
    device = torch.device("cuda:0")
    shape_args = dict(n_genes=args.genes, query_chunk=4096,
                      checkpoint_blocks=args.backbone == "gene_token",
                      checkpoint_attention=args.backbone == "gene_token")
    if args.backbone != "gene_token":
        shape_args["backbone"] = args.backbone
    shape = ModelConfig(**shape_args)
    asset = ROOT / "08-datasets/LungCancer_A549_Integration/training-v1/neighbors.npz"
    if args.genes == 32461:
        with np.load(asset) as stored:
            neighborhood = torch.as_tensor(stored[stored.files[0]], device=device, dtype=torch.long)
        graph_kind = "frozen_real_gene_graph"
    else:
        genes = torch.arange(args.genes, device=device)
        neighborhood = (genes[:, None] + torch.arange(min(32, args.genes), device=device)) % args.genes
        graph_kind = "synthetic_ring"
    if args.backbone == "gene_token":
        model = ResponseModel(shape, neighborhood).to(device)
    else:
        from lung.models import build_response_model
        model = build_response_model(shape, neighborhood).to(device)
    # Exercise nonzero gates, including their derivatives, rather than a zero-gate shortcut.
    with torch.no_grad():
        for block in model.velocity_model.blocks:
            for modulation in getattr(block, "modulations", ()):
                modulation[-1].weight.normal_(0., .005)
                modulation[-1].bias.normal_(0., .01)
    reference = torch.randn(args.batch, args.genes, device=device)
    target = torch.randn_like(reference)
    noise = torch.randn_like(reference)
    tau = torch.linspace(.25, .75, args.batch, device=device)
    inputs = dict(fingerprint=torch.rand(args.batch, 2048, device=device),
                  dose=torch.ones(args.batch, device=device),
                  duration=torch.full((args.batch,), 24., device=device),
                  background=torch.zeros(args.batch, dtype=torch.long, device=device),
                  source=torch.zeros(args.batch, dtype=torch.long, device=device),
                  protocol=torch.zeros(args.batch, dtype=torch.long, device=device),
                  elapsed=torch.zeros(args.batch, device=device),
                  horizon=torch.full((args.batch,), 24., device=device))
    with torch.no_grad():
        tokens, _ = model.encode_cells(reference)
        condition = model.encode_condition(**inputs)

    def workload():
        if args.workload == "velocity":
            with torch.no_grad():
                return model._velocity_with_valid_tau(noise, tau, tokens, condition)
        if args.workload == "ode":
            with torch.no_grad():
                return model.sample(reference, model.encode_condition(**inputs), noise=noise,
                                    steps=args.steps, checkpoint=False)
        model.zero_grad(set_to_none=True)
        if args.workload == "cfm":
            loss = model.cfm_loss(reference, target, model.encode_condition(**inputs), noise=noise, tau=tau)
            loss.backward()
        else:
            from lung.engine.response import exact_mmd_backward
            loss = exact_mmd_backward(model, reference, target,
                lambda n: model.encode_condition(**{key: value[:n] for key, value in inputs.items()}),
                bandwidth=1., steps=args.steps, noise=noise, microbatch=args.batch)
        return loss

    print(f"Benchmark {args.workload}: genes={args.genes}, batch={args.batch}, steps={args.steps}", flush=True)
    for _ in range(args.warmup):
        workload()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
    wall = time.perf_counter()
    result = None
    for i in range(args.repeats):
        starts[i].record()
        result = workload()
        ends[i].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall
    elapsed = [start.elapsed_time(end) / 1000 for start, end in zip(starts, ends)]
    report = {"input_kind": "synthetic_expression_not_training", "graph": graph_kind,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "dtype": str(reference.dtype), "matmul_precision": torch.get_float32_matmul_precision(),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "model": asdict(shape),
              "workload": args.workload, "batch": args.batch, "steps": args.steps,
              "warmup": args.warmup, "repeats": args.repeats, "cuda_seconds": elapsed,
              "median_seconds": statistics.median(elapsed), "wall_seconds": wall,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_values:
        values = {"result": result.detach().cpu() if torch.is_tensor(result) else result,
                  "grads": {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}}
        torch.save(values, args.output.with_suffix(".pt"))
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            workload()
            torch.cuda.synchronize()
        events = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
        report["top_operators"] = [{"op": e.key, "calls": e.count,
            "self_device_ms": e.self_device_time_total / 1000,
            "self_cpu_ms": e.self_cpu_time_total / 1000} for e in events[:20]]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
