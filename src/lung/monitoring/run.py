"""Console progress, real scalar events, durable CSV/plots and run manifests."""

import csv
import importlib.metadata
import math
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from lung.engine.checkpoint import atomic_json
from lung.engine.config import ROOT, config_hash, implementation_identity
from .dashboard import launch_dashboard


class RunMonitor:
    def __init__(self, config, run_id=None, resume=False):
        self.config = config
        self.run_id = run_id or (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" +
                                 config["stage"] + "-" + uuid.uuid4().hex[:6])
        if Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a single safe directory name")
        self.experiment = ROOT / "experiments" / self.run_id
        self.logdir = ROOT / "logs" / self.run_id
        self.weights = ROOT / "checkpoints" / self.run_id
        if self.experiment.exists() and not resume:
            raise FileExistsError(f"Run already exists: {self.run_id}")
        for folder in (self.experiment, self.logdir, self.weights):
            folder.mkdir(parents=True, exist_ok=True)
        self.segment = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.writer = SummaryWriter(str(self.logdir / "events" / self.segment), flush_secs=10)
        self.console_file = (self.logdir / f"console-{self.segment}.log").open("a", encoding="utf-8")
        self.console = Console()
        self.csv_file = (self.experiment / "metrics.csv").open("a", encoding="utf-8", newline="")
        self.csv = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            self.csv.writerow(["run_id", "segment", "stage", "axis", "step", "metric", "value", "seconds"])
        self.started = time.monotonic()
        self.latest = {}
        self.preparation = None
        self._preparation_bar = None
        self._preparation_started = self.started
        self._preparation_last_write = float("-inf")
        self._preparation_written = None
        self._preparation_step = 0
        self.dashboard_process = None
        self.manifest = {"schema_version": 1, "run_id": self.run_id, "status": "running",
                         "task": config["task"], "synthetic": config.get("synthetic", False),
                         "config_hash": config_hash(config), "segment": self.segment,
                         "implementation_identity": implementation_identity(),
                         "experiment": str(self.experiment), "weights": str(self.weights),
                         "logs": str(self.logdir), "formal_training": not config.get("synthetic", False)}
        suffix = f"resume-{self.segment}" if resume else "resolved_config"
        (self.experiment / f"{suffix}.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        active_config = self.experiment / "active_config.yaml"
        temporary_config = active_config.with_suffix(".tmp")
        temporary_config.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        temporary_config.replace(active_config)
        packages = {}
        for name in ("torch", "numpy", "scipy", "pandas", "pyarrow", "rdkit", "tensorboard", "tqdm"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        atomic_json(self.experiment / f"environment-{self.segment}.json",
                    {"executable": sys.executable, "python": platform.python_version(),
                     "packages": packages, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        self.update_status("running")
        self._summary()
        if config.get("dashboard", True):
            self.dashboard_process, status = launch_dashboard(
                ROOT / "logs", self.logdir / "dashboard.json", config.get("open_browser", True))
            self.say(f"本地曲线：{status.get('url', status['status'])}")
            if status["status"] == "running":
                self.say("TensorBoard Settings 中开启 Reload data，可持续刷新曲线。")

    def _summary(self):
        table = Table(title="肺癌细胞响应研究 · 训练")
        table.add_column("项目")
        table.add_column("当前设置")
        for key, value in (("Run", self.run_id), ("阶段", self.config["stage"]),
                           ("Task", self.config["task"]), ("Epochs", self.config["epochs"]),
                           ("响应骨干", self.config.get("response", {}).get("model", {}).get("backbone", "gene_token")),
                           ("解释器", sys.executable), ("Device", self.config["device"]),
                           ("数据性质", "SYNTHETIC / 工程测试" if self.config.get("synthetic") else "公开实测数据 / 模型内模拟")):
            table.add_row(str(key), str(value))
        self.console.print(table)
        if self.config["task"] == "PROXY-A549-V1":
            self.say("奖励身份：状态代理；Tahoe → TIDYI 跨来源迁移尚未验证，不是真实凋亡率。")

    def say(self, message):
        self.console.print(message, markup=False)
        self.console_file.write(str(message) + "\n")
        self.console_file.flush()

    def progress(self, iterable=None, **kwargs):
        # PyCharm Run is commonly not a TTY. Do not silently suppress its progress.
        options = {"disable": not self.config.get("progress", True),
                   "dynamic_ncols": True, "mininterval": 1, **kwargs}
        return tqdm(iterable, **options)

    def preparation_progress(self, phase, current, total):
        """Report preparation separately from optimizer steps and epochs."""
        if not phase or current < 0 or total < 0 or current > total:
            raise ValueError("Invalid preparation phase or progress counts")
        now = time.monotonic()
        previous = self.preparation
        new_phase = (previous is None or previous["phase"] != phase or
                     previous["status"] != "running" or current < previous["current"] or
                     total != previous["total"])
        if new_phase:
            if previous is not None and previous is not self._preparation_written:
                self._write_preparation(now, force=True)
            self._close_preparation_bar()
            self._preparation_started = now
            self.say(f"训练前数据准备：{phase}，总计 {total}。")
            self._preparation_bar = self.progress(total=total, desc=f"Prepare {phase}")
        phase_index = (1 if previous is None else previous["phase_index"] + int(new_phase))
        complete = current == total
        self.preparation = {
            "status": "running", "phase": str(phase), "phase_index": phase_index,
            "phase_status": "complete" if complete else "running",
            "current": int(current), "total": int(total),
            "fraction": float(current / total) if total else None,
            "phase_seconds": (previous["phase_seconds"] if complete and not new_phase and
                              previous["phase_status"] == "complete" else
                              round(now - self._preparation_started, 3)),
            "elapsed_seconds": round(now - self.started, 3),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        if self._preparation_bar is not None:
            self._preparation_bar.update(current - self._preparation_bar.n)
            if complete:
                # Close now: a later phase must not repaint this 100% with its idle time.
                self._close_preparation_bar()
        newly_complete = complete and (new_phase or previous["phase_status"] != "complete")
        self._write_preparation(now, force=new_phase or newly_complete)

    def _close_preparation_bar(self):
        if self._preparation_bar is not None:
            self._preparation_bar.close()
            self._preparation_bar = None

    def _write_preparation(self, now, *, force=False):
        if self.preparation is None or (not force and now - self._preparation_last_write < 5):
            return
        snapshot = self.preparation
        # This axis counts preparation updates, never model epochs or optimizer updates.
        for name in ("current", "total", "fraction", "phase_seconds"):
            if snapshot[name] is not None:
                self.writer.add_scalar(f"preparation/{snapshot['phase']}/{name}",
                                       snapshot[name], self._preparation_step)
        self._preparation_step += 1
        self.writer.flush()
        self.manifest = {**self.manifest, "preparation": snapshot}
        self.update_status(self.manifest["status"])
        self._preparation_last_write = now
        self._preparation_written = snapshot

    def finish_preparation(self, status="complete"):
        """Close without inventing progress; interruption must pass its real status."""
        self._close_preparation_bar()
        if self.preparation is None or self.preparation["status"] != "running":
            return
        now = time.monotonic()
        phase_status = self.preparation["phase_status"]
        self.preparation = {
            **self.preparation, "status": status,
            "phase_status": (status if status != "complete" and phase_status != "complete"
                             else phase_status),
            "phase_seconds": (self.preparation["phase_seconds"] if phase_status == "complete"
                              else round(now - self._preparation_started, 3)),
            "elapsed_seconds": round(now - self.started, 3),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        self._write_preparation(now, force=True)

    def record(self, stage, step, metrics, axis="epoch"):
        clean = {}
        for name, value in metrics.items():
            if value is None:
                continue  # Unavailable is not zero.
            value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"NONFINITE_METRIC: {stage}/{name}={value}")
            tag = f"{stage}/{axis}/{name}"
            self.writer.add_scalar(tag, value, int(step))
            self.csv.writerow([self.run_id, self.segment, stage, axis, step, name, value,
                               round(time.monotonic() - self.started, 3)])
            clean[name] = value
        self.csv_file.flush()
        self.writer.flush()
        self.latest = {"stage": stage, "axis": axis, "step": step, **clean}
        self.update_status("running")
        if axis == "epoch":
            self.say(f"{stage} | epoch {step}/{self.config['epochs']} | " +
                     " | ".join(f"{key}={value:.6g}" for key, value in clean.items()))
            self.plot()

    def update_status(self, state, **fields):
        self.manifest = {**self.manifest, "status": state, "latest": self.latest,
                         "updated_at": datetime.now().astimezone().isoformat(), **fields}
        atomic_json(self.experiment / "run_manifest.json", self.manifest)

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        frame = pd.read_csv(self.experiment / "metrics.csv")
        frame = frame[frame.axis == "epoch"]
        if frame.empty:
            return
        frame.drop_duplicates(["stage", "step", "metric"], keep="last").pivot(
            index=["stage", "step"], columns="metric", values="value").to_csv(
                self.experiment / "results.csv", encoding="utf-8-sig")
        groups = list(frame.groupby("stage", sort=False))
        figure, axes = plt.subplots(len(groups), 1, figsize=(10, max(3, 3 * len(groups))), squeeze=False)
        for (stage, rows), ax in zip(groups, axes.flat):
            eligible = [name for name in rows.metric.unique()
                        if any(token in name for token in ("loss", "mmd", "mse", "return", "score", "f1", "kl"))][:8]
            for name in eligible:
                points = rows[rows.metric == name].drop_duplicates("step", keep="last")
                ax.plot(points.step, points.value, label=name)
            ax.set_title(stage)
            ax.set_xlabel("epoch")
            ax.grid(alpha=.2)
            if eligible:
                ax.legend(fontsize=8)
        figure.suptitle("SYNTHETIC ENGINEERING TEST" if self.config.get("synthetic") else self.run_id)
        figure.tight_layout()
        figure.savefig(self.experiment / "results.png", dpi=130)
        plt.close(figure)

    def close(self, status="complete", **fields):
        self.finish_preparation("complete" if status in {"complete", "prepared_only"} else status)
        self.writer.flush()
        self.writer.close()
        self.csv_file.close()
        self.console_file.close()
        self.update_status(status, **fields)
        # Keep the user's real training dashboard for inspection. Synthetic tests clean only theirs.
        if self.config.get("synthetic") and self.dashboard_process is not None:
            if self.dashboard_process.poll() is None:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=10)
