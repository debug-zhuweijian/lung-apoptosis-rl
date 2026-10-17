"""Console progress, scalar CSV metrics and curves written when a run ends."""

import csv
import math
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from lung.engine.config import ROOT


class RunMonitor:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id = run_id or (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" +
                                 config["stage"] + "-" + uuid.uuid4().hex[:6])
        if Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a single safe directory name")
        self.experiment = ROOT / "experiments" / self.run_id
        self.logdir = ROOT / "logs" / self.run_id
        self.weights = ROOT / "checkpoints" / self.run_id
        if any(folder.exists() for folder in (self.experiment, self.logdir, self.weights)):
            raise FileExistsError(f"Run already exists: {self.run_id}")
        for folder in (self.experiment, self.logdir, self.weights):
            folder.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.csv_file = (self.experiment / "metrics.csv").open("w", encoding="utf-8", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["stage", "axis", "step", "metric", "value", "seconds"])
        self.started = time.monotonic()
        (self.experiment / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self._summary()

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

    def progress(self, iterable=None, **kwargs):
        # PyCharm Run is commonly not a TTY. Do not silently suppress its progress.
        options = {"disable": not self.config.get("progress", True),
                   "dynamic_ncols": True, "mininterval": 1, **kwargs}
        return tqdm(iterable, **options)

    def record(self, stage, step, metrics, axis="epoch"):
        clean = {}
        for name, value in metrics.items():
            if value is None:
                continue  # Unavailable is not zero.
            value = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"NONFINITE_METRIC: {stage}/{name}={value}")
            self.csv.writerow([stage, axis, step, name, value,
                               round(time.monotonic() - self.started, 3)])
            clean[name] = value
        self.csv_file.flush()
        if axis == "epoch":
            self.say(f"{stage} | epoch {step}/{self.config['epochs']} | " +
                     " | ".join(f"{key}={value:.6g}" for key, value in clean.items()))

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        frame = pd.read_csv(self.experiment / "metrics.csv")
        frame = frame[frame.axis == "epoch"]
        if frame.empty:
            return
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
        if self.csv_file.closed:
            return
        self.csv_file.close()
        self.plot()
        details = " | ".join(f"{key}={value}" for key, value in fields.items())
        self.say(f"运行结束：{status}" + (f" | {details}" if details else ""))
