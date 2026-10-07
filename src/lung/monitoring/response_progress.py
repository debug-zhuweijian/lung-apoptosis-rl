"""Throttled host-side response progress; never synchronizes CUDA or estimates ETA."""

import json
import math
import time
from datetime import datetime
from pathlib import Path


class ResponseTrainingProgress:
    """Keep conditions, pairs, microbatches and ODE steps on separate axes.

    MMD generate/replay callbacks use zero-based cell offsets. ODE callbacks
    report steps reached by the host loop, not device-completion measurements.
    Backward start/done callbacks bracket the call, without an added device sync.
    The owner keeps the condition bar silent. Only one stage bar is rendered,
    using carriage returns rather than cursor-up sequences unsupported by IDEs.
    """

    def __init__(self, bar, *, state_path=None, enabled=True, interval=1.,
                 mmd_microbatch=1, clock=None, bar_factory=None):
        if not isinstance(mmd_microbatch, int) or mmd_microbatch < 1:
            raise ValueError("mmd_microbatch must be a positive integer")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        self.bar, self.enabled, self.interval = bar, enabled, interval
        self.bar_factory, self.stage_bar, self.stage = bar_factory, None, None
        self.state_path = Path(state_path) if state_path is not None else None
        self.microbatch = mmd_microbatch
        self.clock = clock or time.monotonic
        self.started = self.clock() if enabled else 0.
        self.condition_started = self.started
        self.last_write = -float("inf")
        self.group_bar, self.group_offset, self.closed = None, None, False
        self.state = {
            "schema_version": 1, "status": "running", "phase": "waiting",
            "condition_index": 0, "conditions_total": 0, "conditions_done": 0,
            "condition_id": None, "cfm_pairs_done": 0, "cfm_pairs_total": 0,
            "mmd_pass": None, "mmd_microbatch_index": 0, "mmd_microbatches_total": 0,
            "mmd_microbatches_done": 0, "pass_ode_steps_reported": 0,
            "pass_ode_steps_total": 0,
            "ode_step": 0, "ode_steps_total": 0, "ode_steps_reported": 0,
            "backward_status": None,
            "progress_semantics": "host_callbacks_without_cuda_synchronization",
        }

    def condition_start(self, index, total, condition_id=None):
        if not self.enabled:
            return
        now = self.clock()
        if self._grouped():
            if self.group_offset is None:
                self.group_offset = index - 1
            self.condition_started = now
            self.state = {**self.state, "phase": "condition start", "condition_index": index,
                          "conditions_total": total, "condition_id": condition_id,
                          "conditions_done": max(self.state["conditions_done"], index - 1),
                          "cfm_chunks_done": 0, "cfm_chunks_total": 0}
            self._publish_grouped(now)
            return
        self._close_stage()
        self.condition_started = now
        self.state = {**self.state, "phase": "condition start", "condition_index": index,
                      "conditions_total": total, "condition_id": condition_id,
                      "cfm_pairs_done": 0, "cfm_pairs_total": 0,
                      "mmd_pass": None, "mmd_microbatch_index": 0,
                      "mmd_microbatches_total": 0, "ode_step": 0,
                      "mmd_microbatches_done": 0, "pass_ode_steps_reported": 0,
                      "pass_ode_steps_total": 0,
                      "ode_steps_total": 0, "backward_status": None}
        self._publish(now, force=True)

    def __call__(self, phase, done, total):
        if not self.enabled:
            return
        now = self.clock()
        if self._grouped():
            self.state = {**self.state, "phase": phase}
            if phase == "CFM grouped backward":
                self.state.update(cfm_chunks_done=done, cfm_chunks_total=total)
            self._publish_grouped(now)
            return
        previous = self.state
        values = {"phase": phase}
        if phase.startswith("CFM"):
            values.update(cfm_pairs_done=done, cfm_pairs_total=total)
        if phase in {"MMD generate", "MMD replay"}:
            new_pass = phase.split()[1] != previous["mmd_pass"]
            values.update(mmd_pass=phase.split()[1],
                          mmd_microbatch_index=min(done // self.microbatch + 1,
                                                   math.ceil(total / self.microbatch)),
                          mmd_microbatches_total=math.ceil(total / self.microbatch),
                          ode_step=0, ode_steps_total=0, backward_status=None)
            if new_pass:
                values.update(mmd_microbatches_done=0, pass_ode_steps_reported=0,
                              pass_ode_steps_total=0)
        if phase in {"ODE generate", "ODE replay"}:
            values.update(ode_step=done, ode_steps_total=total,
                          pass_ode_steps_reported=(max(0, previous["mmd_microbatch_index"] - 1) * total + done),
                          pass_ode_steps_total=previous["mmd_microbatches_total"] * total,
                          ode_steps_reported=previous["ode_steps_reported"] +
                          max(0, done - previous["ode_step"]))
            if phase == "ODE generate" and done == total:
                values["mmd_microbatches_done"] = previous["mmd_microbatch_index"]
        if "backward" in phase:
            values["backward_status"] = "done" if phase.endswith("done") else "started"
            if phase == "ODE backward done":
                values["mmd_microbatches_done"] = previous["mmd_microbatch_index"]
        elif not phase.startswith("CFM"):
            values["backward_status"] = None
        self.state = {**previous, **values}
        self._update_stage()
        # CFM forward/backward/pairs are one stage, not three forced repaints
        # per microbatch. Long calls still publish their return after interval.
        if phase.startswith("CFM"):
            force = (not previous["phase"].startswith("CFM") or
                     (done == total and previous["cfm_pairs_done"] != total))
        else:
            force = phase != previous["phase"] or done == total
        self._publish(now, force=force)

    def condition_done(self):
        if not self.enabled:
            return
        if self._grouped():
            done = min(self.state["conditions_done"] + 1, self.state["conditions_total"])
            self.state = {**self.state, "phase": "condition done",
                          "conditions_done": done, "condition_index": done}
            self._publish_grouped(self.clock())
            return
        self.state = {**self.state, "phase": "condition done",
                      "conditions_done": self.state["condition_index"]}
        self._publish(self.clock(), force=True)
        self._close_stage()

    def close(self, status="complete"):
        if not self.enabled or self.closed:
            return
        # Failed/interrupted runs retain the last actual counters and phase.
        self.state = {**self.state, "status": status}
        if self._grouped():
            self._publish_grouped(self.clock(), force=True)
            if self.group_bar is not None:
                self.group_bar.close()
            self.closed = True
            return
        self._publish(self.clock(), force=True)
        self._close_stage()
        self.closed = True

    def _grouped(self):
        return self.state.get("condition_batching") == "grouped_compact_observed_full_direct"

    def _publish_grouped(self, now, force=False):
        if self.closed or (not force and now - self.last_write < self.interval):
            return
        elapsed = max(0., now - self.started)
        completed = self.state["conditions_done"] - (self.group_offset or 0)
        rate = completed / elapsed if elapsed and completed else None
        self.state = {**self.state, "elapsed_seconds": elapsed,
                      "condition_elapsed_seconds": max(0., now - self.condition_started),
                      "conditions_per_second": rate,
                      "updated_at": datetime.now().astimezone().isoformat()}
        phase = self.state["phase"]
        label = {"condition start": "准备", "condition done": "完成",
                 "CFM grouped backward": "CFM完成",
                 "MMD grouped backward": "MMD完成", "optimizer": "更新"}.get(phase, phase)
        if phase == "CFM grouped backward":
            label += f" {self.state['cfm_chunks_done']}/{self.state['cfm_chunks_total']}批"
        message = f"{label}, {rate:.2f}条件/s" if rate is not None else f"{label}, 速率待完成"
        if self.bar_factory is not None and self.state["conditions_total"]:
            if self.group_bar is None:
                prefix = (f"P{self.state['member']}/{self.state['members']} E{self.state['epoch']} "
                          if "members" in self.state else "")
                self.group_bar = self.bar_factory(total=self.state["conditions_total"],
                    initial=self.group_offset or 0, unit="条件", position=0, leave=True,
                    dynamic_ncols=False, ncols=80, mininterval=float("inf"),
                    desc=f"{prefix}条件", bar_format="{desc}: {percentage:5.1f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]")
            self.group_bar.set_postfix_str(message, refresh=False)
            self.group_bar.update(max(0, self.state["conditions_done"] - self.group_bar.n))
            self.group_bar.refresh()
        elif self.bar is not None:
            self.bar.set_postfix_str(f"条件 {self.state['conditions_done']}/{self.state['conditions_total']} | {message}", refresh=True)
        self._write_state()
        self.last_write = now

    def _close_stage(self):
        if self.stage_bar is not None:
            self.stage_bar.close()
        self.stage_bar, self.stage = None, None

    def _update_stage(self):
        if self.bar_factory is None:
            return
        s, phase = self.state, self.state["phase"]
        if phase.startswith("CFM"):
            stage, label, unit = "CFM", "CFM", "pair"
            current, total = s["cfm_pairs_done"], s["cfm_pairs_total"]
            detail = "反传进行中" if s["backward_status"] == "started" else ""
        elif phase in {"MMD generate", "ODE generate", "MMD replay", "ODE replay",
                       "ODE backward", "ODE backward start", "ODE backward done"}:
            stage = s["mmd_pass"]
            label = "MMD生成" if stage == "generate" else "MMD重放"
            unit = "ODE步"
            current, total = s["pass_ode_steps_reported"], s["pass_ode_steps_total"]
            batch = f"{s['mmd_microbatch_index']}/{s['mmd_microbatches_total']}"
            detail = (f"反传{batch}进行中" if s["backward_status"] == "started" else
                      f"反传{batch}完成" if s["backward_status"] == "done" else f"批{batch}")
        else:
            self._close_stage()
            return
        if stage != self.stage:
            self._close_stage()
        if not total:
            return  # ODE total is known only when its first callback arrives.
        if self.stage_bar is None:
            self.stage = stage
            prefix = (f"P{s['member']}/{s['members']} E{s['epoch']} "
                      if "members" in s else "")
            self.stage_bar = self.bar_factory(
                total=total, unit=unit, position=0, leave=False,
                dynamic_ncols=False, ncols=80,
                desc=f"{prefix}条件{s['condition_index']}/{s['conditions_total']} {label}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]")
        self.stage_bar.set_postfix_str(detail, refresh=False)
        self.stage_bar.update(max(0, current - self.stage_bar.n))

    def _publish(self, now, *, force=False):
        if not force and now - self.last_write < self.interval:
            return
        elapsed = max(0., now - self.started)
        self.state = {**self.state, "elapsed_seconds": elapsed,
                      "condition_elapsed_seconds": max(0., now - self.condition_started),
                      "ode_reported_steps_per_second": (
                          self.state["ode_steps_reported"] / elapsed if elapsed else None),
                      "updated_at": datetime.now().astimezone().isoformat()}
        s = self.state
        parts = [f"条件 {s['condition_index']}/{s['conditions_total']}",
                 f"CFM pairs {s['cfm_pairs_done']}/{s['cfm_pairs_total']}"]
        if s["mmd_pass"] is not None:
            parts.append(f"MMD {s['mmd_pass']} microbatch "
                         f"{s['mmd_microbatch_index']}/{s['mmd_microbatches_total']} "
                         f"已完成 {s['mmd_microbatches_done']}/{s['mmd_microbatches_total']}")
        if s["ode_steps_total"]:
            parts.append(f"ODE {s['ode_step']}/{s['ode_steps_total']} "
                         f"本阶段累计 {s['pass_ode_steps_reported']}/{s['pass_ode_steps_total']} 步")
        parts.append(s["phase"])
        if s["backward_status"] is not None:
            parts.append("backward 调用完成" if s["backward_status"] == "done" else "backward 开始")
        rate = s["ode_reported_steps_per_second"]
        parts.extend([f"ODE累计(主机) {s['ode_steps_reported']} 步",
                      f"{rate:.2f} 步/s" if rate is not None else "速率待计时",
                      f"已耗时 {elapsed:.0f}s"])
        if self.bar is not None:
            message = (f"条件{s['condition_index']}/{s['conditions_total']} · {s['phase']}"
                       if self.bar_factory is not None else " | ".join(parts))
            self.bar.set_postfix_str(message, refresh=True)
        if self.stage_bar is not None and force:
            self.stage_bar.refresh()
        self._write_state()
        self.last_write = now

    def _write_state(self):
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_name(self.state_path.name + ".tmp")
            temporary.write_text(json.dumps(self.state, ensure_ascii=False,
                                            allow_nan=False), encoding="utf-8")
            temporary.replace(self.state_path)
