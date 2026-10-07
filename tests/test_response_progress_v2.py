"""CPU-only progress contract: independent units, throttling and truthful state."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lung.monitoring.response_progress import ResponseTrainingProgress


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class Bar:
    def __init__(self):
        self.messages = []

    def set_postfix_str(self, message, refresh):
        self.messages.append(message)


def test_16_microbatches_keep_condition_pairs_and_ode_axes_separate(tmp_path):
    clock, bar = Clock(), Bar()
    path = tmp_path / "current.json"
    progress = ResponseTrainingProgress(bar, state_path=path, clock=clock, mmd_microbatch=2)
    progress.condition_start(1, 644, "condition-A")
    progress("CFM pairs", 128, 128)
    for offset in range(0, 32, 2):
        progress("MMD generate", offset, 32)
        for step in range(1, 33):
            clock.now += .1
            progress("ODE generate", step, 32)
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["conditions_done"] == 0
    assert state["condition_index"] == 1 and state["conditions_total"] == 644
    assert state["cfm_pairs_done"] == state["cfm_pairs_total"] == 128
    assert state["mmd_microbatch_index"] == state["mmd_microbatches_total"] == 16
    assert state["mmd_microbatches_done"] == 16
    assert state["pass_ode_steps_reported"] == state["pass_ode_steps_total"] == 512
    assert state["ode_step"] == state["ode_steps_total"] == 32
    assert state["ode_steps_reported"] == 512
    assert state["ode_reported_steps_per_second"] == pytest.approx(10.)
    assert "microbatch 16/16" in bar.messages[-1]
    assert "ODE 32/32" in bar.messages[-1]
    assert "条件 1/644" in bar.messages[-1]
    assert "ETA" not in bar.messages[-1] and "%" not in bar.messages[-1]
    assert not path.with_suffix(".json.tmp").exists()


def test_backward_start_and_done_forced_and_replay_counts_are_cumulative(tmp_path):
    clock, bar = Clock(), Bar()
    path = tmp_path / "current.json"
    progress = ResponseTrainingProgress(bar, state_path=path, clock=clock, mmd_microbatch=2)
    progress.condition_start(2, 644)
    progress("CFM backward start", 0, 128)
    assert json.loads(path.read_text())["backward_status"] == "started"
    clock.now = 1.
    progress("CFM backward done", 2, 128)
    assert json.loads(path.read_text())["backward_status"] == "done"
    for phase in ("generate", "replay"):
        progress(f"MMD {phase}", 0, 3)
        progress(f"ODE {phase}", 32, 32)
    progress("ODE backward start", 0, 3)
    before = json.loads(path.read_text())
    assert before["backward_status"] == "started"
    assert before["ode_steps_reported"] == 64
    assert before["mmd_microbatches_done"] == 0
    assert before["pass_ode_steps_reported"] == 32
    assert before["pass_ode_steps_total"] == 64
    clock.now = 120.
    progress("ODE backward done", 2, 3)
    after = json.loads(path.read_text())
    assert after["backward_status"] == "done"
    assert after["elapsed_seconds"] == 120.
    assert after["mmd_microbatches_total"] == 2
    assert after["mmd_microbatches_done"] == 1
    progress.condition_done()
    assert json.loads(path.read_text())["conditions_done"] == 2


def test_intermediate_steps_throttle_both_console_and_disk(tmp_path):
    clock, bar = Clock(), Bar()
    path = tmp_path / "current.json"
    progress = ResponseTrainingProgress(bar, state_path=path, clock=clock)
    progress("ODE generate", 1, 32)
    initial = path.read_bytes()
    clock.now = .5
    progress("ODE generate", 2, 32)
    assert path.read_bytes() == initial and len(bar.messages) == 1
    clock.now = 1.
    progress("ODE generate", 3, 32)
    assert json.loads(path.read_text())["ode_step"] == 3
    assert len(bar.messages) == 2
    progress("ODE generate", 32, 32)
    assert json.loads(path.read_text())["ode_step"] == 32
    assert len(bar.messages) == 3


def test_disabled_is_silent_and_does_not_create_files(tmp_path, capsys):
    bar = Bar()
    path = tmp_path / "current.json"
    def forbidden_clock():
        raise AssertionError("disabled progress must not read clock")
    progress = ResponseTrainingProgress(bar, state_path=path, enabled=False, clock=forbidden_clock)
    progress.condition_start(1, 644)
    progress("ODE generate", 1, 32)
    progress.condition_done()
    progress.close()
    assert not path.exists() and not bar.messages
    assert capsys.readouterr() == ("", "")


def test_interrupted_close_preserves_real_counts_and_new_condition_resets_locals(tmp_path):
    path = tmp_path / "current.json"
    progress = ResponseTrainingProgress(Bar(), state_path=path, clock=Clock())
    progress.condition_start(1, 644)
    progress("CFM pairs", 128, 128)
    progress("MMD generate", 0, 32)
    progress("ODE generate", 32, 32)
    progress.condition_done()
    progress.condition_start(2, 644)
    progress("MMD generate", 0, 32)
    progress("ODE generate", 1, 32)
    progress.close("interrupted")
    state = json.loads(path.read_text())
    assert state["status"] == "interrupted" and state["conditions_done"] == 1
    assert state["condition_index"] == 2 and state["cfm_pairs_done"] == 0
    assert state["ode_step"] == 1 and state["ode_steps_reported"] == 33
    assert state["phase"] == "ODE generate"


class StageBar(Bar):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs, self.n, self.closed = kwargs, 0, False
        self.counts = []
        self.refresh_count = 0

    def update(self, count):
        self.n += count
        self.counts.append(self.n)

    def refresh(self):
        self.refresh_count += 1

    def close(self):
        self.closed = True


def test_stage_bars_use_independent_totals_and_never_reset_between_microbatches():
    stages, main, clock = [], Bar(), Clock()
    def factory(**kwargs):
        bar = StageBar(**kwargs)
        stages.append(bar)
        return bar
    progress = ResponseTrainingProgress(main, bar_factory=factory, clock=clock, mmd_microbatch=2)
    progress.condition_start(1, 644)
    progress("CFM backward start", 0, 128)
    progress("CFM backward done", 128, 128)
    assert stages[0].n == stages[0].kwargs["total"] == 128
    assert stages[0].kwargs["unit"] == "pair"
    for phase in ("generate", "replay"):
        for offset in range(0, 32, 2):
            progress(f"MMD {phase}", offset, 32)
            for step in range(1, 33):
                progress(f"ODE {phase}", step, 32)
            if phase == "replay":
                progress("ODE backward start", offset, 32)
                assert "进行中" in stages[-1].messages[-1]
                progress("ODE backward done", offset + 2, 32)
    assert len(stages) == 3
    for stage in stages[1:]:
        assert stage.kwargs["total"] == stage.n == 512
        assert stage.kwargs["unit"] == "ODE步"
        assert stage.counts == sorted(stage.counts)
        assert stage.kwargs["position"] == 0 and stage.kwargs["leave"] is False
        assert stage.kwargs["ncols"] == 80 and stage.kwargs["dynamic_ncols"] is False
        assert "remaining" not in stage.kwargs["bar_format"]
    assert stages[0].closed and stages[1].closed
    progress.condition_done()
    assert stages[2].closed
    assert max(map(len, main.messages)) < 60


def test_cfm_microbatch_callbacks_share_throttle(tmp_path):
    clock, bar = Clock(), Bar()
    path = tmp_path / "current.json"
    progress = ResponseTrainingProgress(bar, clock=clock, state_path=path)
    progress("CFM backward start", 0, 128)
    initial = path.read_bytes()
    for start in range(0, 126, 2):
        progress("CFM backward start", start, 128)
        progress("CFM backward done", start + 2, 128)
        progress("CFM pairs", start + 2, 128)
    assert len(bar.messages) == 1 and path.read_bytes() == initial
    progress("CFM backward done", 128, 128)
    progress("CFM pairs", 128, 128)
    assert len(bar.messages) == 2
    assert json.loads(path.read_text())["cfm_pairs_done"] == 128


def test_interruption_closes_partial_stage_without_filling_it_and_disable_skips_factory():
    stages = []
    def factory(**kwargs):
        bar = StageBar(**kwargs)
        stages.append(bar)
        return bar
    progress = ResponseTrainingProgress(Bar(), bar_factory=factory, clock=Clock())
    progress.condition_start(1, 644)
    progress("MMD generate", 0, 16)
    assert not stages
    progress("ODE generate", 1, 32)
    progress.close("interrupted")
    assert stages[0].closed and stages[0].n == 1
    assert stages[0].kwargs["total"] == 512
    disabled = ResponseTrainingProgress(Bar(), bar_factory=factory, enabled=False)
    disabled.condition_start(1, 644)
    disabled("CFM pairs", 1, 128)
    disabled.close()
    assert len(stages) == 1


def test_real_non_tty_output_stays_on_one_line_without_cursor_up():
    """PyCharm Run supports carriage return, but not nested tqdm cursor-up."""
    import io
    from types import SimpleNamespace
    from lung.monitoring.run import RunMonitor
    from tqdm.utils import disp_len

    output, clock = io.StringIO(), Clock()
    monitor = SimpleNamespace(config={"progress": True})
    main = RunMonitor.progress(monitor, total=644, disable=True, file=output, position=0)
    def factory(**kwargs):
        return RunMonitor.progress(monitor, file=output, mininterval=0, miniters=1, **kwargs)
    progress = ResponseTrainingProgress(main, bar_factory=factory, clock=clock, mmd_microbatch=2)
    progress.state = {**progress.state, "member": 1, "members": 5, "epoch": 1}
    progress.condition_start(1, 644)
    for start in range(0, 128, 2):
        progress("CFM backward start", start, 128)
        clock.now += 3
        progress("CFM backward done", start + 2, 128)
    for phase in ("generate", "replay"):
        for offset in range(0, 32, 2):
            progress(f"MMD {phase}", offset, 32)
            for step in range(1, 33):
                clock.now += 1
                progress(f"ODE {phase}", step, 32)
            if phase == "replay":
                progress("ODE backward start", offset, 32)
                progress("ODE backward done", offset + 2, 32)
    main.update()
    progress.condition_done()
    progress.close()
    main.close()
    text = output.getvalue()
    assert "P1/5 E1" in text and "128/128" in text and "512/512" in text
    assert "/s" in text and text.count("\r") > 10
    assert "\n" not in text
    assert "\x1b[A" not in text  # No cursor-up needed on a non-TTY console.
    assert all(disp_len(frame) <= 80 for frame in text.split("\r"))
