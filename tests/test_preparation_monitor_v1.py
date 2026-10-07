"""Preparation UI/manifest regressions; no training or dashboard service."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lung.monitoring import RunMonitor


class FakeBar:
    def __init__(self, total, desc):
        self.total, self.desc, self.n = total, desc, 0
        self.close_count = 0

    def update(self, amount):
        self.n += amount

    def close(self):
        self.close_count += 1


class PreparationMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.now = 0.0
        self.writer = Mock()
        for replacement in (
            patch("lung.monitoring.run.ROOT", Path(self.temporary.name)),
            patch("lung.monitoring.run.SummaryWriter", return_value=self.writer),
            patch("lung.monitoring.run.Console"),
            patch("lung.monitoring.run.implementation_identity", return_value="test-only"),
            patch("lung.monitoring.run.torch.cuda.is_available", return_value=False),
            patch("lung.monitoring.run.time.monotonic", side_effect=lambda: self.now),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.monitor = RunMonitor({"stage": "response", "task": "RESPONSE-A549-V1",
                                   "epochs": 1, "device": "cpu", "synthetic": True,
                                   "dashboard": False}, run_id="preparation-ui-test")
        self.bars = []

        def new_bar(**kwargs):
            bar = FakeBar(**kwargs)
            self.bars.append(bar)
            return bar

        self.monitor.progress = Mock(side_effect=new_bar)
        self.addCleanup(self.close_monitor)

    def close_monitor(self):
        if not self.monitor.csv_file.closed:
            self.monitor.close("interrupted")

    def snapshot(self):
        return json.loads((self.monitor.experiment / "run_manifest.json").read_text("utf-8"))

    def test_stage_start_is_visible_and_does_not_create_training_metrics(self):
        csv_position = self.monitor.csv_file.tell()
        self.monitor.preparation_progress("gene_graph_cuda", 0, 128)
        preparation = self.snapshot()["preparation"]
        self.assertEqual(preparation["current"], 0)
        self.assertEqual(preparation["total"], 128)
        self.assertEqual(preparation["fraction"], 0)
        self.assertEqual(preparation["status"], "running")
        self.assertEqual(self.bars[0].n, 0)
        self.assertEqual(self.snapshot()["latest"], {})
        self.assertEqual(self.monitor.csv_file.tell(), csv_position)
        self.assertEqual(self.writer.add_scalar.call_count, 4)
        for call in self.writer.add_scalar.call_args_list:
            self.assertTrue(call.args[0].startswith("preparation/gene_graph_cuda/"))
            self.assertNotIn("epoch", call.args[0])
        self.writer.flush.assert_called()

    def test_snapshots_throttle_to_five_seconds_but_phase_boundaries_flush(self):
        self.monitor.preparation_progress("normalization", 0, 100)
        initial_calls = self.writer.add_scalar.call_count
        self.now = 4.999
        self.monitor.preparation_progress("normalization", 10, 100)
        self.assertEqual(self.writer.add_scalar.call_count, initial_calls)
        self.assertEqual(self.snapshot()["preparation"]["current"], 0)
        self.assertEqual(self.bars[0].n, 10)
        self.now = 5.0
        self.monitor.preparation_progress("normalization", 20, 100)
        self.assertEqual(self.writer.add_scalar.call_count, initial_calls + 4)
        self.assertEqual(self.snapshot()["preparation"]["current"], 20)
        self.now = 5.1
        self.monitor.preparation_progress("gene_graph_cuda", 0, 200)
        self.assertEqual(self.snapshot()["preparation"]["phase"], "gene_graph_cuda")
        self.assertEqual(self.snapshot()["preparation"]["phase_index"], 2)
        self.assertEqual(self.writer.add_scalar.call_count, initial_calls + 8)

    def test_completed_bar_closes_immediately_and_is_not_retimed_by_next_phase(self):
        self.monitor.preparation_progress("fit_expression_cache", 0, 10)
        self.now = 2.0
        self.monitor.preparation_progress("fit_expression_cache", 10, 10)
        self.assertEqual(self.bars[0].close_count, 1)
        self.assertIsNone(self.monitor._preparation_bar)
        self.assertEqual(self.snapshot()["preparation"]["phase_seconds"], 2)
        self.now = 100.0
        self.monitor.preparation_progress("fit_expression_cache", 10, 10)
        self.assertEqual(self.snapshot()["preparation"]["phase_seconds"], 2)
        self.monitor.preparation_progress("gene_graph_cuda", 0, 32)
        self.assertEqual(self.bars[0].close_count, 1)
        self.assertEqual(self.bars[1].n, 0)
        self.assertEqual(self.snapshot()["preparation"]["phase_seconds"], 0)

    def test_same_phase_reset_flushes_pending_work_and_starts_new_bar(self):
        self.monitor.preparation_progress("validation_cache", 0, 10)
        self.now = 1.0
        self.monitor.preparation_progress("validation_cache", 8, 10)
        self.now = 2.0
        self.monitor.preparation_progress("validation_cache", 0, 10)
        self.assertEqual(len(self.bars), 2)
        self.assertEqual(self.bars[0].close_count, 1)
        self.assertEqual(self.bars[1].n, 0)
        self.assertEqual(self.snapshot()["preparation"]["phase_index"], 2)
        currents = [call.args[1] for call in self.writer.add_scalar.call_args_list
                    if call.args[0] == "preparation/validation_cache/current"]
        self.assertEqual(currents, [0, 8, 0])
        self.now = 3.0
        self.monitor.preparation_progress("validation_cache", 1, 10)
        self.assertEqual(self.bars[1].n, 1)

    def test_interrupted_finish_preserves_partial_counts_and_is_idempotent(self):
        self.monitor.preparation_progress("gene_graph_cuda", 0, 100)
        self.now = 3.0
        self.monitor.preparation_progress("gene_graph_cuda", 17, 100)
        self.monitor.finish_preparation("interrupted")
        preparation = self.snapshot()["preparation"]
        self.assertEqual(preparation["status"], "interrupted")
        self.assertEqual(preparation["phase_status"], "interrupted")
        self.assertEqual(preparation["current"], 17)
        self.assertAlmostEqual(preparation["fraction"], .17)
        self.assertEqual(self.bars[0].close_count, 1)
        calls = self.writer.add_scalar.call_count
        self.monitor.finish_preparation()
        self.monitor.close("interrupted")
        self.assertEqual(self.writer.add_scalar.call_count, calls)
        self.assertEqual(self.snapshot()["preparation"]["status"], "interrupted")
        self.assertEqual(self.snapshot()["status"], "interrupted")

    def test_successful_finish_preserves_completed_phase_time(self):
        self.monitor.preparation_progress("graph_selection", 0, 10)
        self.now = 2.0
        self.monitor.preparation_progress("graph_selection", 10, 10)
        self.now = 20.0
        self.monitor.finish_preparation()
        self.monitor.close("prepared_only")
        preparation = self.snapshot()["preparation"]
        self.assertEqual(preparation["status"], "complete")
        self.assertEqual(preparation["phase_status"], "complete")
        self.assertEqual(preparation["phase_seconds"], 2)
        self.assertEqual(preparation["elapsed_seconds"], 20)
        self.assertEqual(self.bars[0].close_count, 1)

    def test_failed_close_flushes_last_partial_snapshot_before_writer_closes(self):
        self.monitor.preparation_progress("normalization", 0, 100)
        self.now = 1.0
        self.monitor.preparation_progress("normalization", 3, 100)
        self.monitor.close("failed", error_type="ExampleFailure")
        manifest = self.snapshot()
        self.assertEqual(manifest["preparation"]["status"], "failed")
        self.assertEqual(manifest["preparation"]["current"], 3)
        self.assertEqual(manifest["status"], "failed")
        self.writer.close.assert_called_once()
        names = [call[0] for call in self.writer.mock_calls]
        self.assertLess(max(i for i, name in enumerate(names) if name == "add_scalar"),
                        names.index("close"))


if __name__ == "__main__":
    unittest.main()
