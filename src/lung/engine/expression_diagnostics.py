"""Frozen correction and calibrated discrepancy reports on independent expression tests."""

import json
from pathlib import Path

import torch

from .auxiliary import load_auxiliary, verify_auxiliary_dependencies, encode_vectors
from .config import file_hash, StageUnavailable
from .response import projected, condition_kwargs
from .residuals import gate_for


class FrozenExpressionDiagnostics:
    def __init__(self, pipeline, data, device, reference):
        self.data, self.device, self.reference = data, device, reference
        self.modules, self.gates, self.calibration = {}, None, None
        for key in ("pool", "backward", "correction", "uncertainty"):
            if pipeline.get(key):
                verify_auxiliary_dependencies(pipeline[key], pipeline)
                self.modules[key] = load_auxiliary(pipeline[key], data, device)
        if pipeline.get("correction_gate"):
            self.gates = json.loads(Path(pipeline["correction_gate"]).read_text(encoding="utf-8"))
            expected = {"response_hashes": [file_hash(p) for p in pipeline["response"]],
                        "pool_hash": file_hash(pipeline["pool"]), "backward_hash": file_hash(pipeline["backward"]),
                        "correction_hash": file_hash(pipeline["correction"]), "data_identity": data.identity}
            if any(self.gates.get(k) != v for k, v in expected.items()):
                raise StageUnavailable("CORRECTION_GATE_VERSION_MISMATCH")
        if pipeline.get("uncertainty_calibration"):
            self.calibration = json.loads(Path(pipeline["uncertainty_calibration"]).read_text(encoding="utf-8"))
            expected = {"data_identity": data.identity, "uncertainty_hash": file_hash(pipeline["uncertainty"]),
                        "gate_hash": file_hash(pipeline["correction_gate"])}
            if any(self.calibration.get(k) != v for k, v in expected.items()):
                raise StageUnavailable("ERROR_CALIBRATION_VERSION_MISMATCH")

    @torch.no_grad()
    def apply(self, raw_members, control, batch):
        if not raw_members or control.ndim != 2 or any(value.shape != control.shape for value in raw_members):
            raise ValueError("Equal-size members in shared control order required")
        raw = torch.cat(raw_members)
        temporary = [projected(value, self.data) for value in raw_members]
        baseline = torch.cat(temporary)
        report = {"correction_enabled": False, "population_correction_applied": False,
                  "member_count": len(raw_members), "control_block_count": len(control),
                  "member_population_size": len(control), "member_major_order": True,
                  "mmd_estimator": "equal_member_shared_control_block_U",
                  "error_calibration_status": "unavailable"}
        if not {"pool", "backward", "correction"} <= self.modules.keys():
            return baseline, baseline, report
        z = encode_vectors(self.reference, baseline.cpu().numpy(), self.device).to(self.device)
        population = self.modules["pool"](z)
        condition = self.reference.encode_condition(**condition_kwargs(batch, 1, self.device))[0]
        e_b = self.modules["backward"].diagnostic(population, condition, control.mean(0))
        variance = torch.stack([value.mean(0) for value in temporary]).var(0, correction=0).mean()
        gate = gate_for(self.gates, batch) if self.gates is not None else False
        output = self.modules["correction"](raw, population, condition, e_b, variance, gate=gate)
        valid = projected(output, self.data)
        report.update(correction_enabled=bool(gate), population_correction_applied=bool(gate),
                      mmd_estimator="empirical_population_V" if gate else "equal_member_shared_control_block_U",
                      backward_discrepancy=float(e_b), ensemble_variance=float(variance))
        domain = f"{batch['source_id']}|{batch['background_id']}|{batch['protocol_id']}"
        in_domain = self.gates is not None and domain in self.gates.get("domains", {})
        if "uncertainty" in self.modules and in_domain:
            estimate = float(self.modules["uncertainty"](population, condition, e_b, variance, int(gate)))
            report["estimated_expression_discrepancy"] = estimate
            if self.calibration is not None:
                quantile = self.calibration["q_error"]
                report.update(error_calibration_status="empirical_group_interval", coverage_guarantee=False,
                              error_interval_low=max(0., estimate-quantile) if isinstance(quantile, (int, float)) else 0.,
                              error_interval_high=estimate+quantile if isinstance(quantile, (int, float)) else None,
                              error_interval_unbounded=quantile == "infinity",
                              conditional_mean_rms_radius=self.calibration["q_mean"])
        return valid, baseline, report
