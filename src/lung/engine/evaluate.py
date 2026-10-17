"""Independent frozen E evaluation. No fit, optimizer, gate selection or calibration."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from lung.models import rbf_mmd
from lung.models.objectives import shared_control_mmd
from lung.rl import (DLPAPlanner, generate_fixed_sequences, reevaluate_fixed_sequences, audit_winners)
from .checkpoint import atomic_json, load_checkpoint, seed_all
from .config import ROOT, StageUnavailable, load_config, project_path, file_hash
from .data import open_data
from .response import tensor, condition_kwargs, projected
from .response_loader import load_response
from .inference_noise import response_noise


@torch.no_grad()
def evaluate_expression(models, data, config, output, pipeline=None, *, role="E"):
    from tqdm.auto import tqdm
    from .expression_diagnostics import FrozenExpressionDiagnostics
    if role not in ("V", "E"):
        raise ValueError("EXPRESSION_EVALUATION_REQUIRES_V_OR_E")
    diagnostics = FrozenExpressionDiagnostics(pipeline or {}, data, config["device"], models[0])
    size = config["response"]["mmd_samples"] if config.get("synthetic") else config["population"]
    rows, predictions = [], {}
    rng = np.random.default_rng(config["seed"] + 901)
    for condition_id in tqdm(data.condition_ids(role), desc=f"Frozen {role} expression",
                             disable=not config.get("progress", True)):
        batch = data.sample_response(condition_id, size, rng, role)
        x, y = tensor(batch["control"], config["device"]), tensor(batch["target"], config["device"])
        generated = []
        population_dependent = any(model.config.reference_mode == "population_mean" for model in models)
        micro = config["response"].get("inference_microbatch", 1)
        for member, model in enumerate(models):
            member_output = []
            reference = model.condition_reference(x)
            from lung.rl import stream_seed
            generator = torch.Generator(device=x.device).manual_seed(
                stream_seed(config["seed"], "frozen-expression-v2", condition_id, member))
            noise = response_noise(model, reference, generator)
            for index in range(0, size, micro):
                end = min(size, index + micro)
                ca = model.encode_condition(**condition_kwargs(batch, end-index, config["device"]))
                raw = model.sample(reference[index:end], ca, noise=noise[index:end],
                    steps=config["response"]["inference_ode_steps"], checkpoint=False)
                member_output.append(raw)
            generated.append(torch.cat(member_output))
        samples, baseline, diagnostics_report = diagnostics.apply(generated, x, batch)
        # All methods share the empirical-population V discrepancy. U terminology
        # is reserved for the baseline with independent control blocks.
        baseline_u = (None if population_dependent else float(shared_control_mmd(
            list(baseline.split(size)), y, data.bandwidth)))
        corrected = diagnostics_report.get("population_correction_applied", False)
        mse = (samples.mean(0) - y.mean(0)).square().mean()
        rows.append({"condition_id": condition_id, "group_id": batch["group_id"], "role": role,
                     "population_mmd_v": float(rbf_mmd(samples, y, data.bandwidth, unbiased=False)),
                     "shared_control_mmd_u": None if corrected else baseline_u,
                     "mean_mse": float(mse),
                     "uncorrected_shared_control_mmd_u": baseline_u,
                     "uncorrected_population_mmd_v": float(rbf_mmd(baseline, y, data.bandwidth, unbiased=False)),
                     "uncorrected_mean_mse": float((baseline.mean(0)-y.mean(0)).square().mean()),
                     **diagnostics_report,
                     "reference_population_dependence": population_dependent,
                     "mmd_estimator": "empirical_population_V_with_conditional_block_U_diagnostic",
                     "n_generated": len(samples), "n_observed": len(y)})
        predictions[condition_id + "__mean"] = samples.mean(0).cpu().numpy()
        predictions[condition_id + "__variance"] = samples.var(0, correction=0).cpu().numpy()
    if not rows:
        raise StageUnavailable("FINAL_TEST_EMPTY")
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "conditions.csv", index=False, encoding="utf-8-sig")
    group_means = frame.groupby("group_id")[["population_mmd_v", "mean_mse"]].mean()
    np.savez_compressed(output / "expression_summary.npz", **predictions)
    return {"evaluation": "heldout_chemical_expression" if role == "E" else "development_expression",
            "role": role, "conditions": len(rows),
            "independent_chemical_groups": len(group_means),
            "group_equal_metrics": {key: float(value) for key, value in group_means.mean().items()},
            "mmd_estimator": "empirical_population_V_not_unbiased_population_estimate",
            "shared_control_U": "reported_only_without_population_correction_or_mean_reference",
            "evaluation_version": "frozen-expression-v2", "apoptosis_rate": None,
            "coupling_assumption": "minibatch_entropic_ot_prior", "model_updates": 0}


def readable_evaluation(evaluation):
    return {"candidates": [{"sequence": list(key), "mean": value,
                             "standard_error": evaluation.standard_errors[key]}
                            for key, value in evaluation.means.items()],
            "repeats": evaluation.repeats, "metadata": evaluation.metadata}


def state_evaluation_config(config, metadata):
    """The evaluated plate is owned by the frozen checkpoint, not today's YAML."""
    fold = metadata.get("state_fold")
    if fold not in {"1_to_8", "8_to_1"}:
        raise StageUnavailable("STATE_CHECKPOINT_FOLD_UNRECORDED")
    return {**config, "state_fold": fold}


def evaluate_decisions(data, config, pipeline, output):
    from .world import BiologicalWorld
    from .decision_evaluation import load_policy_models, native_sequence
    world = BiologicalWorld(data, config, pipeline)
    trained = {}
    for kind in ("daac", "dlpa"):
        paths = pipeline.get(kind, [])
        paths = [paths] if isinstance(paths, str) else paths
        for index, path in enumerate(paths):
            saved = load_checkpoint(path)
            if saved["metadata"]["data_identity"] != data.identity:
                raise StageUnavailable("DECISION_DATA_IDENTITY_MISMATCH")
            if saved["metadata"].get("biological_identity") != world.biological_identity:
                raise StageUnavailable("DECISION_ENVIRONMENT_IDENTITY_MISMATCH")
            if (saved["metadata"].get("versions") != world.adapter.versions
                    or saved.get("learner", {}).get("versions") != world.adapter.versions):
                raise StageUnavailable("DECISION_REWARD_OR_PROTOCOL_VERSION_MISMATCH")
            trained[f"{kind}_seed_{index}"] = (kind, saved)
    if not trained:
        raise StageUnavailable("TRAINED_POLICY_REQUIRED")
    roots = world.roots("E")
    reports = []
    from .decision import dlpa_settings
    for root in roots:
        proposed, native = {}, {}
        if config.get("history_support", "single_exposure_only") == "single_exposure_only":
            legal = world.protocol.legal_mask(world.registry, ())
            enumerated = []
            for index in torch.nonzero(legal, as_tuple=False).flatten().tolist():
                sequence = (world.registry.candidates[index].action_id,)
                if not world.protocol.position(world.registry, sequence)[2]:
                    raise StageUnavailable("ENUMERATION_REQUIRES_COMPLETE_SINGLE_EXPOSURE")
                world.protocol.validate_complete(world.registry, sequence)
                enumerated.append(sequence)
            if not enumerated:
                raise StageUnavailable("EMPTY_ENUMERATION_CANDIDATES")
            proposed["enumeration"] = tuple(enumerated)
        for name, (kind, saved) in trained.items():
            if kind == "daac":
                actor, critics, critic_type = load_policy_models(saved, world.registry, config["device"], kind)
                for label, q in (("actor_native", None), ("q_native", critics)):
                    chosen, trace = native_sequence(actor, world.adapter, root, config["seed"],
                                                    critics=q, critic_type=critic_type)
                    native[name + "_" + label] = {"sequence": chosen, "trace": trace}
                    proposed[name + "_" + label] = (chosen,)
                proposed[name + "_reranked"] = generate_fixed_sequences(actor, world.adapter, root,
                    count=4 if config.get("synthetic") else 256, seed=config["seed"])
            else:
                proposal, _, _ = load_policy_models(saved, world.registry, config["device"], kind)
                chosen, trace = native_sequence(proposal, world.adapter, root, config["seed"])
                native[name + "_proposal_native"] = {"sequence": chosen, "trace": trace}
                proposed[name + "_proposal_native"] = (chosen,)
                planner = DLPAPlanner(world.adapter, proposal, dlpa_settings(config))
                plan = planner.plan(root, config["seed"])
                chosen = min(plan.search_scores, key=lambda s: (-plan.search_scores[s], s))
                native[name + "_search_native"] = {"sequence": chosen,
                    "selection": "search_scores_before_independent_reevaluation",
                    "top_ties": [s for s in plan.sequences if plan.search_scores[s] == plan.search_scores[chosen]]}
                proposed[name + "_search_native"] = (chosen,)
                proposed[name + "_reranked"] = plan.sequences
        pool = sorted(set(sequence for sequences in proposed.values() for sequence in sequences))
        evaluation = reevaluate_fixed_sequences(world.adapter, root, pool, seed=config["seed"]+1001,
                                                 repeats=2 if config.get("synthetic") else None)
        winners = audit_winners(world.adapter, root, evaluation, proposed, config["seed"]+2001,
                                repeats=2 if config.get("synthetic") else 4096)
        reports.append({"root": root.root_id, "role": "E", "proposed_by": proposed,
                        "native_before_world_reevaluation": native,
                        "reevaluation": readable_evaluation(evaluation),
                        "winners": winners["winners"], "audit": readable_evaluation(winners["audit"]),
                        "reselected_after_audit": False})
        atomic_json(output / "decisions.json", reports)
    return {"evaluation": "frozen_model_fixed_sequence_comparison", "roots": len(roots),
            "task": config["task"], "real_sequence_outcomes_available": False,
            "enumeration": config.get("history_support", "single_exposure_only") == "single_exposure_only",
            "native_and_reranked_separate": True, "Q_ranking": "saved_Q1_with_Q2_diagnostic",
            "transfer_status": "unvalidated", "apoptosis_rate": None, "model_updates": 0}


def main(argv=None):
    command = argparse.ArgumentParser(description="冻结模型的独立 test；不训练、不调门控、不校准")
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--weights", help="单个 response/state 权重；潜空间预测器使用 --pipeline")
    group.add_argument("--pipeline")
    command.add_argument("--data")
    command.add_argument("--device")
    command.add_argument("--mode", choices=("response", "state", "decision"), default="response")
    arguments = command.parse_args(argv)
    try:
        config, pipeline = load_config(), {}
        input_checkpoint = None
        if arguments.pipeline:
            record = json.loads(project_path(arguments.pipeline).read_text(encoding="utf-8"))
            config, pipeline = record["config"], record["artifacts"]
        if config.get("synthetic"):
            raise StageUnavailable("SYNTHETIC_ARTIFACT_NOT_SUPPORTED")
        if arguments.data:
            config["data"] = arguments.data
        if arguments.device:
            config["device"] = arguments.device
        if arguments.weights:
            saved = load_checkpoint(project_path(arguments.weights))
            input_checkpoint = saved
            if saved["metadata"].get("synthetic"):
                raise StageUnavailable("SYNTHETIC_ARTIFACT_NOT_SUPPORTED")
            if saved["metadata"]["stage"] == "response":
                pipeline["response"] = [str(project_path(arguments.weights))]
                config["response"] = saved["metadata"]["response_config"]
            elif saved["metadata"]["stage"] == "state":
                config = state_evaluation_config(config, saved["metadata"])
                if saved["metadata"].get("reference_representation"):
                    pipeline["representation"] = saved["metadata"]["reference_representation"]
                else:
                    pipeline["response"] = [saved["metadata"]["reference_response"]]
                pipeline["state"] = str(project_path(arguments.weights))
                arguments.mode = "state"
            else:
                raise StageUnavailable("COMPLETE_PIPELINE_REQUIRED: use --pipeline for latent response or decision evaluation")
        if pipeline.get("inference_ode_steps"):
            config["response"]["inference_ode_steps"] = pipeline["inference_ode_steps"]
        data = open_data(config, "response", prepare=False, test=True)
        output = ROOT / "experiments" / "test" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output.mkdir(parents=True)
        paths = [path for value in pipeline.values() for path in (value if isinstance(value, list) else [value])
                 if isinstance(path, str) and Path(path).is_file()]
        before = {path: file_hash(path) for path in paths}
        seed_all(config["seed"]+900)
        if arguments.mode == "response":
            models = [load_response(path, data, config["device"], saved=input_checkpoint)
                      for path in pipeline["response"]]
            report = evaluate_expression(models, data, config, output, pipeline)
        elif arguments.mode == "state":
            from .auxiliary import (load_auxiliary, encode_vectors, state_metrics,
                                    auxiliary_representation)
            saved_state = input_checkpoint if input_checkpoint is not None else load_checkpoint(pipeline["state"])
            config = state_evaluation_config(config, saved_state["metadata"])
            reference, _ = auxiliary_representation(data, pipeline, config["device"])
            classifier = load_auxiliary(pipeline["state"], data, config["device"], pipeline=pipeline, saved=saved_state)
            # The opposite plate is diagnostic final_test; not Tahoe response labels.
            batch = data.state_arrays(config["state_fold"], "final_test" if not data.synthetic else "test")
            with torch.no_grad():
                z = encode_vectors(reference, batch["x"], config["device"]).to(config["device"])
                metrics = state_metrics(classifier(z)/classifier.temperature,
                                        tensor(batch["labels"], config["device"], torch.long))
            report = {"evaluation": "TIDYI_opposite_plate_diagnostic", "metrics": metrics,
                      "state_fold": config["state_fold"],
                      "state_loss": saved_state["metadata"].get("state_loss"),
                      "state_aggregate_lambda": saved_state["metadata"].get("state_aggregate_lambda"),
                      "label_semantics": "original_TIDYI_sorting_labels_not_apoptosis_outcomes",
                      "new_blind_test": False, "cross_source_validated": False, "model_updates": 0}
        else:
            report = evaluate_decisions(data, config, pipeline, output)
        after = {path: file_hash(path) for path in paths}
        if before != after:
            raise RuntimeError("EVALUATION_MUTATED_FROZEN_ARTIFACT")
        report.update({"synthetic": config.get("synthetic", False), "data_identity": data.identity,
                       "frozen_hashes": after, "test_used_for_selection": False})
        atomic_json(output / "report.json", report)
        print(json.dumps({"status": "complete", "report": str(output / 'report.json'), **report},
                         ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}")
        return 1
