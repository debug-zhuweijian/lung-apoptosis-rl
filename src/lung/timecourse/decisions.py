"""Independent DA-AC and DLPA training and frozen, common-candidate comparison."""
from pathlib import Path
import csv
import json

import numpy as np
import torch

from lung.engine.checkpoint import atomic_json, atomic_save, load_checkpoint, rng_state, restore_rng
from lung.engine.config import config_hash, file_hash, implementation_identity
from lung.engine.decision import daac_settings, dlpa_settings
from lung.rl import (DAACTrainer, DLPAPlanner, ModelRollout, ProposalDistiller,
                     generate_fixed_sequences, reevaluate_fixed_sequences, stream_seed)
from lung.rl.networks import ConditionalActor, JointActor
from .decision_world import TimeDecisionWorld
from .decision_audit import audit_population
from .state_proxy import FrozenStateProxy


def _train(world, config, stage, seed, folder, resume=False, progress=None):
    from torch.utils.tensorboard import SummaryWriter
    folder.mkdir(parents=True, exist_ok=True)
    identity = config_hash({"versions": world.adapter.versions, "config":config,"stage":stage,"seed":seed,
                            "code":implementation_identity()})
    if stage == "daac":
        learner = DAACTrainer(world.registry,world.adapter.versions,daac_settings(config),config["device"],seed)
        planner = None
    else:
        learner = ProposalDistiller(world.registry,world.adapter.versions,config["device"],seed,
                                    config[stage].get("lr",1e-4))
        planner = DLPAPlanner(world.adapter,learner.proposal,dlpa_settings(config))
    path = folder / "last.pt"
    epoch, history = 0, []
    if resume and path.exists():
        saved = load_checkpoint(path,identity)
        learner.load_state_dict(saved["learner"])
        if planner:
            planner.load_state_dict(saved["planner"])
        epoch, history = saved["epoch"],saved["history"]
        if "rng" in saved:
            restore_rng(saved["rng"])
    actor = learner.actor if stage == "daac" else learner.proposal
    if epoch >= int(config.get("epochs",1000)):
        updates = min(learner.actor_step, learner.critic_step) if stage == "daac" else learner.distill_step
        if updates < 1:
            raise ValueError("NO_ACTUAL_POLICY_TRAINING_UPDATES")
        return actor, str(path.resolve()), {"updates":updates,"epoch":epoch,
                                           "actor_updates":learner.actor_step if stage == "daac" else learner.distill_step,
                                           "critic_updates":learner.critic_step if stage == "daac" else None,
                                           "parameter_delta_this_invocation":0.0}
    initial = {k:v.detach().clone() for k,v in actor.named_parameters()}
    def save():
        atomic_save(path,{"schema_version":1,"identity":identity,"epoch":epoch,"learner":learner.state_dict(),
            "model":actor.state_dict(),"rng":rng_state(),"planner":planner.state_dict() if planner else None,"history":history,
            "metadata":{"stage":stage,"role":"D_only","seed":seed,"config":config,
                        "environment":world.adapter.versions,"phi":"fixed_zero_single_step"}})
    save()
    events = folder/"events"
    if config.get("log_root") and config.get("log_anchor"):
        events = Path(config["log_root"])/folder.relative_to(Path(config["log_anchor"]).resolve())/"events"
    writer = SummaryWriter(str(events),purge_step=epoch)
    try:
        for epoch_index in range(epoch,int(config.get("epochs",1000))):
            metrics = []
            if stage == "daac":
                for unit in range(config[stage].get("episodes_per_epoch",128)):
                    env = ModelRollout(world.adapter,world.root,stream_seed(seed,epoch_index,unit),
                                       score_repeats=config.get("scoring_repeats",32))
                    action, p = learner.select(env.state,env.legal_mask)
                    transition = env.step(action,p,policy_step=learner.actor_step)
                    metrics.append({**learner.observe(transition),"base_reward":transition.base_reward})
            else:
                assert planner is not None
                learner.begin_epoch()
                for unit in range(config[stage].get("roots_per_epoch",64)):
                    learner.add(planner.plan(world.root,stream_seed(seed,epoch_index,unit)))
                for _ in range(config[stage].get("distill_updates",200)):
                    metrics.append(learner.update(config[stage].get("batch_size",256)))
            epoch = epoch_index+1
            numeric = {key:float(np.mean([m[key] for m in metrics if key in m]))
                for key in {k for m in metrics for k,v in m.items() if isinstance(v,(int,float))}}
            if not numeric or not all(np.isfinite(v) for v in numeric.values()):
                raise ValueError("INVALID_DECISION_TRAINING_METRICS")
            history.append({"epoch":epoch,**numeric})
            for key,value in numeric.items():
                writer.add_scalar(key,value,epoch)
            save()
            with (folder/"metrics.csv").open("w",newline="",encoding="utf-8") as out:
                fields = sorted({k for row in history for k in row})
                csv_writer = csv.DictWriter(out,fieldnames=fields)
                csv_writer.writeheader()
                csv_writer.writerows(history)
            if progress:
                progress(f"time_decision/{stage}/{world.board}/seed{seed}",epoch,config["epochs"])
    finally:
        writer.close()
    updates = min(learner.actor_step, learner.critic_step) if stage == "daac" else learner.distill_step
    if updates < 1:
        raise ValueError("NO_ACTUAL_POLICY_TRAINING_UPDATES")
    delta = sum(float((v.detach()-initial[k]).square().sum()) for k,v in actor.named_parameters())
    return actor, str(path.resolve()), {"updates":updates,"epoch":epoch,
        "actor_updates":learner.actor_step if stage == "daac" else learner.distill_step,
        "critic_updates":learner.critic_step if stage == "daac" else None,
        "parameter_delta_this_invocation":delta}


def _transfer(actor, registry, device):
    if type(actor) not in (ConditionalActor, JointActor):
        raise ValueError("UNSUPPORTED_POLICY_TRANSFER_CLASS")
    with torch.random.fork_rng(devices=[]):
        copied = type(actor)(registry).to(device)
    state = {k:v for k,v in actor.state_dict().items() if k not in {"action_features","stop_mask"}}
    missing, unexpected = copied.load_state_dict(state,strict=False)
    if set(missing) != {"action_features","stop_mask"} or unexpected:
        raise ValueError("POLICY_TRANSFER_PARAMETER_MISMATCH")
    return copied.eval().requires_grad_(False)


def _compare(world, actors, config, seed):
    plans = {"enumeration":tuple((a.action_id,) for a in world.registry.candidates)}
    common = reevaluate_fixed_sequences(world.adapter,world.root,plans["enumeration"],seed,
        repeats=config.get("audit_repeats",32),namespace="time-common-rerank")
    native = {}
    for name, actor in actors.items():
        actor = _transfer(actor,world.registry,config["device"])
        if name == "daac":
            plans[name] = generate_fixed_sequences(actor,world.adapter,world.root,
                config.get("evaluation_candidates",256),stream_seed(seed,"daac-proposal"))
            obs = world.adapter.observe(world.root,(),stream_seed(seed,"native-state"))
            with torch.no_grad():
                p = actor([obs],world.protocol.legal_mask(world.registry,())[None])[0]
            native[name] = world.registry.candidates[int(p.argmax())].action_id
        else:
            result = DLPAPlanner(world.adapter,actor,dlpa_settings(config)).plan(world.root,stream_seed(seed,"dlpa-proposal"))
            plans[name] = result.sequences
            native[name] = min(result.search_scores,key=lambda x:(-result.search_scores[x],x))[0]
    rows = []
    for name, candidates in plans.items():
        winner = min(candidates,key=lambda x:(-common.means[x],x))
        native_plan = (native.get(name,winner[0]),)
        audit = reevaluate_fixed_sequences(world.adapter,world.root,[winner,native_plan],seed,
            repeats=config.get("audit_repeats",32),namespace="time-independent-audit")
        rows.append({"algorithm":name,"native_winner":native_plan[0],"native_audit":audit.means[native_plan],
            "reranked_winner":winner[0],"audit":audit.means[winner],"audit_se":audit.standard_errors[winner],
            "model_enumeration_gap":max(common.means.values())-common.means[winner],
            "candidate_ids":[p[0] for p in candidates],"reselected_after_audit":False})
    return {"rows":rows,"enumeration":{p[0]:v for p,v in common.means.items()},
            "baseline":{"Vehicle":common.means[("STOP",)],"STOP":common.means[("STOP",)],"same_reference":True},
            "sampling_certificate":common.metadata["fixed_pool_bound_met"]}


def run_comparison(root, config, nested_artifacts, state_pipeline, output, resume=False, progress=None):
    from .nested import FrozenTimePredictor
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    settings = {**config,**config.get("time_decision",{})}
    settings.setdefault("epochs",1000)
    settings.setdefault("device","cpu")
    predictor = FrozenTimePredictor(root,config,nested_artifacts)
    proxy = FrozenStateProxy(predictor.data,state_pipeline,settings["device"])
    identity = config_hash({"code":implementation_identity(),"config":settings,"data":predictor.data.identity,
        "proxy":proxy.identity,"nested":{p:file_hash(p) for p in nested_artifacts["artifacts"]}})
    manifest_path = output/"identity.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        raise ValueError("DECISION_RESUME_IDENTITY_MISMATCH")
    atomic_json(manifest_path,{"identity":identity,"status":"running","roles":"D train; C never; E frozen report only"})
    table = predictor.data.conditions
    rows72 = table.loc[table.time_point.eq(72) & ~table.is_vehicle]
    boards = sorted(set(zip(rows72.replicate.astype(str),rows72.culture_plate.astype(str))))
    expected_boards = {("rep1","plate49"):{1.,10.}, ("rep1","plate50"):{.01,.1},
                       ("rep2","plate51"):{.1,1.}, ("rep2","plate52"):{.01,10.}}
    if not config.get("synthetic") and set(boards) != set(expected_boards):
        raise ValueError("FOUR_REGISTERED_72H_BOARDS_REQUIRED")
    seeds = settings.get("policy_seeds",[0,1,2,3,4])
    if not seeds or len(set(seeds)) != len(seeds) or settings["epochs"] < 1:
        raise ValueError("INVALID_POLICY_TRAINING_BUDGET")
    results, paths, trained, development_enumeration = [], [], [], []
    for board in boards:
        subset = rows72.loc[rows72.replicate.astype(str).eq(board[0]) & rows72.culture_plate.astype(str).eq(board[1])]
        ids = {role:subset.loc[subset.group_id.isin(predictor.data.drug_groups[role]),"condition_id"].astype(str).tolist()
               for role in ("D","E")}
        if not ids["D"] or not ids["E"]:
            raise ValueError("MISSING_BOARD_CANDIDATES")
        if not config.get("synthetic") and (len(ids["D"]) != 56 or len(ids["E"]) != 20):
            raise ValueError("EXPECTED_28_OR_10_DRUGS_TIMES_TWO_DOSES")
        if not config.get("synthetic"):
            for role in ("D","E"):
                for _,drug_rows in table.loc[ids[role]].groupby("group_id"):
                    if set(drug_rows.dose.astype(float)) != expected_boards[board]:
                        raise ValueError("REGISTERED_BOARD_DOSE_MISMATCH")
        if (table.loc[ids["D"]+ids["E"],"reference_n"] < 1).any():
            raise ValueError("MATCHED_VEHICLE_REQUIRED_NO_CROSS_BOARD_FALLBACK")
        train = TimeDecisionWorld(predictor,proxy,ids["D"],board,"T",identity,settings)
        audit_paths = audit_population(train,settings,output/f"{board[0]}_{board[1]}"/"population_audit",progress)
        enumeration = reevaluate_fixed_sequences(train.adapter,train.root,
            [(cid,) for cid in (*ids["D"],"STOP")],0,repeats=settings.get("audit_repeats",32),
            namespace="D-enumeration-report-only")
        development_enumeration.append({"board":board,"scores":{p[0]:v for p,v in enumeration.means.items()}})
        for seed in seeds:
            actors, counters = {}, {}
            for stage in ("daac","dlpa"):
                actor,path,counter = _train(train,settings,stage,seed,output/f"{board[0]}_{board[1]}"/f"seed_{seed}"/stage,resume,progress)
                paths.append(path)
                actors[stage],counters[stage] = actor,counter
            trained.append((board,ids,seed,actors,counters,audit_paths))
    # Every policy has frozen before any E model ranking or measured E lookup.
    for board,ids,seed,actors,counters,audit_paths in trained:
        world = TimeDecisionWorld(predictor,proxy,ids["E"],board,"E",identity,settings)
        world.enable_population_audit(*audit_paths)
        results.append({"board":board,"seed":seed,"training":counters,"comparison":_compare(world,actors,settings,seed)})
    e_data = predictor.data.view((),predictor.data.drug_groups["E"],"final_decision_report")
    measured = {}
    for _,ids,_,_,_,_ in trained:
        for cid in ids["E"]:
            if cid not in measured:
                measured[cid] = proxy(e_data.sample_target(cid,None,np.random.default_rng(0)))
    for row in results:
        scores = {cid:measured[cid] for cid in row["comparison"]["enumeration"] if cid != "STOP"}
        first = next(iter(scores))
        scores["STOP"] = proxy(e_data.sample_controls(first,None,np.random.default_rng(0))["control"])
        best = max(scores.values())
        row["measured_expression_proxy"] = scores
        row["measured_population_status"] = {cid:str(table.loc[cid,"status"]) for cid in scores if cid != "STOP"}
        for algorithm in row["comparison"]["rows"]:
            selected = algorithm["reranked_winner"]
            algorithm["measured_proxy_selection_gap"] = best-scores[selected]
            algorithm["measured_proxy_rank"] = 1+sum(value>scores[selected] for value in scores.values())
            algorithm["native_measured_proxy_selection_gap"] = best-scores[algorithm["native_winner"]]
    report = output/"comparison.json"
    atomic_json(report,{"identity":identity,"status":"complete","results":results,
        "development_enumeration":development_enumeration,
        "scope":"unvalidated_cross_source_state_proxy_not_apoptosis_rate","horizon":72,
        "potential":"zero:single-step-terminal-return-unchanged;critic-scales-not-comparable-to-old-Phi",
        "root":"matched_endpoint_Vehicle_population_not_same_cell_initial_state",
        "training_role":"D","calibration_reward_access":False,"E_target_access":"after_all_training_frozen"})
    artifacts = [str(p.resolve()) for p in output.rglob("*") if p.is_file()]
    if settings.get("log_root"):
        artifacts.extend(str(p.resolve()) for p in Path(settings["log_root"]).rglob("*") if p.is_file())
    atomic_json(manifest_path,{"identity":identity,"status":"complete","weights":{p:file_hash(p) for p in paths}})
    return {"status":"complete","report":str(report),"weights":paths,"artifacts":artifacts}
