"""Metadata only unless --train explicitly requests real G fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from lung.engine.endpoint_protocol_features_v2 import EndpointProtocolFeatureInputsV2
from lung.engine.endpoint_protocol_learning import (fit_g, terminal_q_preflight, generate_terminal_decisions, fit_terminal_q, validate_reward)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=ROOT / '08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1')
    p.add_argument('--evaluation-plan', type=Path)
    p.add_argument('--cache-manifest', type=Path)
    p.add_argument('--fold', default='viel_leave_group_01')
    p.add_argument('--branch', default='main')
    p.add_argument('--output', type=Path)
    p.add_argument('--train', action='store_true')
    p.add_argument('--validate-reward', action='store_true')
    p.add_argument('--reward-rule', type=Path)
    p.add_argument('--kind', choices=('G', 'Q'), default='G')
    p.add_argument('--g-checkpoint', type=Path)
    p.add_argument('--reward-qualification', type=Path)
    p.add_argument('--decision-output', type=Path)
    args = p.parse_args(argv)
    bridge = EndpointProtocolFeatureInputsV2.load(args.inputs)
    missing = []
    plan = None
    if args.evaluation_plan is None:
        missing.append('explicit_observation_role_contract')
    else:
        from lung.data.endpoint_protocol_evaluation import EndpointProtocolEvaluationPlan
        plan = EndpointProtocolEvaluationPlan.load(args.evaluation_plan)
        plan.metadata_units(args.fold, branch=args.branch, role='fit')
    if args.cache_manifest is None or not args.cache_manifest.is_file():
        missing.append('audited_real_standard_P_cache')
    else:
        from lung.engine.endpoint_protocol_producer import load_standard_cache
        load_standard_cache(args.cache_manifest, bridge)
    pipeline_path = ROOT / 'experiments/20261010-endpoint-feature-preflight-joint/pipeline.json'
    pipeline = json.loads(pipeline_path.read_text(encoding='utf-8'))['pipeline']
    registered_missing = [k for k in ('response', 'pool', 'state', 'representation') if not pipeline.get(k)]
    report = {'status': 'pending' if missing else 'ready_for_explicit_G_baseline_fit',
              'missing': missing, 'model_execution_performed': False,
              'formal_training_ready': False, 'private_Q': terminal_q_preflight(),
              'registered_pipeline_missing_stages': registered_missing
                    if args.cache_manifest is None else [],
              'actual_fit_performed': False}
    if args.kind == 'Q' and (args.g_checkpoint is None or args.reward_qualification is None):
        missing.append('qualified_frozen_G_reward_artifacts')
        report = {**report, 'status': 'pending', 'missing': missing}
    if args.validate_reward and args.train:
        p.error('Choose --validate-reward or --train separately')
    if args.validate_reward and not missing:
        if args.reward_rule is None or args.g_checkpoint is None or args.output is None:
            p.error('--validate-reward requires --reward-rule --g-checkpoint --output')
        qualification = validate_reward(bridge, plan, fold_id=args.fold, branch=args.branch,
            cache_manifest=args.cache_manifest, g_path=args.g_checkpoint,
            rule_path=args.reward_rule, output=args.output)
        report = {**report, 'model_execution_performed': True,
            'status': 'selection_validation_completed',
            'measured_validation_passed': qualification['measured_validation_passed']}
    if args.train and not missing:
        if args.output is None:
            p.error('--train requires a new --output checkpoint path')
        if args.kind == 'G':
            fit_g(bridge, plan, fold_id=args.fold, branch=args.branch,
                  cache_manifest=args.cache_manifest, output=args.output)
        else:
            if args.decision_output is None:
                p.error('Q training requires new --decision-output')
            generate_terminal_decisions(bridge, plan, fold_id=args.fold, branch=args.branch,
                cache_manifest=args.cache_manifest, g_path=args.g_checkpoint,
                qualification_path=args.reward_qualification, output=args.decision_output)
            fit_terminal_q(args.decision_output, bridge=bridge, plan=plan, fold_id=args.fold,
                branch=args.branch, cache_manifest=args.cache_manifest, output=args.output)
        report = {**report, 'actual_fit_performed': True, 'model_execution_performed': True,
                  'status': args.kind + '_fit_completed_biological_effectiveness_pending'}
    print(json.dumps(report, ensure_ascii=False))
    return 2 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())


