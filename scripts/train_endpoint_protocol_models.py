"""Metadata only unless --train explicitly requests real G fitting."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from lung.engine.endpoint_protocol_features_v2 import EndpointProtocolFeatureInputsV2
from lung.engine.endpoint_protocol_learning import fit_g


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=ROOT / '08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1')
    p.add_argument('--evaluation-plan', type=Path)
    p.add_argument('--cache-manifest', type=Path)
    p.add_argument('--fold', default='viel_leave_group_01')
    p.add_argument('--branch', default='main')
    p.add_argument('--output', type=Path)
    p.add_argument('--train', action='store_true')
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
              'formal_training_ready': False,
              'registered_pipeline_missing_stages': registered_missing
                    if args.cache_manifest is None else [],
              'actual_fit_performed': False}
    if args.train and not missing:
        if args.output is None:
            p.error('--train requires a new --output checkpoint path')
        fit_g(bridge, plan, fold_id=args.fold, branch=args.branch,
              cache_manifest=args.cache_manifest, output=args.output)
        report = {**report, 'actual_fit_performed': True, 'model_execution_performed': True,
                  'status': 'G_fit_completed_biological_effectiveness_pending'}
    print(json.dumps(report, ensure_ascii=False))
    return 2 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())


