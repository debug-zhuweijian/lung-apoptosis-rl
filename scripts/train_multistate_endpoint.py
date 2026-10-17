"""Metadata-only multi-state endpoint preflight. No fitting CLI in this version."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from lung.data.martin_state_function import DOMAIN_IDS, MartinStateFunctionInputs
from lung.data.multistate_role_plan import MultiStateRolePlan
from lung.engine.multistate_endpoint_model import preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / '08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/bridge_v2')
    parser.add_argument('--role-plan', type=Path, default=ROOT / 'experiments/20261011-data-alignment-round17/multistate-role-plan-v1')
    parser.add_argument('--fold', default='cycle_final:HCC2935')
    parser.add_argument('--head', choices=DOMAIN_IDS, default=DOMAIN_IDS[0])
    parser.add_argument('--native-cache-index', type=Path)
    args = parser.parse_args(argv)
    source = MartinStateFunctionInputs(args.source, project_root=ROOT)
    plan = MultiStateRolePlan(args.role_plan, source)
    result = preflight(source, plan, fold_id=args.fold, head_id=args.head,
                       native_cache_index=args.native_cache_index)
    print(json.dumps(result, ensure_ascii=False))
    return 2  # Valid metadata is explicitly still a pending scientific chain.


if __name__ == '__main__':
    raise SystemExit(main())
