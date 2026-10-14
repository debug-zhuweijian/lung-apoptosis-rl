"""Default data-only full-P preflight; explicit future --fit-ae or --fit.

Every real model operation still needs the user's separate model-run authority.
Preparation never activates this authority or resumes any legacy experiment.
"""
import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(directory))

from lung.engine import full_protocol_response as producer
from lung.engine import full_protocol_response_learning as learning
from lung.engine.checkpoint import atomic_json
from lung.engine.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modular_research.yaml")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit-ae", action="store_true")
    mode.add_argument("--fit", action="store_true")
    parser.add_argument("--ae-run", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and not (args.fit or args.fit_ae):
        parser.error("--resume needs an explicit fitting mode")
    if (args.fit or args.fit_ae) and args.output is None:
        parser.error("Explicit fitting requires an independent experiment output")
    if args.fit and args.ae_run is None:
        parser.error("--fit requires a completed --fit-ae run")
    output = ((ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()) if args.output else None
    if output is not None and not output.is_relative_to(ROOT / "experiments"):
        parser.error("Output must remain in this project's experiments directory")
    config = load_config(args.config)
    if args.fit_ae:
        result = learning.fit_matching_ae(config, output, resume=args.resume)
    else:
        data = learning.NativeProtocolData(config)
        settings = learning.settings_from_config(config)
        report = learning.preflight(data, settings, read_samples=not args.fit)
        if args.fit:
            ae_run = (ROOT / args.ae_run).resolve()
            learning.require(ae_run.is_relative_to(ROOT / "experiments"), "AE_RUN_PROJECT_BOUNDARY")
            declaration = json.loads((ae_run / "AE_result.json").read_text(encoding="utf-8"))
            plan = {"config": learning.joint_config(config), "bindings": report["bindings"],
                "protocol_identity": data.protocol_identity, "execution_dependencies": learning.execution_dependencies(),
                "AE_declaration": declaration, "settings": settings}
            assets = output.with_name(output.name + "-assets")
            # This equality is checked before the first real PT or model access.
            learning._new_or_resume(assets, plan, args.resume)
            if args.resume:
                completed = learning.completed_response_before_cache(output, data, settings, report["bindings"], assets)
                if completed is not None:
                    print(json.dumps(completed, ensure_ascii=False, indent=2, allow_nan=False))
                    return 0
            shared, ae_path, _ = learning.export_matching_ae(declaration["weights"], data, assets, report["bindings"], config)
            shared.to(settings["device"]).eval().requires_grad_(False)
            from lung.engine.representation_cache import ensure_representation_cache, load_representation_cache, encoder_identity
            directory, _ = ensure_representation_cache(data.data, shared, assets / "latent_cache", settings["device"])
            cache = load_representation_cache(directory, settings["device"], expected_data_identity=data.identity,
                expected_encoder=encoder_identity(shared, shared.config))
            result = learning.fit_response(data, shared, settings, output, bindings=report["bindings"],
                representation_path=ae_path, cache=cache, resume=args.resume)
            representation = json.loads((assets / "representation_declaration.json").read_text(encoding="utf-8"))
            atomic_json(output / "model_declaration.json", {"bindings": report["bindings"],
                "representation": representation, "response": {"stage": producer.STAGE, "synthetic": False,
                    "data_identity": data.identity, "checkpoint": result["weights"],
                    "checkpoint_sha256": producer.inputs.sha(result["weights"]),
                    "optimizer_step": result["selected_optimizer_step"],
                    "representation_hash": representation["checkpoint_sha256"],
                    "training_protocol_coverage": learning.COVERAGE,
                    "reference_contract": result["reference_contract"]}, "registered_audit_id": None})
        else:
            result = report
            if output is not None:
                learning.require(not output.exists(), "NEW_PREFLIGHT_OUTPUT_REQUIRED")
                output.mkdir(parents=True)
                atomic_json(output / "preflight.json", report)
                (output / "matching_joint_AE.yaml").write_text(yaml.safe_dump(learning.joint_ae_config(config),
                    allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("bindings", "execution_dependencies")},
        ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
