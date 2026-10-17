"""Small engine facade over the independently permissioned training reader."""

import numpy as np

from .config import ROOT, config_hash, project_path

ROLES = {"T": "fit", "V": "selection", "R": "residual_fit", "G": "aux_selection",
         "C": "calibration", "E": "final_test"}


class EngineData:
    synthetic = False

    def __init__(self, config, roles=("T", "V"), allow_test=False, require_preprocessing=True):
        from lung.data import TrainingData
        self.reader = TrainingData(ROOT, config_path=project_path(config["data"]),
                                   allowed_roles=tuple(ROLES[role] for role in roles),
                                   allow_test=allow_test, require_preprocessing=require_preprocessing,
                                   row_cache_options=config.get("data_loading", {}).get("row_cache"))
        self.manifest = self.reader.manifest
        self.identity = config_hash({"signature": self.manifest["signature"],
                                    "files": self.manifest["files"]})
        self.mu, self.scale = self.reader.mean, self.reader.scale
        self.neighbors = self.reader.neighbors
        self.n_genes = self.reader.n_genes
        self.assets_dir = self.reader.assets_path
        self.context_sizes = {"n_sources": len(self.reader.context_vocabulary["source_id"]),
                              "n_backgrounds": len(self.reader.context_vocabulary["background_id"]),
                              "n_protocols": len(self.reader.context_vocabulary["protocol_id"])}
        if "sciplex_source" in self.reader.config and "model" in config.get("response", {}):
            from lung.models import ModelConfig
            shape = ModelConfig(**config["response"]["model"])
            if any(getattr(shape, key) < size for key, size in self.context_sizes.items()):
                raise ValueError("MODEL_CONTEXT_CARDINALITY_MISMATCH: new source/protocol embeddings required")
        self.bandwidth = None
        if self.reader.bandwidths is not None:
            bandwidths = np.asarray(self.reader.bandwidths)
            if bandwidths.shape != (5,) or not np.allclose(bandwidths / bandwidths[2], [.25, .5, 1., 2., 4.]):
                raise ValueError("MMD_BANDWIDTH_ASSET_MISMATCH")
            self.bandwidth = float(bandwidths[2])

    def condition_ids(self, role):
        return self.reader.condition_ids(ROLES[role])

    def iter_response_epoch(self, role, seed):
        return self.reader.iter_response_epoch(ROLES[role], seed)

    @staticmethod
    def flatten(condition):
        return {"fingerprint": condition["fingerprint"], "dose": condition["dose_um"],
                "duration": condition["duration_h"], "source_id": condition["source_id"],
                "background_id": condition["background_id"], "protocol_id": condition["protocol_id"],
                "condition_id": condition["condition_id"], "group_id": condition["chemical_id"],
                "is_vehicle": condition["is_vehicle"], "condition": condition}

    def sample_response(self, condition_id, n, rng, role):
        result = self.reader.sample_response(condition_id, n=n, rng=rng, role=ROLES[role])
        return {**result, **self.flatten(result["condition"])}

    def sample_response_group(self, keys, cfg, rng, mmd_rng, role):
        batches = self.reader.sample_response_group(keys, cfg, rng, mmd_rng, role=ROLES[role])
        return [(tuple({**item, **self.flatten(item["condition"])} if item is not None else None
                       for item in pair)) for pair in batches]

    def action_features(self, condition_id):
        return self.flatten(self.reader.action_features(condition_id))

    def sample_controls(self, condition_id, n, rng, role):
        result = self.reader.sample_controls(condition_id, n=n, rng=rng, role=ROLES[role])
        return {**result, **self.flatten(result["condition"])}

    def state_arrays(self, fold, role):
        x, labels = self.reader.state_arrays(fold=fold, role=role)
        return {"x": x, "labels": labels}


def open_data(config, stage, prepare=True, test=False, progress=None):
    if prepare:
        from lung.data import prepare_training
        prepare_training(ROOT, config_path=project_path(config["data"]),
                         build_preprocessing=True, progress=progress, device=config['device'])
    permitted = {
        "response": ("T", "V"), "pool": ("T", "V"), "state": ("T", "V", "C"),
        "backward": ("R",), "correction": ("R", "G"),
        "endpoint": ("T", "V", "R", "G", "C"),
        "daac": ("T", "V"), "dlpa": ("T", "V"),
        "all": ("T", "V", "R", "G", "C")}
    return EngineData(config, ("E",) if test else permitted[stage], allow_test=test)
