"""Registered drug groups and immutable time-completion rules."""
from lung.research.sciplex_diagnostics import D28, C9, E10
from lung.engine.config import config_hash

GROUPS = {"D": list(D28), "C": list(C9), "E": list(E10)}
VEHICLE = "S0000"
RULES = {"version": "timecourse-v1", "groups": GROUPS,
         "known": "registered47_and_vehicle24", "target": "explicit_drugs72",
         "reference": "same_time_replicate_culture_plate_vehicle",
         "normalization": "log1p10000_common_axis_cell_weighted_known24_only",
         "graph": "exact_absolute_pearson_known24_only", "paired_cells": False,
         "bandwidth": "median_positive_pair_mean_squared_distance_known24_only",
         "outer": [list(D28[i:i+7]) for i in range(0, 28, 7)],
         "checkpoint_selection": "fixed_final_epoch"}
PROTOCOL_ID = config_hash(RULES)


def settings(config):
    return config.get("timecourse", config)
