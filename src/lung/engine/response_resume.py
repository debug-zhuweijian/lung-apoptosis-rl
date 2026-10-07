"""Response checkpoints commit complete optimizer groups, never partial gradients."""

import math


def update_state(epoch, position, keys, aggregate, rng, mmd_rng, generator, total, warmup):
    return {"schema_version": 1, "epoch": epoch, "next_condition_index": position,
            "keys": list(keys), "aggregate": dict(aggregate),
            "numpy_rng": rng.bit_generator.state, "mmd_numpy_rng": mmd_rng.bit_generator.state,
            "mmd_torch_rng": generator.get_state(), "total_updates": total,
            "warmup_updates": warmup}


def validate_resume(saved, draws, group_size, total, warmup):
    epoch, step = saved["epoch"], saved["counters"]["optimizer_step"]
    updates = math.ceil(draws / group_size)
    state = saved.get("extra", {}).get("update_resume")
    position = 0
    if state is not None:
        position = state["next_condition_index"]
        if (state.get("schema_version") != 1 or state["epoch"] != epoch
                or not isinstance(position, int) or not 0 < position <= draws
                or (position != draws and position % group_size)
                or len(state["keys"]) != draws):
            raise ValueError("RESPONSE_RESUME_POSITION_MISMATCH")
    schedule = state or saved.get("extra", {}).get("schedule")
    if schedule and (schedule["total_updates"] != total or schedule["warmup_updates"] != warmup):
        raise ValueError("RESPONSE_RESUME_SCHEDULE_MISMATCH")
    if epoch < 0 or step != epoch * updates + math.ceil(position / group_size):
        raise ValueError("RESPONSE_RESUME_STEP_MISMATCH")
    if saved["counters"].get("condition_draw", epoch*draws + position) != epoch*draws + position:
        raise ValueError("RESPONSE_RESUME_POSITION_MISMATCH: condition counter")
    states = saved["optimizer"]["state"].values()
    actual_steps = [int(item["step"]) for item in states if "step" in item]
    if (step and not actual_steps) or any(value != step for value in actual_steps):
        raise ValueError("RESPONSE_RESUME_STEP_MISMATCH: optimizer state")
