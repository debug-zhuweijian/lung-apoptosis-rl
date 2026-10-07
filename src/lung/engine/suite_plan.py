"""Registered research graph; primary branches are fixed before evaluation."""


def task_graph(config):
    from .optimization import candidate_names
    primary = config.get("suite", {}).get("primary_state", "weighted_ce")
    if primary not in ("weighted_ce", "ce", "ce_aggregate"):
        raise ValueError("suite_primary_state must be weighted_ce, ce or ce_aggregate")
    if config.get("suite", {}).get("primary_response", "cfm_mmd") != "cfm_mmd":
        raise ValueError("The registered primary response is cfm_mmd")
    response = "response_cfm_mmd"
    state = "state_" + primary
    graph = {
        "prepare": [], "ridge": ["prepare"],
        response: ["prepare"], "response_cfm_only": ["prepare"],
        "response_condition_only": ["prepare"], "response_population_mean": ["prepare"],
        "response_random_product": ["prepare"],
        "response_compare": ["ridge", response, "response_cfm_only"],
        "response_final_test": ["ridge", response, "response_cfm_only", "response_condition_only",
                                "response_population_mean", "response_random_product"],
        "state_weighted_ce": [response], "state_ce": [response],
        "state_ce_aggregate": [response],
        "state_compare": ["state_weighted_ce", "state_ce", "state_ce_aggregate"],
        "tahoe_auxiliary": [response, state],
        "tahoe_decision": ["tahoe_auxiliary"],
        "time_prepare": ["prepare"], "time_nested": ["time_prepare"],
        "time_decision": ["time_nested", state, response],
    }
    candidates = ["response_" + name for name in candidate_names(config)]
    if candidates:
        before_compare = {}
        for name, dependencies in graph.items():
            if name == "response_compare":
                before_compare.update({candidate: ["prepare"] for candidate in candidates})
                before_compare["optimization_compare"] = [response, "response_cfm_only", *candidates]
            before_compare[name] = (dependencies + candidates if name == "response_final_test" else dependencies)
        graph = before_compare
    return {**graph, "summary": list(graph)}


RESUMABLE = frozenset({"response_cfm_mmd", "response_cfm_only", "response_condition_only",
                       "response_population_mean", "response_random_product", "state_weighted_ce",
                       "state_ce", "state_ce_aggregate", "tahoe_auxiliary",
                       "tahoe_decision", "time_nested", "time_decision",
                       "response_cfm_then_full", "response_one_step_then_full", "response_latent_flow"})
