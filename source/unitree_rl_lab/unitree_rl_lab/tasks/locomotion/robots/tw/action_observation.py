"""Opt-in issue #8: change only policy last-action observation semantics."""
def configure_action_observation(env_cfg, profile):
    if profile == "requested":
        return
    if profile != "applied":
        raise ValueError(profile)
    env_cfg.observations.policy.last_action.params["action_name"] = "JointPositionAction"
