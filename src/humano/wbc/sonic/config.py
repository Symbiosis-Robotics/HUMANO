"""SONIC simulation configuration for evaluation scene preparation."""


def load_sonic_config(hand_spec=None) -> dict:
    import tyro
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = tyro.cli(
        SimLoopConfig,
        config=(tyro.conf.ConsolidateSubcommandArgs,),
        args=[],
    )
    if hand_spec is not None:
        config.wbc_version = hand_spec.wbc_version
    sonic_config = config.load_wbc_yaml()
    if hand_spec is not None:
        sonic_config["WBC_VERSION"] = hand_spec.wbc_version
        if hand_spec.physics_dt is not None:
            sonic_config["SIMULATE_DT"] = hand_spec.physics_dt
    sonic_config["ENV_NAME"] = "humano"

    sonic_config["ENABLE_ELASTIC_BAND"] = False
    sonic_config["HOLD_STAND_UNTIL_SONIC_START"] = True
    sonic_config["INITIALIZE_GROUND_STANDING_POSE_BEFORE_VIEWER"] = True
    sonic_config.setdefault("ELASTIC_BAND_Z", 1.0)

    return sonic_config
