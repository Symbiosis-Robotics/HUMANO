from humano.policy._lerobot_policy import LerobotPolicyAgent


def ActAgent(robot, host, port, *, use_native_sonic=False, **kwargs):
    return LerobotPolicyAgent(robot, host, port, use_native_sonic=use_native_sonic, **kwargs)
