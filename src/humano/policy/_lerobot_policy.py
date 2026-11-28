from __future__ import annotations

from typing import Any

from humano.policy.sonic_policy import SonicPolicyAgent


def LerobotPolicyAgent(
    robot,
    host: str,
    port: int,
    *,
    use_native_sonic: bool = False,
    **kwargs: Any,
):
    """Route LeRobot SONIC-token policies through the shared SONIC-token bridge."""
    return SonicPolicyAgent(
        robot,
        host,
        port,
        use_native_sonic=use_native_sonic,
        **kwargs,
    )
