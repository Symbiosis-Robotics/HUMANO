"""Evaluate HTTP policies producing one native whole-body command per 20 ms control tick."""

import os
from collections import deque
from pathlib import Path

import numpy as np

from humano.policy.client import HttpActionClient
from humano.policy.server import validate_controller_chunk
from humano.policy.sonic_policy import NativeMotorBridge, _policy_state


class NativeCommandAgent(NativeMotorBridge):
    bootstrap_stabilize_steps = 0
    prefetch_warmup_steps = 0

    def check_step_dt(self, step_dt):
        if not np.isclose(step_dt, self.controller.control_dt):
            raise ValueError(
                f"{self.controller_name} requires a 20 ms environment step, got {step_dt}"
            )

    def __init__(self, robot, host, port):
        directory = os.environ.get(self.weights_env)
        if not directory:
            raise ValueError(f"Set {self.weights_env} to the {self.controller_name} model location")
        self.robot = robot
        self.weights_dir = Path(directory).expanduser().resolve()
        self.controller = self.controller_type(self.weights_dir, robot._body_joint_names)
        self.client = HttpActionClient(host, port)
        self.pending = deque()

    def reset(self, policy_seed=None):
        self.controller.reset()
        self.pending.clear()
        self.first_query = True
        self.seed = policy_seed

    def get_action(self, observation, *, info, instruction):
        dof = self.robot.hand_spec.dof
        if not self.pending:
            actions, *_ = self.client.query_action(
                {"ego_view": observation["head_stereo_left"]},
                instruction,
                _policy_state(self.robot, observation),
                {},
                history={"reset": True} if self.first_query else {},
                dataset=self.controller_name,
                inference_seed=self.seed,
            )
            self.pending.extend(validate_controller_chunk(actions, dof, self.controller_name))
            self.first_query = False
        row = self.pending.popleft()
        cmd = self.controller.step(self._extract_robot_state(), row[: self.body_width])
        if len(row) == self.body_width:
            return self._motor_cmd_to_action(cmd)
        return self._motor_cmd_to_action(
            cmd, row[self.body_width : self.body_width + dof], row[self.body_width + dof :]
        )

    def close(self):
        self.pending.clear()
