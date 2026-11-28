"""Native HumanoidArena TWIST2 G1 Dex3 controller."""

import numpy as np
import onnxruntime as ort

from humano.wbc.base import MotorCommand, RobotState, WholeBodyController


def _bilateral(parts):
    return [f"{side}_{part}_joint" for side in ("left", "right") for part in parts]


JOINT_NAMES = (
    _bilateral(("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + _bilateral(
        (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )
)
# Arena's Dex3 articulation defaults center the policy output. The standing
# reference is the separate mimic target used by its live action provider.
_DEFAULT = np.array(
    [-0.20, 0, 0, 0.42, -0.23, 0] * 2
    + [0] * 3
    + [0.35, 0.18, 0, 0.87, 0, 0, 0]
    + [0.35, -0.18, 0, 0.87, 0, 0, 0],
    dtype=np.float32,
)
STAND_REFERENCE = np.array(
    [0, 0, 0.8, 0, 0, 0]
    + [-0.2, 0, 0, 0.4, -0.2, 0] * 2
    + [0] * 3
    + [0, 0.4, 0, 1.2, 0, 0, 0]
    + [0, -0.4, 0, 1.2, 0, 0, 0],
    dtype=np.float32,
)
_KP = np.array([130, 100, 100, 160, 60, 40] * 2 + [150] * 3 + ([40] * 4 + [20] * 3) * 2)
_KD = np.array([3, 3, 2, 4.5, 6.5, 2] * 2 + [4, 4.5, 8] + ([5] * 4 + [1] * 3) * 2)


class Twist2Controller(WholeBodyController):
    control_dt = 0.02

    def __init__(self, model_path, joint_names):
        names = list(joint_names)
        if len(names) != 29 or set(names) != set(JOINT_NAMES):
            raise ValueError("TWIST2 requires the 29 named G1 body joints")
        self.to_policy = [names.index(n) for n in JOINT_NAMES]
        self.to_robot = np.argsort(self.to_policy)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.model = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.model.get_inputs(), self.model.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].shape[1:] != [1432]
            or outputs[0].shape[1:] != [29]
        ):
            raise ValueError("Expected TWIST2 ONNX [batch,1432] -> [batch,29]")
        self.input_name = inputs[0].name
        self.reset()

    def reset(self):
        self.history = np.zeros((10, 127), dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)

    def step(self, robot_state: RobotState, motion_signal) -> MotorCommand:
        ref = np.asarray(motion_signal, dtype=np.float32)
        if ref.shape != (35,) or not np.isfinite(ref).all():
            raise ValueError("TWIST2 requires a finite 35-dimensional reference")
        q = np.asarray(robot_state.body_q, dtype=np.float32)[self.to_policy]
        dq = np.asarray(robot_state.body_dq, dtype=np.float32)[self.to_policy]
        dq[[4, 5, 10, 11]] = 0
        w, x, y, z = np.asarray(robot_state.base_quat, dtype=np.float32)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        current = np.r_[
            ref,
            np.asarray(robot_state.base_ang_vel, dtype=np.float32) * 0.25,
            roll,
            pitch,
            q - _DEFAULT,
            dq * 0.05,
            self.last_action,
        ].astype(np.float32)
        observation = np.concatenate((current, self.history.ravel(), ref)).clip(-100, 100)[None]
        raw = self.model.run(None, {self.input_name: observation})[0][0]
        self.history = np.concatenate((self.history[1:], current[None]))
        self.last_action = raw.copy()
        return MotorCommand(
            q_target=(raw * 0.5 + _DEFAULT)[self.to_robot],
            kp=_KP[self.to_robot],
            kd=_KD[self.to_robot],
        )
