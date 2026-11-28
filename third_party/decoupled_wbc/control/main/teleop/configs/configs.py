from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Optional

import yaml

import decoupled_wbc
from decoupled_wbc.control.main.config_template import ArgsConfig as ArgsConfigTemplate
from decoupled_wbc.control.policy.wbc_policy_factory import WBC_VERSIONS
from decoupled_wbc.control.utils.network_utils import resolve_interface


def override_wbc_config(
    wbc_config: dict, config: "BaseConfig", missed_keys_only: bool = False
) -> dict:
    """Override WBC YAML values with dataclass values.

    Args:
        wbc_config: The loaded WBC YAML configuration dictionary
        config: The BaseConfig dataclass instance with override values
        missed_keys_only: If True, only add keys that don't exist in wbc_config.
                         If False, validate all keys exist and override all.

    Returns:
        Updated wbc_config dictionary with overridden values

    Raises:
        KeyError: If any required keys are missing from the WBC YAML configuration
                  (only when missed_keys_only=False)
    """
    # Override yaml values with dataclass values
    key_to_value = {
        "INTERFACE": config.interface,
        "ENV_TYPE": config.env_type,
        "VERSION": config.wbc_version,
        "SIMULATOR": config.simulator,
        "SIMULATE_DT": 1 / float(config.sim_frequency),
        "ENABLE_OFFSCREEN": config.enable_offscreen,
        "ENABLE_ONSCREEN": config.enable_onscreen,
        "model_path": config.wbc_model_path,
        "enable_waist": config.enable_waist,
        "with_hands": config.with_hands,
        "verbose": config.verbose,
        "verbose_timing": config.verbose_timing,
        "upper_body_max_joint_speed": config.upper_body_joint_speed,
        "keyboard_dispatcher_type": config.keyboard_dispatcher_type,
        "enable_gravity_compensation": config.enable_gravity_compensation,
        "gravity_compensation_joints": config.gravity_compensation_joints,
        "high_elbow_pose": config.high_elbow_pose,
    }

    if missed_keys_only:
        # Only add keys that don't exist in wbc_config
        for key in key_to_value:
            if key not in wbc_config:
                wbc_config[key] = key_to_value[key]
    else:
        # Set all keys (overwrite existing)
        for key in key_to_value:
            wbc_config[key] = key_to_value[key]

    # g1 kp, kd, sim2real gap
    if config.env_type == "real":
        # update waist pitch damping, index 14
        wbc_config["MOTOR_KD"][14] = wbc_config["MOTOR_KD"][14] - 10

    return wbc_config


@dataclass
class BaseConfig(ArgsConfigTemplate):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: Literal[tuple(WBC_VERSIONS)] = "gear_wbc"
    """Version of the whole body controller."""

    wbc_model_path: str = (
        "policy/GR00T-WholeBodyControl-Balance.onnx," "policy/GR00T-WholeBodyControl-Walk.onnx"
    )
    """Path to WBC model file (relative to decoupled_wbc/sim2mujoco/resources/robots/g1)"""
    """gear_wbc model path: policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"""

    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    """Whole body policy class."""

    # System Configuration
    interface: str = "sim"
    """Interface to use for the control loop. [sim, real, lo, enxe8ea6a9c4e09]"""

    simulator: str = "mujoco"
    """Simulator to use."""

    sim_sync_mode: bool = False
    """Whether to run the control loop in sync mode."""

    control_frequency: int = 50
    """Frequency of the control loop."""

    sim_frequency: int = 200
    """Frequency of the simulation loop."""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""

    with_hands: bool = True
    """Enable hand functionality. When False, robot operates without hands."""

    high_elbow_pose: bool = False
    """Enable high elbow pose configuration for default joint positions."""

    verbose: bool = True
    """Whether to print verbose output."""

    # Additional common fields
    enable_offscreen: bool = False
    """Whether to enable offscreen rendering."""

    enable_onscreen: bool = True
    """Whether to enable onscreen rendering."""

    upper_body_joint_speed: float = 1000
    """Upper body joint speed."""

    env_name: str = "default"
    """Environment name."""

    ik_indicator: bool = False
    """Whether to draw IK indicators."""

    verbose_timing: bool = False
    """Enable verbose timing output every iteration."""

    keyboard_dispatcher_type: str = "raw"
    """Keyboard dispatcher to use. [raw, ros]"""

    messaging_backend: str = "ros2"
    """Messaging backend. Options: ros2, zmq."""

    zmq_host: str = "localhost"
    """ZMQ host for subscriber connections (zmq backend only)."""

    # Gravity Compensation Configuration
    enable_gravity_compensation: bool = False
    """Enable gravity compensation using pinocchio dynamics."""

    gravity_compensation_joints: Optional[list[str]] = None
    """Joint groups to apply gravity compensation to (e.g., ['arms', 'left_arm', 'right_arm'])."""
    # Teleop/Device Configuration
    body_control_device: str = "dummy"
    """Device to use for body control. Options: dummy, vive, iphone, leapmotion, joycon."""

    hand_control_device: Optional[str] = "dummy"
    """Device to use for hand control. Options: None, manus, joycon, iphone."""

    body_streamer_ip: str = "10.112.210.229"
    """IP address for body streamer (vive only)."""

    body_streamer_keyword: str = "knee"
    """Body streamer keyword (vive only)."""

    enable_visualization: bool = False
    """Whether to enable visualization."""

    enable_real_device: bool = True
    """Whether to enable real device."""

    teleop_frequency: int = 20
    """Teleoperation frequency (Hz)."""

    # Deployment/Camera Configuration
    robot_ip: str = "192.168.123.164"
    """Robot IP address"""

    root_output_dir: str = "outputs"
    """Root output directory"""

    # Policy settings
    enable_upper_body_operation: bool = True
    """Enable upper body operation"""

    upper_body_operation_mode: Literal["teleop", "inference"] = "teleop"
    """Upper body operation mode"""

    def __post_init__(self):
        # Resolve interface (handles sim/real shortcuts, platform differences, and error handling)
        self.interface, self.env_type = resolve_interface(self.interface)

    def load_wbc_yaml(self) -> dict:
        """Load and merge wbc yaml with dataclass overrides"""
        # Get the base path to decoupled_wbc and convert to Path object
        package_path = Path(os.path.dirname(decoupled_wbc.__file__))

        if self.wbc_version == "gear_wbc":
            config_path = str(package_path / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml")
        else:
            raise ValueError(
                f"Invalid wbc_version: {self.wbc_version}, please use one of: " f"gear_wbc"
            )

        with open(config_path) as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)

        # Override yaml values with dataclass values
        wbc_config = override_wbc_config(wbc_config, self)

        return wbc_config


@dataclass
class ControlLoopConfig(BaseConfig):
    """Config for running the G1 control loop."""

    pass


@dataclass
class ComposedCameraClientConfig:
    """Config for running the composed camera client."""

    camera_port: int = 5555
    """Port number"""

    camera_host: str = "localhost"
    """Host IP address"""

    fps: float = 20.0
    """FPS of the camera viewer"""


@dataclass
class SimLoopConfig(BaseConfig):
    """Config for running the simulation loop."""

    mp_start_method: str = "spawn"
    """Multiprocessing start method"""

    enable_image_publish: bool = False
    """Enable image publishing in simulation"""

    camera_port: int = 5555
    """Camera port for image publishing"""

    verbose: bool = False
    """Verbose output, override the base config verbose"""
