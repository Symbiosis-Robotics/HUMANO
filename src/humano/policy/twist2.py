from humano.policy.native_command import NativeCommandAgent
from humano.wbc.twist2 import Twist2Controller


class Twist2Agent(NativeCommandAgent):
    controller_name = "twist2"
    controller_type = Twist2Controller
    weights_env = "HUMANO_TWIST2_MODEL_PATH"
    body_width = 35
