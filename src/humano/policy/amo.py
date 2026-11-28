from humano.policy.native_command import NativeCommandAgent
from humano.wbc.amo import AmoController


class AmoAgent(NativeCommandAgent):
    controller_name = "amo"
    controller_type = AmoController
    weights_env = "HUMANO_AMO_WEIGHTS_DIR"
    body_width = 22
