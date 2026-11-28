"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from humano.core.controller import Controller, ControllerCfg


class BaseController(Controller):
    def __init__(self, controller_cfg):
        super().__init__(controller_cfg)


class BaseControllerCfg(ControllerCfg):
    clazz: type[Controller] = BaseController
