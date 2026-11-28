"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from humano.core.asset import Asset
from humano.core.scene import TabletopScene


class ShowHouse(TabletopScene):
    def __init__(self, table: Asset) -> None:
        self.uid = "show_house"
        self.table = table
