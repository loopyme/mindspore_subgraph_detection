"""This file is used to define node in the simplified MindSpore graph."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(init=True, order=False, eq=False, frozen=True)
class SNode:
    id: int
    type: str
    upstream: Tuple[int]
    downstream: Tuple[int]

    def __lt__(self, other):
        # negative node-id should not lower than any normal nodes
        return 0 < self.id < other.id

    def __gt__(self, other):
        # negative node-id should larger than normal nodes
        return self.id < 0 or self.id > other.id

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        if self.id > 0:
            return f"Node-{self.id} {self.type}"
        else:
            return f"Node {self.type}"
