"""This file is used to define nodes in the simplified MindSpore graph."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(init=True, order=False, eq=False, frozen=True)
class SNode:
    """A dataclass which stores all infos we need about nodes"""
    id: int
    type: str
    upstream: Tuple[int, ...]
    downstream: Tuple[int, ...]

    def __lt__(self, other):
        # negative node-id (non-normal-type nodes) should not lower than any normal nodes
        return 0 < self.id < other.id

    def __repr__(self):
        if self.id > 0:
            return f"Node-{self.id} {self.type}"
        else:
            return f"Node {self.type}"
