"""This file is used to define node in the simplified MindSpore graph."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(init=True, order=True, eq=True, frozen=True)
class SNode:
    val: int
    type: str
    upstream: Tuple[int]
    downstream: Tuple[int]

    @property
    def id(self):
        return self.val

    def __repr__(self):
        return f"Node-{self.val} {self.type}"
