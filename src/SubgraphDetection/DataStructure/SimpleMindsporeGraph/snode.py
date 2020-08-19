"""This file is used to define nodes in the simplified MindSpore graph."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(init=True, order=False, eq=False, frozen=False)
class SNode:
    """A dataclass which stores all infos we need about nodes"""

    # Unique id of a node,
    # sometime we forged it (compare to the origin msgraph) to store infos about scope
    id: int

    # The type of node/scope,
    # sometimes it's generated randomly (Happened when DETAILED_ISOMORPHIC_CHECK is True)
    type: str

    # Upstream or downstream of a node, note that it's the leaf node id.
    upstream: Tuple[int, ...]
    downstream: Tuple[int, ...]

    # Father scope of a node, which may not be store as a scope object.
    scope: str

    # Due to special needs, the levels here are counted from bottom to top.
    # Level of one node/scope is the max of it's child node/scope level plus 1.
    # For example, the level of the leaf node is 1.
    level: int

    def __lt__(self, other):
        # negative node-id (non-normal-type nodes) should not lower than any normal nodes
        return 0 < self.id < other.id

    def __repr__(self):
        if self.id > 0:
            return f"Node-{self.id} {self.type}"
        else:
            return f"Node {self.type}"
