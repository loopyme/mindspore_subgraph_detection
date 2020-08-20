"""This file is used to define nodes in the simplified MindSpore graph."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(init=True, order=True, frozen=False, unsafe_hash=True)
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

    # Origin name of node, useful when dump the result
    name: str

    def __repr__(self):
        return f"Node-{self.id} {self.type}"
