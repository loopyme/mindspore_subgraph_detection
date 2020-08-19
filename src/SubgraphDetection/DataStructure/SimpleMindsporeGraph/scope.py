"""This file is used to define scopes (name_scope) in the simplified MindSpore graph."""
from dataclasses import dataclass
from typing import Tuple

from SubgraphDetection.DataStructure.SimpleMindsporeGraph.snode import SNode


@dataclass
class Scope(SNode):
    """A dataclass which stores all infos we need about scopes"""

    member: Tuple[int, ...]

    def __repr__(self):
        return f"Scope-{self.id} [{self.member}]"
