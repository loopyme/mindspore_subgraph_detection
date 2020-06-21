"""This file is used to define node set in the simplified MindSpore graph."""
from collections import deque
from typing import Tuple, Set, Deque

from DataStucture.SimpleMindsporeGraph.snode import SNode
from config import SAFE_MODE


class SubgraphCore:
    """Mainly used for the collection control in subgraph core growing"""

    def __init__(self, nodes: Tuple[SNode] = None):
        """
        Init a SubgraphCore with a tuple of Snode.

        Args:
            nodes : The initial nodes, which should be all in same type
        """
        if nodes is None:
            # return null obj
            return
        if SAFE_MODE and not all([n.type == nodes[0].type for n in nodes]):
            raise ValueError("Subgraph Core Nodes should be in the same pattern")

        # The pattern of the subgraph, correspond to the nodes
        self.pattern: Deque[str] = deque[nodes[0].type]

        # Nodes the make up the subgraph, each tuple hold one place in the pattern
        self.nodes: Deque[Tuple[SNode]] = deque[nodes]

        # Holds the index of all the boundary pattern items, only these items will be traversed
        self.boundary_pattern_index: Set[int] = set(range(len(self.pattern)))

        # Iter of set boundary_pattern_index, useful when executing traversal
        self.pointer = None

        # Unique id of SubgraphCore, used to avoid additional calculations
        # Lazy computed with property-id
        self._id: int = 0

        # Record the minimum-id node info to speed up SubgraphCore id calculation
        self.min_node_id = min(nodes)
        self.min_node_index = (0, nodes.index(self.min_node_id))

    def __copy__(self):
        copy_obj = SubgraphCore()
        copy_obj.pattern = self.pattern.copy()
        copy_obj.nodes = self.nodes.copy()
        copy_obj.boundary_pattern_index = self.boundary_pattern_index.copy()
        copy_obj.pointer = None
        copy_obj._id = self.id
        copy_obj.min_node_id = self.min_node_id
        copy_obj.min_node_index = self.min_node_index

    @property
    def id(self):
        """
        Get a unique id of SubgraphCore

        Returns: Hash of a string, which make up by the subgraph instance nodes that contains the smallest id node,
            nodes are numbered in ascending order, and split by '-'

        """
        if self._id == 0:
            self._id = hash(
                "-".join(sorted(tup[self.min_node_index[1]] for tup in self.nodes))
            )
        return self._id

    def __next__(self):
        return self.nodes[self.pointer.__next__()]

    def __iter__(self):
        # only these nodes whose id in boundary_nodes will be traversed
        self.pointer = self.boundary_pattern_index.__iter__()
        return self

    def get_iter_of_all(self):
        # force to a iter of all nodes
        self.pointer = self.boundary_pattern_index.__iter__()
        return self

    def set_interior(self, pattern_index: int):
        self.boundary_pattern_index.remove(pattern_index)

    def set_boundary(self, pattern_index: int):
        self.boundary_pattern_index.add(pattern_index)

    def __getitem__(self, node_id: int):
        return self.nodes[node_id]

    def grow(self, node_pattern: str, nodes: Tuple[SNode]):
        """
        Make the core grow

        Args:
            node_pattern: what type of node are growed
            nodes: growed nodes

        Returns: None
        """

        # add the node_pattern and nodes
        self.pattern.append(node_pattern)
        self.nodes.append(nodes)
        self.set_boundary(len(self.pattern))

        # clear the id cache
        self._id = 0

        # update the minimum-id node info
        new_min_node_id = min(nodes)
        if new_min_node_id < self.min_node_id:
            self.min_node_id = new_min_node_id
            self.min_node_index = (len(self.pattern) - 1, nodes.index(self.min_node_id))
