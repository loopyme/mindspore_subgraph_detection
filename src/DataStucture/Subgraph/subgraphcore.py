"""This file is used to define node set in the simplified MindSpore graph."""
from collections import deque
from typing import Tuple, Set

from DataStucture.SimpleMindsporeGraph.snode import SNode
from DataStucture.Subgraph.subgraph import Subgraph


class SubgraphCore(Subgraph):
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

        core_pattern = deque()
        core_pattern.append(nodes[0].type)
        core_nodes = deque()
        core_nodes.append(nodes)

        super().__init__(
            pattern=core_pattern,
            nodes=core_nodes,
            min_node_id=min(nodes),
            min_node_index=(0, nodes.index(min(nodes))),
        )

        # Holds the index of all the boundary pattern items, only these items will be traversed
        self.boundary_pattern_index: Set[int] = set(range(len(self.pattern)))

        # Iter of set boundary_pattern_index, useful when executing traversal
        self.pointer = None

    def __copy__(self):
        copy_obj = SubgraphCore()
        copy_obj.pattern = self.pattern.copy()
        copy_obj.nodes = self.nodes.copy()
        copy_obj.boundary_pattern_index = self.boundary_pattern_index.copy()
        copy_obj.pointer = None
        copy_obj._id = self.id
        copy_obj.min_node_id = self.min_node_id
        copy_obj.min_node_index = self.min_node_index
        return copy_obj

    def __next__(self):
        """
        Traverse to get the equivalent-nodes-tuple in subgraph instance

        Notes:
            only these nodes whose id in boundary_nodes will be traversed
        Returns:
            equivalent_nodes tuple
        """
        return self.nodes[self.pointer.__next__()]

    def __iter__(self):
        """
        Get a iter of the equivalent nodes in subgraph instance

        Returns:
            self
        """
        self.pointer = self.boundary_pattern_index.__iter__()
        return self

    def set_interior(self, pattern_index: int):
        self.boundary_pattern_index.remove(pattern_index)

    def set_boundary(self, pattern_index: int):
        self.boundary_pattern_index.add(pattern_index)

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

    def __contains__(self, node: SNode):
        return any([node in pattern_nodes for pattern_nodes in self.nodes])
