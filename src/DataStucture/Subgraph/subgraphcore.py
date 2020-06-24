"""This file is used to define node set in the simplified MindSpore graph."""
from collections import deque
from typing import Tuple, Set, Deque, Union

from DataStucture.SimpleMindsporeGraph.snode import SNode
from DataStucture.Subgraph.subgraph import Subgraph


class SubgraphCore(Subgraph):
    """Mainly used for the collection control in subgraph core growing"""

    def __init__(self, nodes: Union[Tuple[SNode], None]):
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
        core_nodes = [deque((n,)) for n in nodes]

        super().__init__(
            pattern=core_pattern,
            nodes=core_nodes,
            min_node=min(nodes),
            min_node_index=nodes.index(min(nodes)),
        )

        # Holds the index of all the boundary pattern items, only these items will be traversed
        self.boundary_pattern_index: Set[int] = set(range(len(self.pattern)))

        # Iter of set boundary_pattern_index, useful when executing traversal
        self.pointer = None

    def __next__(self) -> Tuple:
        """
        Traverse to get the equivalent-nodes-tuple in subgraph instance

        Notes:
            only these nodes whose id in boundary_nodes will be traversed
        Returns:
            equivalent_nodes tuple
        """
        index = self.pointer.__next__()
        return tuple(n[index] for n in self.nodes)

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

    def grow(
            self,
            node_pattern: Deque[str],
            grow_nodes: Deque[Deque[SNode]],
            keep_instance_index: Tuple[int],
    ):
        """"""
        # TODO:check if node_pattern, grow_nodes, keep_instance_index is valid

        # do some copy
        new_core = SubgraphCore(nodes=None)
        new_core.pattern = self.pattern.copy() + node_pattern

        # update the nodes and boundary patterns
        new_core.nodes = [
            self.nodes[index] + deque(grow_nodes[i])
            for i, index in enumerate(keep_instance_index)
        ]

        # mark the boundary(undetected) pattern
        new_core.boundary_pattern_index = set(
            range(len(self.pattern), len(new_core.pattern))
        )

        # clear the pointer and _id
        new_core.pointer = None
        new_core._id = 0

        # update the min_node info
        new_core.min_node = self.min_node
        new_core.min_node_index = self.min_node_index

        # update the minimum-id node info
        new_min_node_in_instance = tuple(min(n) for n in grow_nodes)
        new_min_node = min(new_min_node_in_instance)
        if new_min_node < new_core.min_node:
            new_core.min_node = new_min_node
            new_core.min_node_index = new_min_node_in_instance.index(new_min_node)

        return new_core

    def __contains__(self, node: SNode):
        return any([node in pattern_nodes for pattern_nodes in self.nodes])

    def commit(self):
        del self.pointer
        del self.boundary_pattern_index
        del self.min_node
        del self.min_node_index
        return self
