"""This file is used to define subgraph core in the simplified MindSpore graph."""
from collections import deque
from typing import Tuple, Set, Deque, Union

from SubgraphDetection.DataStructure import SNode, Subgraph
from SubgraphDetection.config import MIN_SUBGRAPH_INSTANCE_NUMBER, MIN_SUBGRAPH_NODE_NUMBER


class SubgraphCore(Subgraph):
    """The growing core of subgraph"""

    def __init__(self, nodes: Union[Tuple[SNode, ...], None]):
        """
        Init a SubgraphCore with a tuple of Snode.

        Args:
            nodes : The initial nodes, which should be all in same type
        """
        if nodes is None:
            # return null obj
            return

        core_pattern: Deque[str] = deque()
        core_pattern.append(nodes[0].type)
        core_nodes = [deque((n,)) for n in nodes]

        super().__init__(
            pattern=core_pattern,
            nodes=core_nodes,
            min_node=min(nodes),
            min_node_index=nodes.index(min(nodes)),
        )

        # Holds the index of all the boundary pattern items, only these items will be traversed
        self.__boundary_pattern_index: Set[int] = set(range(len(self._pattern)))

        # Iter of set boundary_pattern_index, useful when executing traversal
        self.__pointer = None

    def __next__(self) -> Tuple:
        """
        Traverse to get the equivalent-nodes-tuple in subgraph instance

        Notes:
            only these nodes whose id in boundary_nodes will be traversed

        Returns:
            equivalent nodes tuple
        """
        index = self.__pointer.__next__()
        return tuple(n[index] for n in self._nodes)

    def __iter__(self):
        """
        Get a iter of the equivalent nodes in subgraph instance

        Returns:
            self
        """
        self.__pointer = self.__boundary_pattern_index.__iter__()
        return self

    def grow(
            self,
            node_pattern: Deque[str],
            grow_nodes: Deque[Deque[SNode]],
            keep_instance_index: Tuple[int, ...],
    ):
        """
        Let the core grow

        Args:
            node_pattern: The type of new nodes
            grow_nodes: The new nodes
            keep_instance_index: Which instances is going to keep

        Returns:
            The new core grow from self
        """
        # TODO:check if node_pattern, grow_nodes, keep_instance_index is valid

        # do some copy
        new_core = SubgraphCore(nodes=None)
        new_core._pattern = self._pattern.copy() + node_pattern

        # update the nodes and boundary patterns
        new_core._nodes = [
            self._nodes[index] + deque(grow_nodes[i])
            for i, index in enumerate(keep_instance_index)
        ]

        # mark the boundary(undetected) pattern
        new_core.__boundary_pattern_index = set(
            range(len(self._pattern), len(new_core._pattern))
        )

        # clear the pointer and _id
        new_core.__pointer = None
        new_core._id = 0

        # update the min_node info
        new_core._min_node = self._min_node
        new_core._min_node_index = self._min_node_index

        # update the minimum-id node info
        new_min_node_in_instance = tuple(min(n) for n in grow_nodes)
        new_min_node = min(new_min_node_in_instance)
        if new_min_node < new_core._min_node:
            new_core._min_node = new_min_node
            new_core._min_node_index = new_min_node_in_instance.index(new_min_node)

        return new_core

    @property
    def is_valid_for_commit(self):
        """Check whether if self is valid subgraph"""
        return len(self._nodes) >= MIN_SUBGRAPH_INSTANCE_NUMBER and len(self._nodes[0]) >= MIN_SUBGRAPH_NODE_NUMBER

    def commit(self):
        """
        Commit the core after finish growing

        Returns:
            self
        """
        del self.__pointer
        del self.__boundary_pattern_index
        del self._min_node
        del self._min_node_index
        return self
