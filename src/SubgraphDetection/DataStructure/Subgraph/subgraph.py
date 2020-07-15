"""This file is used to define subgraph in the simplified MindSpore graph."""
from typing import Deque, List

from SubgraphDetection.DataStructure import SNode
from SubgraphDetection.config import CONFIG


class Subgraph:
    """The subgraph: Not a subclass of SMSGraph, organized to improve performance"""

    def __init__(
            self,
            pattern: Deque[str],
            nodes: List[Deque[SNode]],
            min_node: SNode,
            min_node_index: int,
    ):
        """
        Init a Subgraph with pattern,nodes,and min id nodes info

        Args:
            pattern: The pattern of the subgraph, correspond to the nodes
            nodes: Nodes the make up the subgraph, each tuple hold one place in the pattern
            min_node: The least id node
            min_node_index: The index of least id node

        Example:
            suppose we have a subgraph with two instance:
                - Node1(biaAdd)->Node2(Conv2D)
                - Node3(biaAdd)->Node4(Conv2D)
            Then the member variables should be:
                - pattern           : ['biaAdd', 'Conv2D']
                - nodes             : [ (1,2)  , (3,4)   ]
                - min_node_id       : Node-1
                - min_node_index    : (0,0)
                - id                : hash('1-2')
        """
        # check if every subgraph instance is in the same pattern
        if CONFIG.SAFE_MODE and not all(
                [all([n[j].type == t for j, t in enumerate(pattern)]) for n in nodes]
        ):
            raise ValueError("Subgraph Nodes should be in the same pattern")

        # The pattern of the subgraph, correspond to the nodes
        self._pattern: Deque[str] = pattern

        # Nodes that make up the subgraph, each tuple hold one place in the pattern
        self._nodes: List[Deque[SNode]] = nodes

        # Unique id of SubgraphCore, used to avoid additional calculations
        # Lazy computed with property-id
        self._id: int = 0

        # Record the minimum-id node info to speed up SubgraphCore id calculation
        # Incremental updated
        self._min_node = min_node
        self._min_node_index = min_node_index

    @property
    def id(self):
        """
        Get a unique id of a Subgraph

        Returns:
            Hash of a string, which make up by the subgraph instance nodes that contains the smallest id node,
            nodes are numbered in ascending order, and split by '-'
        """
        if self._id == 0:
            self._id = hash(
                "-".join(
                    [str(node.id) for node in sorted(self._nodes[self._min_node_index])]
                )
            )
        return self._id

    @property
    def nodes(self):
        return self._nodes

    @property
    def pattern(self):
        return self._pattern

    @property
    def feature_nodes(self):
        return set(node.id for node in self._nodes[self._min_node_index])
