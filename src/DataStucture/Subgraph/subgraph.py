"""This file is used to define subgraph in the simplified MindSpore graph."""
from typing import Tuple, Deque

from DataStucture.SimpleMindsporeGraph.snode import SNode
from config import SAFE_MODE


class Subgraph:
    """The subgraph: Not a subclass of SMSGraph, organized to improve performance"""

    def __init__(self, pattern, nodes, min_node_id, min_node_index):
        """
        Init a Subgraph with pattern,nodes,and min id nodes info

        Args:
            pattern: The pattern of the subgraph, correspond to the nodes
            nodes: Nodes the make up the subgraph, each tuple hold one place in the pattern
            min_node_id: The id of least id node
            min_node_index: The index of least id node

        Notes:
            Here is an example: suppose we have a subgraph with two instance:
                - Node1(biaAdd)->Node2(Conv2D)
                - Node3(biaAdd)->Node4(Conv2D)
            Then the member variables should be:
                - pattern           : ['biaAdd', 'Conv2D']
                - nodes             : [ (1,3)  , (2,4)   ]
                - min_node_id       : 1
                - min_node_index    : (0,0)
                - id                : hash('1-2')
        """
        # check if subgraph is valid
        if SAFE_MODE and not all(
                [all([n[i].type == pattern[i] for n in nodes]) for i in range(len(pattern))]
        ):
            raise ValueError("Subgraph Nodes should be in the same pattern")

        # The pattern of the subgraph, correspond to the nodes
        self.pattern: Deque[str] = pattern

        # Nodes that make up the subgraph, each tuple hold one place in the pattern
        self.nodes: Deque[Tuple[SNode]] = nodes

        # Unique id of SubgraphCore, used to avoid additional calculations
        # Lazy computed with property-id
        self._id: int = 0

        # Record the minimum-id node info to speed up SubgraphCore id calculation
        # Incremental updated
        self.min_node_id = min_node_id
        self.min_node_index = min_node_index

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

    def __getitem__(self, node_id: int):
        return self.nodes[node_id]
