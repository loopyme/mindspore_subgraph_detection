"""This file is used to define node set in the simplified MindSpore graph."""
from typing import Dict, Set

from DataStucture.SimpleMindsporeGraph.snode import SNode


class NodeSet:
    """Mainly used for the collection control in subgraph core growing"""

    def __init__(self, nodes: Dict[int, SNode]):
        """
        Init a NodeSet with a dict.

        Args:
            nodes: The dict contains all the SNodes, with id as key and SNode as value
        """

        # contains all the nodes: with id as key and SNode as value
        self.nodes: Dict[int, SNode] = nodes

        # boundary_nodes holds the ids of all the boundary points, only these nodes will be traversed
        self.boundary_nodes: Set[int] = set(self.nodes.keys())

        # useful when executing traversal, point to boundary_nodes Set
        self.pointer = None

    def __next__(self):
        return self.nodes[self.pointer.__next__()]

    def __iter__(self):
        # only these nodes whose id in boundary_nodes will be traversed
        self.pointer = self.boundary_nodes.__iter__()
        return self

    def get_iter_of_all(self):
        # force to a iter of all nodes
        self.pointer = set(self.nodes.keys()).__iter__()
        return self

    def set_interior(self, node_id: int):
        self.boundary_nodes.remove(node_id)

    def set_boundary(self, node_id: int):
        self.boundary_nodes.add(node_id)

    def __getitem__(self, node_id: int):
        return self.nodes[node_id]

    def append(self, node: SNode):
        self.nodes[node.id] = node

    def __add__(self, other):
        self.nodes.update(other.nodes)

    def __contains__(self, node: SNode):
        return node.id in self.nodes.keys()
