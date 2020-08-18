from typing import Deque

from mindinsight.datavisual.data_transform.graph import MSGraph

from SubgraphDetection.DataStructure import SMSGraph
from SubgraphDetection.DataStructure import Subgraph


class ResultCheck:
    def __init__(self, graph: MSGraph, subgraph_deque: Deque[Subgraph]):
        """
        Check the subgraph detection result
        Args:
            graph: The origin graph
            subgraph_deque: The result
        """

        self.graph_size = len(
            tuple(
                node
                for node in graph._normal_node_map.values()
                if node.type not in set(SMSGraph.non_normal_node_type[:1])
            )
        )
        self.num_subgraph = len(subgraph_deque)
        self.subgraph_size = tuple(len(g.nodes[0]) for g in subgraph_deque)
        self.subgraph_count = tuple(len(g.nodes) for g in subgraph_deque)
        self.new_graph_size = self.graph_size - len(
            set(
                node
                for subgraph in subgraph_deque
                for instance in subgraph.nodes
                for node in instance
            )
        )

    @property
    def MDL(self) -> float:
        """
        Calculate MDL(Minimum description length) of the result

        Notes:
            MDL = dl(g)/(dl(s)+dl(g|s))
             - dl(g): size of the origin graph
             - dl(s): total size of the subgraphs
             - dl(g|s)): total size of the graph which remove all the detected subgraphs
        """

        return self.graph_size / (sum(self.subgraph_size) + self.new_graph_size)

    @property
    def minimum_description_length(self):
        return self.MDL

    @property
    def reduce_ratio(self) -> float:
        return (
                       self.graph_size - sum(self.subgraph_size) - self.new_graph_size
               ) / self.graph_size

    def __repr__(self):
        return (
            f"+++++++++Result+++++++++\n"
            f"Graph Size = {self.graph_size}\n"
            f"No. subgraph = {self.num_subgraph}\n"
            f"No. Subgraph instance= {self.subgraph_count}\n"
            f"Subgraph size= {self.subgraph_size}\n"
            f"MDL = {self.MDL:.2f}\n"
            f"Reduce ratio = {self.reduce_ratio:.2f}\n"
        )
