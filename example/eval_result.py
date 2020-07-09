from typing import Deque

from SubgraphDetection.DataStructure import Subgraph, SMSGraph
from SubgraphDetection.Util import phase_pb_file


class EvalResult:
    def __init__(self, graph_path: str, subgraph_deque: Deque[Subgraph]):
        """
        Eval the subgraph detection result
        Args:
            graph_path: The path of the origin pb file
            subgraph_deque: The result
        """
        graph = phase_pb_file(graph_path)
        self.graph_size = len(tuple(node for node in graph._normal_node_map.values() if
                                    node.type not in set(SMSGraph.non_normal_node_type[:1])))
        self.num_subgraph = len(subgraph_deque)
        self.subgraph_size = tuple(len(g.nodes[0]) for g in subgraph_deque)
        self.subgraph_count = tuple(len(g.nodes) for g in subgraph_deque)
        self.new_graph_size = self.graph_size - len(
            tuple(node for subgraph in subgraph_deque for instance in subgraph.nodes for node in instance))

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
        return (self.graph_size - sum(self.subgraph_size) - self.new_graph_size) / self.graph_size

    def __repr__(self):
        return f"+++++++++Eval Result+++++++++\n" \
               f"Graph Size = {self.graph_size}\n" \
               f"No. subgraph = {self.num_subgraph}\n" \
               f"No. Subgraph instance= {self.subgraph_count}\n" \
               f"Subgraph size= {self.subgraph_size}\n" \
               f"MDL = {self.MDL:.2f}\n" \
               f"Reduce ratio = {self.reduce_ratio:.2f}"
