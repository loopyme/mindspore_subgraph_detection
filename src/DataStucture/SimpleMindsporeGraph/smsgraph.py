"""This file is used to define the simplified MindSpore graph."""

from mindinsight.datavisual.data_transform.graph import MSGraph
from mindinsight.datavisual.data_transform.graph.node import Node, NodeTypeEnum

from DataStucture.NodeSet.nodeset import NodeSet
from DataStucture.SimpleMindsporeGraph.snode import SNode


class SMSGraph:
    """The object describes the simplified MindSpore graph, and it is used for subgraph detection."""

    non_normal_node_type = [
        NodeTypeEnum.NAME_SCOPE.value,
        NodeTypeEnum.PARAMETER.value,
        NodeTypeEnum.CONST.value,
        NodeTypeEnum.AGGREGATION_SCOPE.value,
    ]

    def __init__(self, graph: MSGraph):
        """
        Init a SMSGraph with a MSGraph object
        
        Args:
            graph: The MSGraph required to be parsed
        """

        # Used to store all snodes, and the key is node id, value is `SNode` object.
        self.node_set: NodeSet = SMSGraph.parse_MSGraph(graph)

    @staticmethod
    def parse_MSGraph(msgraph: MSGraph) -> NodeSet:
        """
        Parse a MSGraph to SMSGraph
        Args:
            msgraph: The MSGraph required to be parsed

        Returns:
            snode_map: all normal(defined by non_normal_node_type) snodes, and the key is node id, value is `SNode` object.
        """
        node_map = msgraph._normal_node_map

        def get_node_id(node_name: str) -> int:
            """
            help to track down the input/output node type
            """
            node: Node = node_map[node_name]

            # Id less than 0 indicates an non-normal node
            non_normal_node_type_id = {
                NodeTypeEnum.NAME_SCOPE.value: -1,
                NodeTypeEnum.PARAMETER.value: -2,
                NodeTypeEnum.CONST.value: -3,
                NodeTypeEnum.AGGREGATION_SCOPE.value: -4,
            }
            if node.type in non_normal_node_type_id.keys():
                return non_normal_node_type_id[node.type]
            else:
                return int(node.node_id)

        # (id, type, upstream, downstream) is all that we need
        # ! Only returns all normal snodes
        # TODO: Extract and make good use of Scope and Aggregation Scope info.
        return NodeSet(nodes={
            node.node_id: SNode(
                node.node_id,
                node.type,
                tuple(map(get_node_id, node.input.keys())),
                tuple(map(get_node_id, node.output.keys())),
            )
            for node in node_map.values()
            if node.type not in SMSGraph.non_normal_node_type
        })
