from functools import partial
from typing import Dict, Tuple, List, Any, Union

from mindinsight.datavisual.data_transform.graph.node import Node, NodeTypeEnum

non_normal_node_type = [
    NodeTypeEnum.NAME_SCOPE.value,
    NodeTypeEnum.PARAMETER.value,
    NodeTypeEnum.CONST.value,
    NodeTypeEnum.AGGREGATION_SCOPE.value,
]


def get_node_id(node_map: Dict[str, Node], node_name: str) -> str:
    """help to track down the input/output node type"""
    node: Node = node_map[node_name]
    if node is None:
        return ""
    elif node.type == NodeTypeEnum.NAME_SCOPE.value:
        return "S"
    elif node.type == NodeTypeEnum.PARAMETER.value:
        return "P"
    elif node.type == NodeTypeEnum.CONST.value:
        return "C"
    elif node.type == NodeTypeEnum.AGGREGATION_SCOPE.value:
        return "AS"
    else:
        return node.node_id


def get_nodes_info(
        node_map: Dict[str, Node]
) -> Tuple[List[Dict[str, Union[list, Any]]], List[Dict[str, Union[list, Any]]]]:
    """Extract all that we need in subgraph detection from a node"""
    get_node_id_partial = partial(get_node_id, node_map)

    def get_node_info(node: Node):
        return {
            "type": node.type,
            "id": node.node_id,
            "input": list(map(get_node_id_partial, node.input.keys())),
            "output": list(map(get_node_id_partial, node.output.keys())),
            "proxy_input": list(map(get_node_id_partial, node.proxy_input.keys())),
            "proxy_output": list(map(get_node_id_partial, node.proxy_output.keys())),
        }

    return (
        [
            get_node_info(node)
            for _, node in node_map.items()
            if node.type not in non_normal_node_type
        ],
        [
            get_node_info(node)
            for _, node in node_map.items()
            if node.type in non_normal_node_type
        ],
    )
