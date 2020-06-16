import logging

from mindinsight.datavisual.common.log import logger
from mindinsight.datavisual.data_transform.graph import MSGraph

from detect_subgraph.prase_pb import phase_pb_file
from playground.help import print_table
from playground.lenet.prase_pb import get_nodes_info

logger.setLevel(logging.ERROR)

graph: MSGraph = phase_pb_file("./ms_output.pb")

node_map = graph._normal_node_map

normal_nodes_info, non_normal_nodes_info = get_nodes_info(node_map)
print_table(normal_nodes_info)
