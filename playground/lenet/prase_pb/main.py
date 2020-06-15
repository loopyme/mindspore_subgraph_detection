import logging

from mindinsight.datavisual.common.log import logger
from mindinsight.datavisual.data_transform.graph import MSGraph

from detect_subgraph.prase_pb import phase_pb_file

logger.setLevel(logging.ERROR)

graph: MSGraph = phase_pb_file("./ms_output.pb")

for item in graph._node_id_map_name:
    print(item)

print(" ")
for item in graph._normal_node_map:
    print(item)
