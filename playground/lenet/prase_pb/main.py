import logging

from mindinsight.datavisual.common.log import logger
from mindinsight.datavisual.data_transform.graph import MSGraph

from DataStucture.SimpleMindsporeGraph.smsgraph import SMSGraph
from PrasePBFile import phase_pb_file
from playground.help import print_table
from playground.lenet.prase_pb import get_nodes_info

logger.setLevel(logging.ERROR)

graph: MSGraph = phase_pb_file("./ms_output.pb")

node_map = graph._normal_node_map

normal_nodes_info, non_normal_nodes_info = get_nodes_info(node_map)
print_table(normal_nodes_info)

"""
-----------------------------------  --  -----------------  ------------  ------  --
Conv2D                                1  ['P', 'P']         ['2']         []      []
ReLU                                  2  ['1']              ['3']         []      []
MaxPool                               3  ['2']              ['4']         []      []
Conv2D                                4  ['3', 'P']         ['5']         []      []
ReLU                                  5  ['4']              ['6']         []      []
MaxPool                               6  ['5']              ['7']         []      []
Reshape                               7  ['6', 'C']         ['8']         []      []
MatMul                                8  ['7', 'P']         ['9']         ['AS']  []
BiasAdd                               9  ['8', 'P']         ['10']        ['AS']  []
ReLU                                 10  ['9']              ['11']        []      []
MatMul                               11  ['10', 'P']        ['12']        ['AS']  []
BiasAdd                              12  ['11', 'P']        ['13']        ['AS']  []
ReLU                                 13  ['12']             ['14']        []      []
MatMul                               14  ['13', 'P']        ['15']        ['AS']  []
BiasAdd                              15  ['14', 'P']        ['16', '17']  ['AS']  []
SparseSoftmaxCrossEntropyWithLogits  16  ['15', 'P']        ['17']        []      []
make_tuple                           17  ['16', '15', 'P']  []            []      []
-----------------------------------  --  -----------------  ------------  ------  --
"""

g = SMSGraph(graph)
for i in g.node_set:
    print(i)
