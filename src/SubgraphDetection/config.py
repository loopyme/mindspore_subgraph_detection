"""The file stores the config options in SubgraphDetection"""

# Whether to do some extra computation to make sure safety
SAFE_MODE = True

# The worker number of Thread Pool, -1 = cqu_count
MAX_WORKER = -1

# The minimum instance number of a subgraph, subgraph with fewer instances will not be detected
MIN_SUBGRAPH_INSTANCE_NUMBER = 2

# The minimum node number of a subgraph, subgraph instance with fewer nodes will not be detected
MIN_SUBGRAPH_NODE_NUMBER = 4

# Penalty terms are imposed on sub-sub-graph in thresholds to avoid multiple level subgraphs
SUB_SUB_GRAPH_THRESHOLD_PENALTY = 2
