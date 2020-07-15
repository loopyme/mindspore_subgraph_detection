"""The file stores the config options in SubgraphDetection"""

from dataclasses import dataclass


@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=False)
class Config:
    # Whether to print all the detailed running infos
    VERBOSE: bool = False

    # Whether to do some extra computation to make sure safety
    SAFE_MODE: bool = False

    # The worker number of Thread Pool, -1 = cqu_count
    MAX_WORKER: int = -1

    # The minimum instance number of a subgraph, subgraph with fewer instances will not be detected
    MIN_SUBGRAPH_INSTANCE_NUMBER: int = 2

    # The minimum node number of a subgraph, subgraph instance with fewer nodes will not be detected
    MIN_SUBGRAPH_NODE_NUMBER: int = 4

    # Penalty terms are imposed on sub-sub-graph in zthresholds to avoid multiple level subgraphs
    SUB_SUB_GRAPH_THRESHOLD_PENALTY: int = 2

    # Whether to check the result after finish calculation
    CHECK_RESULT: bool = False

    def set(self, attrs: dict):
        for name, value in attrs.items():
            print(name, value)
            self.__setattr__(name, value)


# Package-level config
config = Config()
