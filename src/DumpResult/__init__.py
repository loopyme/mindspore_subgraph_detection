import json
from collections import deque
from typing import Deque

from mindinsight.datavisual.common.log import logger

from DataStucture.Subgraph.subgraph import Subgraph
from DataStucture.Subgraph.subgraphcore import SubgraphCore


def dump_result(subgraph_deque: Deque[Subgraph], file_path: str):
    """

    Args:
        subgraph_deque:
        file_path:

    Returns:

    """

    class JsonEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, deque):
                return self.default(tuple(o))
            elif isinstance(o, (list, tuple)):
                return o
            elif isinstance(o, (Subgraph, SubgraphCore)):
                return self.default(o.nodes)
            return str(o)

    logger.info("Start to load graph from pb file, file path: %s.", file_path)
    with open(file_path, "w") as f:
        f.write(json.dumps(subgraph_deque, indent=4, cls=JsonEncoder))

    logger.info("Build graph success, file path: %s.", file_path)
