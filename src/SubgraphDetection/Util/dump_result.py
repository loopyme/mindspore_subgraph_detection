import json
from collections import deque
from typing import Deque

from mindinsight.datavisual.common.log import logger

from SubgraphDetection.DataStructure import Subgraph, SubgraphCore


def dump_result(subgraph_deque: Deque[Subgraph], file_path: str):
    """
    Dump the result to file

    Args:
        subgraph_deque: A deque of Subgraphs
        file_path: Where should we save the result

    Returns:
        None
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

    logger.info(
        "Start to dump subgraph detecting result to file, file path: %s.", file_path
    )
    with open(file_path, "w") as f:
        f.write(json.dumps(subgraph_deque, indent=4, cls=JsonEncoder))

    logger.info("Write subgraph file success, file path: %s.", file_path)
