from collections import deque
from typing import Tuple

from DataStucture.SimpleMindsporeGraph.snode import SNode
from DataStucture.Subgraph.subgraphcore import SubgraphCore


def core_grow(executor, core: SubgraphCore) -> deque:
    """
    Make one core grow to next epoch

    Args:
        executor: The Executor Object, Locks and some other apis need to be invoked from it
        core: The core waiting for grow

    Returns:
        A tuple of new cores, grow from the input one
    """

    # TODO: implement _check_neighbors
    def _check_neighbors(nodes: Tuple[SNode]):
        pass

    # find and check the neighbors
    new_cores = deque()
    for eq_node in core:
        new_cores += _check_neighbors(eq_node)

    # commit the graph core if no further graph possibilities
    if len(new_cores) == 0:
        executor.commit(core)
    else:
        # destroy the core
        del core

    # register core to avoid extra computation
    new_core_id = tuple(map(lambda x: x.id, new_cores))
    executor.lock.acquire()
    is_core_valid = executor.register_core(new_core_id)
    executor.lock.release()

    # return all the registered cores
    return deque(new_core for i, new_core in enumerate(new_cores) if is_core_valid[i])
