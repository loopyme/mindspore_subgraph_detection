from collections import deque, Counter
from typing import Tuple

from DataStucture.SimpleMindsporeGraph.snode import SNode
from DataStucture.Subgraph.subgraphcore import SubgraphCore
from config import MIN_SUBGRAPH_INSTANCE_NUMBER, MIN_SUBGRAPH_NODE_NUMBER


def core_grow(executor, core: SubgraphCore) -> deque:
    """
    Make one core grow to next epoch

    Args:
        executor: The Executor Object, Locks and some other apis need to be invoked from it
        core: The core waiting for grow

    Returns:
        A tuple of new cores, grow from the input one
    """

    # TODO: rewrite in cython
    def _check_stream_on_one_position(stream_nodes):
        # count the common neighbor type
        neighbor_type = tuple(
            "-".join(tuple(node.type for node in neighbor_nodes_per_node))
            for neighbor_nodes_per_node in stream_nodes
        )
        common_type = tuple(
            t[0]
            for t in Counter(neighbor_type).most_common()
            if t[1] >= MIN_SUBGRAPH_INSTANCE_NUMBER
        )

        # Tidy pattern-nodes data
        pattern_nodes_dic = {p: deque() for p in common_type}
        pattern_keep_core_dic = {p: deque() for p in common_type}
        for i, p in enumerate(neighbor_type):
            if p in common_type:
                pattern_nodes_dic[p].append(stream_nodes[i])
                pattern_keep_core_dic[p].append(i)

        return deque(
            core.grow(
                deque(pattern.split("-")), nodes, tuple(pattern_keep_core_dic[pattern])
            )
            for pattern, nodes in pattern_nodes_dic.items()
            if pattern != ""
        )

    def _check_neighbors_on_one_position(nodes: Tuple[SNode]):
        # find all possible upstream and downstream nodes
        upstream_nodes = tuple(
            tuple(executor.graph[node_id] for node_id in node.upstream)
            for node in nodes
        )
        # TODO: optimize the algorithm to allow downstream check
        # downstream_nodes = tuple(
        #     tuple(executor.graph[node_id] for node_id in node.downstream)
        #     for node in nodes
        # )
        return _check_stream_on_one_position(
            upstream_nodes
        )  # + _check_stream_on_one_position(downstream_nodes)

    # find and check the neighbors
    new_cores = deque()
    for eq_node in core:
        new_cores += _check_neighbors_on_one_position(eq_node)

    # commit the graph core if no further graph possibilities
    if (
            len(new_cores) == 0
            and len(core.nodes) >= MIN_SUBGRAPH_INSTANCE_NUMBER
            and len(core.nodes[0]) >= MIN_SUBGRAPH_NODE_NUMBER
    ):
        executor.commit_core(core.commit())
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
