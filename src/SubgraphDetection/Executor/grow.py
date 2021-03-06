"""This file implement the core grow logic"""
from collections import deque, Counter
from typing import Tuple, Deque

from SubgraphDetection.DataStructure import SNode
from SubgraphDetection.DataStructure import SubgraphCore
from SubgraphDetection.config import CONFIG


def core_grow(executor, core: SubgraphCore) -> deque:
    """
    Make one core grow to next epoch

    Args:
        executor: The Executor object, Locks and some other apis need to be invoked from it
        core: The core waiting for grow

    Returns:
        A tuple of new cores, grow from the input one
    """

    def _check_stream_on_one_position(stream_nodes):
        # get the neighbor type
        neighbor_type = tuple(
            "/".join(tuple(node.type for node in neighbor_nodes_per_node))
            for neighbor_nodes_per_node in stream_nodes
        )

        # remove redundant node pattern: patch to handle cases where multiple nodes of the same type
        # are connected to the same node
        neighbor_node = tuple(
            tuple(node for node in neighbor_nodes_per_node)
            for neighbor_nodes_per_node in stream_nodes
        )
        duplicate_nodes_pattern = {n: 0 for n in neighbor_type}
        node_count = Counter(neighbor_node).most_common()
        for node, count in node_count:
            if count > 1:
                duplicate_nodes_pattern["/".join(tuple(n.type for n in node))] = (
                        count - 1
                )

        # count the common neighbor type
        common_type = tuple(
            t[0]
            for t in Counter(neighbor_type).most_common()
            if t[1] - duplicate_nodes_pattern[t[0]]
            >= CONFIG.MIN_SUBGRAPH_INSTANCE_NUMBER
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
                deque(pattern.split("/")), nodes, tuple(pattern_keep_core_dic[pattern])
            )
            for pattern, nodes in pattern_nodes_dic.items()
            if pattern != ""
        )

    def _check_neighbors_on_one_position(nodes: Tuple[SNode, ...]):
        def _get_upstream_nodes(node):

            if not CONFIG.SCOPE_BOUNDARY:
                return tuple(executor.graph_nodes[node_id] for node_id in node.upstream)
            else:
                # use scope as a search boundary
                res = tuple(
                    filter(
                        lambda upstream_node: upstream_node.scope == node.scope,
                        tuple(
                            executor.graph_nodes[node_id] for node_id in node.upstream
                        ),
                    )
                )
                return res

        # if nodes are not normal, return empty deque
        if nodes[0].id < 0:
            return deque()

        # find all possible upstream and downstream nodes
        upstream_nodes = tuple(_get_upstream_nodes(node) for node in nodes)
        if None in upstream_nodes:
            return tuple()

        # TODO: optimize the algorithm to allow downstream check
        # downstream_nodes = tuple(
        #     tuple(executor.graph[node_id] for node_id in node.downstream)
        #     for node in nodes
        # )
        return _check_stream_on_one_position(
            upstream_nodes
        )  # + _check_stream_on_one_position(downstream_nodes)

    # find and check the neighbors
    new_cores: Deque[SubgraphCore] = deque()
    for eq_node in core:
        new_cores += _check_neighbors_on_one_position(eq_node)

    # commit the graph core if no further graph possibilities
    if len(new_cores) == 0 and core.is_valid_for_commit:
        executor.commit_core(core)
    else:
        # destroy the core
        del core

    # remove it if the new_core is too large
    new_cores = tuple(
        filter(lambda x: x.size < CONFIG.MAX_SUBGRAPH_NODE_NUMBER, new_cores)
    )

    # register core to avoid extra computation
    new_core_id = tuple(map(lambda x: x.id, new_cores))
    executor.lock.acquire()
    is_core_valid = executor.register_core(new_core_id)
    executor.lock.release()

    # return all the registered cores
    return deque(new_core for i, new_core in enumerate(new_cores) if is_core_valid[i])
