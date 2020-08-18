"""This file is used to define the executor of subgraph detection process"""
from collections import deque
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from functools import reduce
from operator import __add__
from os import cpu_count
from threading import Lock
from typing import Tuple, Set, Union, Deque

from mindinsight.datavisual.common.log import logger
from mindinsight.datavisual.data_transform.graph import MSGraph

from SubgraphDetection.DataStructure import SMSGraph, SubgraphCore, Subgraph
from SubgraphDetection.Executor.grow import core_grow
from SubgraphDetection.config import CONFIG


class Executor:
    def __init__(self, graph: Union[SMSGraph, MSGraph]):
        """
        init a subgraph detect executor
        Args:
            graph: The whole graph which executor working on
        """
        # check and store the whole graph
        if isinstance(graph, MSGraph):
            self.graph: SMSGraph = SMSGraph(graph)
        elif isinstance(graph, SMSGraph):
            self.graph: SMSGraph = graph
        else:
            raise ValueError(
                f'The subgraph detection executor obtains an unrecognized graph input of type "{type(graph)}",'
                f" which should be SMSGraph or MSGraph"
            )

        # muti-thread executor
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=CONFIG.MAX_WORKER if CONFIG.MAX_WORKER > 0 else cpu_count()
        )

        # exclusive lock, useful when register the cores
        self.lock: Lock = Lock()

        # all of the detected cores
        self._registered_core: Set[int] = set()

        # cores that waiting for grow
        self._core_deque: Deque[SubgraphCore] = deque(
            map(SubgraphCore, self.graph.frequent_nodes())
        )

        # all of the detected graphs
        self._commit_subgraph: Deque[Subgraph] = deque()

    def get_subgraph(self) -> Deque[Subgraph]:
        return self._commit_subgraph

    def run(self) -> Deque[Subgraph]:
        """
        Run until every subgraph core is committed or destroyed

        Returns:
            Deque of subgraph, all the detected subgraphs
        """
        i = 1
        while self._core_deque:
            if CONFIG.VERBOSE:
                logger.info(
                    f"Epoch {i:>4}: There are {len(self._core_deque):>5} cores growing in the current epoch"
                )
            self.next_epoch()
            i += 1

        return self.get_subgraph()

    def next_epoch(self):
        """Let all cores in core_deque grow to next epoch and update the core_deque"""
        self._core_deque = reduce(
            __add__,
            (
                grow_core.result()
                for grow_core in as_completed(
                (
                    self._executor.submit(core_grow, self, core)
                    for core in self._core_deque
                )
            )
            ),
        )
        self.check_subgraph()

    def register_core(self, core_ids: Tuple[int, ...]) -> Tuple[bool, ...]:
        """
        Register the cores to avoid extra computation.

        Notes:
            If a core is registered, means it is already been computed or added to next epoch core deque.
            This function should be exclusive, make sure to acquire the lock before.

        Args:
            core_ids:   Tuple of core ids that should be checked

        Returns:
            Tuple of bool, correspond to the input core_ids tuple.

        """
        res = [True] * len(core_ids)
        for i, cid in enumerate(core_ids):
            if cid in self._registered_core:
                res[i] = False
            else:
                self._registered_core.add(cid)
        return tuple(res)

    def commit_core(self, core: SubgraphCore):
        """
        Let subgraph core commit itself to subgraph.

        Args:
            core: subgraph core which is finish growing

        Returns:
            None
        """
        core.commit()
        self._commit_subgraph.append(core)

    def check_subgraph(self):
        """
        Check if there are any sub-subgraph in self._commit_subgraph and delete the match.
        Run after an epoch is finished.

        Notes:
            Penalty terms are imposed on sub-sub-graph in thresholds to avoid multiple level subgraphs

        Returns:
            None
        """
        subgraph_size = tuple(len(g.pattern) for g in self._commit_subgraph)
        subgraph_instance = tuple(len(g.nodes) for g in self._commit_subgraph)
        subgraph_pattern = tuple(g.feature_nodes for g in self._commit_subgraph)
        remove_graph = deque()
        for i in range(len(self._commit_subgraph)):
            for j in range(len(self._commit_subgraph)):
                if i == j or i in remove_graph or j in remove_graph:
                    continue
                elif (
                        subgraph_size[i] > subgraph_size[j]
                        and subgraph_instance[i]
                        >= subgraph_instance[j]
                        - CONFIG.SUB_SUB_GRAPH_THRESHOLD_PENALTY
                        * (subgraph_size[i] - subgraph_size[j])
                        and subgraph_pattern[i] >= subgraph_pattern[j]
                ) or (
                        subgraph_size[i] < subgraph_size[j]
                        and subgraph_instance[i]
                        <= subgraph_instance[j]
                        + CONFIG.SUB_SUB_GRAPH_THRESHOLD_PENALTY
                        * (subgraph_size[j] - subgraph_size[i])
                        and subgraph_pattern[j] <= subgraph_pattern[i]
                ):
                    remove_graph.append(j)

        self._commit_subgraph = deque(
            g for i, g in enumerate(self._commit_subgraph) if i not in remove_graph
        )
