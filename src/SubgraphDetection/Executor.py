from collections import deque
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from functools import reduce
from operator import __add__
from os import cpu_count
from threading import Lock
from typing import Tuple, Set

from DataStucture.SimpleMindsporeGraph.smsgraph import SMSGraph
from DataStucture.Subgraph.subgraph import Subgraph
from DataStucture.Subgraph.subgraphcore import SubgraphCore
from SubgraphDetection.grow import core_grow
from config import MAX_WORKER


class Executor:
    def __init__(self, graph: SMSGraph):
        # store the whole graph
        self.graph: SMSGraph = graph

        # muti-thread executor
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=MAX_WORKER if MAX_WORKER > 0 else cpu_count()
        )

        # exclusive lock, useful when register the cores
        self.lock: Lock = Lock()

        # all of the detected cores
        self._registered_core: Set[Subgraph] = set()

        # cores that waiting for grow
        self._core_deque = deque(map(SubgraphCore, self.graph.node_count()))

        # all of the detected graphs
        self._commit_subgraph = deque()

    def run(self):
        """
        Run until every subgraph core is committed or destroyed

        Returns:
            Deque of subgraph, all the detected subgraphs
        """
        while self._core_deque:
            self.next_epoch()
        return self._commit_subgraph

    def next_epoch(self):
        """Make all core in core_deque grow to next epoch and update the core_deque"""
        self._core_deque = reduce(
            __add__,
            (
                grow_core.result()
                for grow_core in as_completed(
                (
                    self.executor.submit(core_grow, self, core)
                    for core in self._core_deque
                )
            )
            ),
        )

    def register_core(self, core_ids: Tuple[int]) -> Tuple[bool]:
        """
        Register the cores to avoid extra computation.

        Notes:
            If a core is registered, means it is already computed or added to next epoch core deque.
            This should be a exclusive function, make sure to acquire the lock before.

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

    def commit_core(self, core):
        self._commit_subgraph.append(core)
