"""This file is used to define node in the simplified MindSpore graph."""
from typing import Tuple

from mindinsight.datavisual.data_transform.graph import NodeTypeEnum


class SNode:
    def __init__(
            self,
            MSGraph_id: int,
            type: NodeTypeEnum,
            upstream: Tuple[int],
            downstream: Tuple[int],
    ):
        self.MSGraph_id = MSGraph_id
        self.type = type
        self.upstream = upstream
        self.downstream = downstream

    @property
    def id(self):
        return self.MSGraph_id

    def __str__(self):
        return f'Node-{self.MSGraph_id} {self.type}'
