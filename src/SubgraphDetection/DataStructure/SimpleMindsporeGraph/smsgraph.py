"""This file is used to define the simplified MindSpore graph."""
from collections import deque
from copy import deepcopy
from typing import Dict, List, Tuple, Deque, Union

from mindinsight.datavisual.data_transform.graph import MSGraph
from mindinsight.datavisual.data_transform.graph.node import NodeTypeEnum

from SubgraphDetection.DataStructure.SimpleMindsporeGraph.scope import Scope
from SubgraphDetection.DataStructure.SimpleMindsporeGraph.snode import SNode
from SubgraphDetection.config import CONFIG


class SMSGraph:
    """This object describes the simplified MindSpore graph, and it is used for subgraph detection."""

    non_normal_node_type = [
        NodeTypeEnum.NAME_SCOPE.value,
        NodeTypeEnum.AGGREGATION_SCOPE.value,
        NodeTypeEnum.PARAMETER.value,
        NodeTypeEnum.CONST.value,
    ]

    def __init__(self, graph: MSGraph):
        """
        Init a SMSGraph with a MSGraph object
        
        Args:
            graph: The MSGraph required to be parsed
        """

        # Store all normal type snodes, and the key is node id, value is `SNode` object.
        self._normal_node: Dict[int, SNode]

        # Store info about name scopes, useful when hierarchical detect,
        # the key is scope name, value is `Scope` object.
        self._scope_node: Dict[str, Scope]

        # Store info about parameter&const nodes, useful when hierarchical detect,
        # the key is nodes id, value is `SNode` object.
        self._parameter_node: Dict[int, Scope]

        (
            self._normal_node,
            self._scope_node,
            self._parameter_node,
        ) = SMSGraph.parse_MSGraph(graph)

        # Nodes with larger index are name scopes (They are actually forged to be a normal nodes)
        self._scope_index = max(self._normal_node.keys())

        # Run scope isomorphism check
        self._check_scope()

    @staticmethod
    def parse_MSGraph(
            msgraph: MSGraph,
    ) -> Tuple[Dict[int, SNode], Dict[str, Scope], Dict[int, SNode]]:
        """
        Parse a MSGraph to SMSGraph

        Notes:
            In this function, We assigned id to every node-like object,
            even the scope_node & parameter_node.

        Args:
            msgraph: The MSGraph required to be parsed

        Returns:
            normal_node: all normal(defined by non_normal_node_type) snodes,
                        and the key is node id, value is `SNode` object.
            scope_node: all name scope nodes that we created,
                        and the key is scope name, value is `Scope` object.
            parameter_node: all parameter and const nodes,
                        and the key is node id, value is `SNode` object.
        """
        node_map = msgraph._normal_node_map

        # re-assign node id
        node_index = {name: i for i, name in enumerate(node_map.keys())}
        get_node_id = lambda name: node_index[name]

        parameter_node = {
            get_node_id(name): SNode(
                id=get_node_id(name),
                type="parameter" if node.type == -2 else "const",
                upstream=(),
                downstream=tuple(map(get_node_id, node.output.keys())),
                scope=name[: name.rfind("/")],
                level=1,
                name=name,
            )
            for i, (name, node) in enumerate(node_map.items())
            if node.type in (NodeTypeEnum.PARAMETER.value, NodeTypeEnum.CONST.value)
        }

        # (id, type, upstream, downstream, scope) is all that we need
        normal_node = {
            get_node_id(name): SNode(
                id=get_node_id(name),
                type=node.type,
                upstream=tuple(map(get_node_id, node.input.keys())),
                downstream=tuple(map(get_node_id, node.output.keys())),
                scope=name[: name.rfind("/")],
                level=1,
                name=name,
            )
            for name, node in node_map.items()
            if node.type not in SMSGraph.non_normal_node_type
        }

        scope_node = {
            name: Scope(
                id=get_node_id(name),
                upstream=tuple(map(get_node_id, node.input.keys())),
                downstream=tuple(map(get_node_id, node.output.keys())),
                scope=name[: name.rfind("/")],
                member=None,
                level=-1,
                type="",
                name=name,
            )
            for i, (name, node) in enumerate(node_map.items())
            if node.type == NodeTypeEnum.NAME_SCOPE.value
        }
        return normal_node, scope_node, parameter_node

    @staticmethod
    def frequent_nodes(nodes: Dict[int, SNode]) -> Deque[Tuple[SNode, ...]]:
        """
        Count the frequent nodes and return a deque of node tuples, which may be used to build subgraph core later
        Those node whose occurrences less than MIN_SUBGRAPH_INSTANCE_NUMBER will not returned

        Args:
            nodes: Nodes that we are counting

        Returns:
            Each tuple contains same-type nodes
        """
        count_res: Deque[Tuple[SNode, ...]] = deque()

        # sort node by type
        sorted_node_map: List[Tuple[str, SNode]] = sorted(
            nodes.items(), key=lambda x: x[1].type, reverse=True
        )

        # count it
        temp_type = ""
        node_buffer: Deque[SNode] = deque()
        for n in sorted_node_map:
            if n[1].type != temp_type:
                # different type from later one
                temp_type = n[1].type
                if len(node_buffer) >= CONFIG.MIN_SUBGRAPH_INSTANCE_NUMBER:
                    count_res.append(tuple(node_buffer))

                node_buffer.clear()
            node_buffer.append(n[1])

        # check the remaining ones
        if len(node_buffer) >= CONFIG.MIN_SUBGRAPH_INSTANCE_NUMBER:
            count_res.append(tuple(node_buffer))
        return count_res

    def __getitem__(self, node_id: int) -> SNode:
        return self._normal_node[node_id]

    def get_node_count(self):
        """Count the number of nodes"""
        return len(self._normal_node) + len(self._parameter_node)

    def _check_scope(self):
        """Run scope isomorphism check and fill in the scope info (level, type & member)"""

        # figure out the member nodes of scopes
        scope_member_buffer = {scope_name: [] for scope_name in self._scope_node.keys()}
        for n in self._normal_node.values():
            if n.scope in scope_member_buffer.keys():
                scope_member_buffer[n.scope].append(n.id)
        for s in self._scope_node.values():
            if s.scope in scope_member_buffer.keys():
                scope_member_buffer[s.scope].append(s.id)
        for name, member in scope_member_buffer.items():
            self._scope_node[name].member = tuple(member)

        # figure out the level
        # Use a monkey patch to force all scope node update their level recursively
        scope_node_id_name = {
            scope.id: name for name, scope in self._scope_node.items()
        }

        def get_level(scope: Scope):
            """
            A recursive function, used to update the current node level according to the node tree

            Returns:
                None
            """
            if scope.level == -1:
                member_level = tuple(
                    self._normal_node[i].level
                    if i in self._normal_node.keys()
                    else self._scope_node[scope_node_id_name[i]].get_level()
                    for i in scope.member
                )
                # TODO: This is weird, why is there some scope without members?
                if not member_level:
                    scope.level = -1
                else:
                    scope.level = max(member_level) + 1
            return scope.level

        Scope.get_level = get_level
        for s in self._scope_node.values():
            s.get_level()
        del Scope.get_level  # revoke monkey patch

        # figure out the type
        if not CONFIG.DETAILED_ISOMORPHIC_CHECK:
            # only the scope with the same name will be treated as isomorphism
            for name, scope in self._scope_node.items():
                scope.type = name[name.rfind("/") + 1:]
        else:
            # isomorphism check: check level, size and connected node count
            # TODO: detailed isomorphism check (very expensive, may be unnecessary)
            scope_basic_info_buffer = {
                scope_name: hash("{level}-{size}-{upstream}-{downstream}-{upstream_type}-{downstream_hash}".format(
                    level=scope_name.count("/"),
                    size=len(scope.member),
                    upstream=len(scope.upstream),
                    downstream=len(scope.downstream),
                    upstream_type=hash("".join(self._normal_node[n].type for n in scope.upstream)),
                    downstream_hash=hash("".join(self._normal_node[n].type for n in scope.downstream)),
                ))
                for scope_name, scope in self._scope_node.items()
            }
            isomorphic_scope_candidate = []
            scope_basic_info_buffer_reverse = {}
            for scope_id, info in scope_basic_info_buffer.items():
                if info not in scope_basic_info_buffer_reverse.keys():
                    scope_basic_info_buffer_reverse[info] = [
                        scope_id
                    ]
                else:
                    scope_basic_info_buffer_reverse[info].append(
                        scope_id
                    )
            for info, scope_ids in scope_basic_info_buffer_reverse.items():
                if len(scope_ids) > 1:
                    isomorphic_scope_candidate.append(scope_ids)

            for candidate in isomorphic_scope_candidate:
                scope_type = hash(candidate[0])
                for s in candidate:
                    self._scope_node[s].type = str(scope_type)

    def get_level_node(self, level: int) -> Dict[int, Union[SNode, Scope]]:
        """
        Get a projection of a certain level of whole graph

        Args:
            level: which level are we dealing with

        Returns:
            Node dict of the certain level
        """
        if level == 1:
            # TODO: maybe we should remove some node to avoid sub-sub-graph
            node_set = self._normal_node.copy()
            node_set.update(self._parameter_node)
        else:
            node_set = {
                scope.id: deepcopy(scope)
                for name, scope in self._scope_node.items()
                if scope.level == level
            }

            if not node_set:
                return {}

            # redirect the scope upstream and downstream
            def get_node_scope(node: SNode, target_level) -> Scope:
                n_level, n_scope = node.level, node
                while n_level < target_level:
                    if node.scope in self._scope_node.keys():
                        n_scope = self._scope_node[n_scope.scope]
                        n_level = n_scope.level
                    else:
                        n_scope = None
                        break
                if n_level > target_level:
                    n_scope = None
                return n_scope.id if n_scope is not None else None

            for _, n in node_set.items():
                n.upstream = tuple(
                    get_node_scope(self._normal_node[node_id], level)
                    for node_id in n.upstream
                )
                n.downstream = tuple(
                    get_node_scope(self._normal_node[node_id], level)
                    for node_id in n.downstream
                )
                n.upstream = tuple(set(filter(lambda x: x is not None, n.upstream)))
                n.downstream = tuple(set(filter(lambda x: x is not None, n.downstream)))
        return node_set

    def get_max_level(self) -> int:
        """
        Get the max level of the graph scope nodes
        Returns:
            max level
        """
        return max(scope.level for scope in self._scope_node.values())
