import time
from typing import Deque

from SubgraphDetection import __version__
from SubgraphDetection.DataStructure import Subgraph
from SubgraphDetection.Executor.executor import Executor
from SubgraphDetection.Util import dump_result, phase_pb_file
from SubgraphDetection.Util.check_result import ResultCheck
from SubgraphDetection.config import CONFIG


def detect_subgraph(graph_path, result_path, **kwargs) -> Deque[Subgraph]:
    """
    Detect the subgraph in a mindspore computational graph

    Notes:
        Config options are in SubgraphDetection.config

    Args:
        graph_path: The pb file where the whole graph are stored.
        result_path: The json file where the detected subgraphs should be dumped.
        **kwargs: Any other args will pass to config

    Returns:
        Deque of subgraph, all the detected subgraphs
    """
    time_st = time.time()
    CONFIG.set(kwargs)
    graph = phase_pb_file(graph_path)
    executor = Executor(graph)
    result = executor.run()
    dump_result(result, result_path)

    if CONFIG.CHECK_RESULT:
        print(ResultCheck(graph, result))
    if CONFIG.VERBOSE or CONFIG.CHECK_RESULT:
        print(f"Detecting finished and result have been write to {result_path}, "
              f"total usage of time:{time.time() - time_st} s")
    return result


def detect_subgraph_in_console():
    """
    Run detect_subgraph in console

    Returns:
        Deque of subgraph, all the detected subgraphs
    """
    import argparse

    def parse_args() -> argparse.Namespace:
        """Parse the command line arguments for the `detect_subgraph_in_console` binary.

        :return: Namespace with parsed arguments.
        """
        parser = argparse.ArgumentParser(
            prog="detect-subgraph",
            description="Detect subgraphs in a Mindspore computational graph",
        )

        parser.add_argument(
            "graph path",
            type=str,
            help="The path of the pb file where the whole graph are stored",
        )

        parser.add_argument(
            "result path",
            type=str,
            help="The path of json file where the detected subgraphs should be dumped.",
        )

        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"SubgraphDetection {__version__}",
        )

        parser.add_argument(
            "--verbose",
            dest="VERBOSE",
            help="Print details to console",
            action="store_true",
        )
        parser.add_argument(
            "--safe-mode",
            "-s",
            dest="SAFE_MODE",
            help="Do some extra computation to make sure safety",
            action="store_true",
        )
        parser.add_argument(
            "-w",
            "--worker",
            dest="MAX_WORKER",
            help="The worker number of Thread Pool, -1 = cqu_count",
            type=int,
            default=CONFIG.MAX_WORKER,
        )
        parser.add_argument(
            "-i",
            "--min-instance",
            dest="MIN_SUBGRAPH_INSTANCE_NUMBER",
            help="The minimum instance number of a subgraph, "
                 "subgraph with fewer instances will not be detected",
            type=int,
            default=CONFIG.MIN_SUBGRAPH_INSTANCE_NUMBER,
        )
        parser.add_argument(
            "-n",
            "--min-nodes",
            dest="MIN_SUBGRAPH_NODE_NUMBER",
            help="The minimum node number of a subgraph, "
                 "subgraph instance with fewer nodes will not be detected",
            type=int,
            default=CONFIG.MIN_SUBGRAPH_NODE_NUMBER,
        )

        parser.add_argument(
            "-p",
            "--penalty",
            dest="SUB_SUB_GRAPH_THRESHOLD_PENALTY",
            help="Impose penalty terms on sub-sub-graph in thresholds "
                 "to avoid multiple level subgraphs",
            type=int,
            default=CONFIG.SUB_SUB_GRAPH_THRESHOLD_PENALTY,
        )
        parser.add_argument(
            "--check_result",
            "-c",
            dest="CHECK_RESULT",
            help="Check the result after finish calculation",
            action="store_true",
        )

        return parser.parse_args()

    args = vars(parse_args())
    return detect_subgraph(args.pop("graph path"), args.pop("result path"), **args)
