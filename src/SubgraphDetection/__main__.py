from SubgraphDetection.Executor.executor import Executor
from SubgraphDetection.Util import dump_result, phase_pb_file


def detect_subgraph(graph_path, result_path):
    """
    Detect the subgraph in a mindspore computational graph

    Notes:
        Config options are in SubgraphDetection.config

    Args:
        graph_path: The pb file where the whole graph are stored.
        result_path: The json file where the detected subgraphs should be dumped.

    Returns:
        None
    """
    graph = phase_pb_file(graph_path)
    executor = Executor(graph)
    result = executor.run()
    dump_result(result, result_path)
