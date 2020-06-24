from SubgraphDetection.Core.executor import Executor
from SubgraphDetection.Util import dump_result, phase_pb_file


def detect_subgraph(graph_path, result_path):
    graph = phase_pb_file(graph_path)
    executor = Executor(graph)
    result = executor.run()
    dump_result(result, result_path)
