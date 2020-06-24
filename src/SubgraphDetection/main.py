from DumpResult import dump_result
from PrasePBFile import phase_pb_file
from SubgraphDetection.Executor import Executor


def detect_subgraph(graph_path, result_path):
    graph = phase_pb_file(graph_path)
    executor = Executor(graph)
    result = executor.run()
    dump_result(result, result_path)
