from typing import Union

from google.protobuf.text_format import ParseError
from mindinsight.datavisual.common.log import logger
from mindinsight.datavisual.data_access.file_handler import FileHandler
from mindinsight.datavisual.data_transform.graph import MSGraph
from mindinsight.datavisual.proto_files import mindinsight_anf_ir_pb2 as anf_ir_pb2
from mindinsight.utils.exceptions import UnknownError


def phase_pb_file(file_path: str) -> Union[MSGraph, None]:
    """
    Parse pb file to graph

    Args:
        file_path (str): The file path of pb file.

    Returns:
        MSGraph, if load pb file and build graph success, will return the graph, else return None.
    """

    logger.info("Start to load graph from pb file, file path: %s.", file_path)
    model_proto = anf_ir_pb2.ModelProto()
    try:
        model_proto.ParseFromString(FileHandler(file_path).read())
    except ParseError:
        logger.warning("The given file is not a valid pb file, file path: %s.", file_path)
        return None

    graph = MSGraph()

    try:
        graph.build_graph(model_proto.graph)
    except Exception as ex:
        logger.error("Build graph failed, file path: %s.", file_path)
        logger.exception(ex)
        raise UnknownError(str(ex))

    logger.info("Build graph success, file path: %s.", file_path)
    return graph
