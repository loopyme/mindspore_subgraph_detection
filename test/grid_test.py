import io
import sys
from pathlib import Path
from time import time

from SubgraphDetection import detect_subgraph


def _build_config_grid():
    return [
        {
            "DETAILED_ISOMORPHIC_CHECK": DETAILED_ISOMORPHIC_CHECK,
            "SCOPE_BOUNDARY": SCOPE_BOUNDARY,
            "MAX_SUBGRAPH_NODE_NUMBER": MAX_SUBGRAPH_NODE_NUMBER,
            "MIN_SUBGRAPH_NODE_NUMBER": MIN_SUBGRAPH_NODE_NUMBER,
            "MIN_SUBGRAPH_INSTANCE_NUMBER": MIN_SUBGRAPH_INSTANCE_NUMBER,
        }
        for DETAILED_ISOMORPHIC_CHECK in (True, False)
        for SCOPE_BOUNDARY in (True, False)
        for MAX_SUBGRAPH_NODE_NUMBER in range(24, 49, 12)
        for MIN_SUBGRAPH_NODE_NUMBER in range(3, 6)
        for MIN_SUBGRAPH_INSTANCE_NUMBER in range(2, 5)
    ]


def mean(array):
    return sum(array) / len(array) if array else 0


def write_result_header():
    with open("./test_result.csv", "w") as f:
        f.write(
            ",".join(
                [
                    "DETAILED_ISOMORPHIC_CHECK",
                    "SCOPE_BOUNDARY",
                    "MAX_SUBGRAPH_NODE_NUMBER",
                    "MIN_SUBGRAPH_NODE_NUMBER",
                    "MIN_SUBGRAPH_INSTANCE_NUMBER",
                    "pb_name",
                    "time",
                    "graph_size",
                    "num_subgraph",
                    "ave_num_subgraph_instance",
                    "ave_subgraph_size",
                    "MDL",
                    "reduce_ratio",
                ]
            ) + "\n"
        )


def write_result(test_result):
    with open("./test_result.csv", "a") as f:
        f.write(",".join(map(str, test_result[-1].values())) + "\n")


def test_detect():
    # Do not print!
    stdout, sys.stdout = sys.stdout, io.StringIO()

    config = _build_config_grid()
    result = []
    write_result_header()

    for c in config:
        # test the config item on every pb fies we got
        for pb_file in Path("./pb").iterdir():
            if pb_file.suffix == ".pb":
                time_st = time()
                res_check = detect_subgraph(
                    graph_path=str(pb_file.absolute()),
                    result_path=f"./result/{pb_file.name}.json",
                    check_result=True,
                    **c,
                )
                time_use = time() - time_st
                result.append(
                    {
                        **c,
                        "pb_name": pb_file.name,
                        "time": time_use,
                        "graph_size": res_check.graph_size,
                        "num_subgraph": res_check.num_subgraph,
                        "ave_num_subgraph_instance": mean(res_check.subgraph_count),
                        "ave_subgraph_size": mean(res_check.subgraph_size),
                        "MDL": res_check.MDL,
                        "reduce_ratio": res_check.reduce_ratio
                    }
                )
                write_result(result)
                print(pb_file, round(time_use, 2), file=stdout)

    sys.stdout = stdout


test_detect()
