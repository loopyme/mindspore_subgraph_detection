from SubgraphDetection import detect_subgraph
from example.eval_result import EvalResult

if __name__ == "__main__":
    # detect the subgraphs and save to file
    res = detect_subgraph(
        graph_path="./ms_output_0train.pb",
        result_path="./subgraph.json")
    print(EvalResult("./ms_output_0train.pb", res))
