from SubgraphDetection import detect_subgraph

if __name__ == "__main__":
    # detect the subgraphs and save to file
    res = detect_subgraph(
        graph_path="./ms_output_0train.pb",
        result_path="./subgraph.json")
