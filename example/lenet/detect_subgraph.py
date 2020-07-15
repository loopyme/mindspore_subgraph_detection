from SubgraphDetection import detect_subgraph
from example.lenet.train.__main__ import train_lenet

if __name__ == "__main__":
    # From lenet Tutorial,train the net and save the graph
    train_lenet()

    # detect the subgraphs and save to file
    res = detect_subgraph(
        graph_path="./ms_output.pb",
        result_path="./subgraph.json")
