# 使用指南

## 安装
首先需要手动安装一些依赖项，也即本项目的上游项目Mindspore和Mindinsight，建议到[Mindspore文档](https://www.mindspore.cn/install)中查询相关安装方式。

本项目托管于Gitee，并在Github和中科院软件所智能软件研究中心Gitlab上保持镜像，可以从下面的任何托管仓库下载本项目的Release包，若想获得最新版本，也可以clone repo。
 - https://gitee.com/loopyme/mindspore_subgraph_detection
 - https://github.com/loopyme/mindspore_subgraph_detection
 - https://isrc.iscas.ac.cn/gitlab/summer2020/students/proj-2017182

下载完毕后，在你习惯的Python环境下，只需运行```python setup.py install```就能安装SubgraphDetection

接下来，可以在终端中输入```detect-subgraph -v```查看安装的项目版本，若出现版本号，安装就成功了。

## 运行

### 终端运行

安装结束后，在命令行输入```detect-subgraph -h```以查看所有的SubgraphDetection命令行参数和使用方法。
```sh
usage: detect-subgraph [-h] [-v] [--verbose] [--safe-mode] [-w MAX_WORKER]
                       [-i MIN_SUBGRAPH_INSTANCE_NUMBER]
                       [-n MIN_SUBGRAPH_NODE_NUMBER]
                       [-p SUB_SUB_GRAPH_THRESHOLD_PENALTY] [--check_result]
                       graph path result path

Detect subgraphs in a Mindspore computational graph

positional arguments:
  graph path            The path of the pb file where the whole graph are
                        stored
  result path           The path of json file where the detected subgraphs
                        should be dumped.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --verbose             Print details to console
  --safe-mode, -s       Do some extra computation to make sure safety
  -w MAX_WORKER, --worker MAX_WORKER
                        The worker number of Thread Pool, -1 = cqu_count
  -i MIN_SUBGRAPH_INSTANCE_NUMBER, --min-instance MIN_SUBGRAPH_INSTANCE_NUMBER
                        The minimum instance number of a subgraph, subgraph
                        with fewer instances will not be detected
  -n MIN_SUBGRAPH_NODE_NUMBER, --min-nodes MIN_SUBGRAPH_NODE_NUMBER
                        The minimum node number of a subgraph, subgraph
                        instance with fewer nodes will not be detected
  -p SUB_SUB_GRAPH_THRESHOLD_PENALTY, --penalty SUB_SUB_GRAPH_THRESHOLD_PENALTY
                        Impose penalty terms on sub-sub-graph in thresholds to
                        avoid multiple level subgraphs
  --check_result, -c    Check the result after finish calculation
```
若使用默认配置项，使用下面的命令就能触发子图挖掘运算（后面的两个参数分别为计算图文件路径和预设的结果输出路径）

```sh
detect-subgraph ./ms_output.pb ./subgraph.json
```

### Python中运行
接口十分清晰，只需要向函数传递计算图文件路径和结果路径即可。

```python
from SubgraphDetection import detect_subgraph

detect_subgraph(
    graph_path="./ms_output.pb",
    result_path="./subgraph.json")
```

也可以在后面附加上一些其他配置项：

```python
from SubgraphDetection import detect_subgraph

detect_subgraph(
    graph_path="./ms_output.pb",
    result_path="./subgraph.json"，
    check_result = True,
    verbose=True,
    max_worker=4)
```

## 运行参数

目前已提供了简洁的参数调整接口，以下运行参数都可自行调整，其中项目的高效执行严重依赖于`MIN_SUBGRAPH_INSTANCE_NUMBER`和`MIN_SUBGRAPH_NODE_NUMBER`的正确选取。

|配置项|解释|可选值|默认值|
|:--:|:--:|:--:|:--:|
|`SAFE_MODE`|安全模式，是否进行一些额外的运算来保证计算中间结果是正常的|(bool) True,False|False|
|`VERBOSE`|详细的输出|(bool)True,False|False|
|`CHECK_RESULT`|自动的检查输出的各项参数|(bool)True,False|False|
|`MAX_WORKER`|并发线程数,-1表示使用CPU数目|(int)>0 or -1|-1|
|`MIN_SUBGRAPH_INSTANCE_NUMBER`|子图模式的频繁阈值，实例数小于该值的子图模式将不被接受|(int)>=0|2|
|`MIN_SUBGRAPH_NODE_NUMBER`|子图模式的大小阈值，节点数小于该值的子图模式将不被接受|(int)>=0|4|
|`SUB_SUB_GRAPH_THRESHOLD_PENALTY`|子图的子结构的罚项，考虑到某个子图的子结构可能是更频繁的子图，需要施加罚项以控制子图阶数|(int)>=0|2|
