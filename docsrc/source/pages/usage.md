# 使用指南

接口十分清晰，只需要向函数传递计算图文件路径和结果路径即可。

```python
from SubgraphDetection import detect_subgraph

detect_subgraph(
    graph_path="./ms_output.pb",
    result_path="./subgraph.json")
```

运行参数在`SubgraphDetection.config`中定义，目前暂未参数调整接口，若需调整参数，需要前往该文件进行手动配置以下配置项：

|配置项|解释|可选值|默认值|
|:--:|:--:|:--:|:--:|
|`SAFE_MODE`|安全模式，是否进行一些额外的运算来保证计算中间结果是正常的|(bool) True,False|True|
|`MAX_WORKER`|并发线程数,-1表示使用CPU数目|(int)>0 or -1|-1|
|`MIN_SUBGRAPH_INSTANCE_NUMBER`|子图模式的频繁阈值，实例数小于该值的子图模式将不被接受|(int)>=0|2|
|`MIN_SUBGRAPH_NODE_NUMBER`|子图模式的大小阈值，节点数小于该值的子图模式将不被接受|(int)>=0|4|
|`SUB_SUB_GRAPH_THRESHOLD_PENALTY`|子图的子结构的罚项，考虑到某个子图的子结构可能是更频繁的子图，需要施加罚项以控制子图阶数|(int)>=0|2|