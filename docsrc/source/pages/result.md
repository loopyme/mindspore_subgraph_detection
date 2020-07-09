# 测试结果

目前在以下项目产生的计算图上进行了测试，测试结果如下表：

|计算图来源|原计算图大小|子图模式个数|子图个数|子图模式大小|MDL|压缩比|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[lenet_mnist](https://gitee.com/mindspore/mindspore/tree/r0.3/example/lenet_mnist)|30|1|(2)|(8)|1.36|0.27|
|[lstm_aclImdb](https://gitee.com/mindspore/mindspore/tree/r0.3/example/lstm_aclImdb)|145|2|(2,6)|(6,10)|1.63|0.39|

