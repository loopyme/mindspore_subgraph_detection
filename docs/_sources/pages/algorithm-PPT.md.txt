---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.jpg')
header: 深度学习模型计算图相同子结构的识别和展示
---

# <!-- fit -->深度学习模型计算图相同子结构的识别和展示

> 本项目受到 [“开源软件供应链点亮计划-暑期2020” ](https://isrc.iscas.ac.cn/summer2020/#/index) 的资助和支持,由peter@mail.loopy.tech设计开发，由gaohan19@huawei.com指导。



---

## 1. 简介

MindSpore是华为自研的深度学习框架，其中的计算图模式是一种业界主流的用来进行数据传递与计算的形式。计算图主要包含节点和有向边，节点表示计算和控制操作，边表示数据的流向和控制等关系。计算图的高效合理展示，有助于用户更好的理解模型结构、发现和调试模型训练过程中出现的问题。然而，大型深度学习模型往往有着复杂的计算图结构，包含有成千上万个节点和更多的边。在这些点和边之中，包含有许多结构相同或高度相似的子结构，这些子结构不仅从图的拓扑结构上，甚至从深度学习语义上具有高度的相似性。快速识别大型计算图中上述的相同子结构，能够支持后续用收折、重叠等方式大幅减少页面中同时呈现的节点和边的数目，从而大幅改善计算图的展示效果。

---

## 2. 为什么需要新算法？

当前应用情景与常规子图挖掘大不相同：

 - 度量方式：支持度变得不那么重要，而应以压缩输入图的程度来度量
 - 单图挖掘：在单个大图中寻找子图，而不是多个小图
 - 有向无环图：计算图是有向无环图
 - 无边权：计算图的所有边都是相同的
 - 算法级并行：在并行场景下能快速稳定执行
 - 层次化算子节点：计算图中的各个算子节点是具有层级的

由此，我们基于Apriori思想改进了现有的频繁子图挖掘算法，并使用Python进行了编码实现

---

## 3. 算法简述

本项目分为两个模块，分别用于形成节点集和在特定节点集中进行子图挖掘。

整个算法基于一个基本的事实规律：**频繁子图的任意子图或子节点都是频繁的**。由此，我们只需要找到频繁的节点，并由频繁的节点开始进行聚合和生长，就能找到需要的频繁子图。

---
### 3.1. 节点集形成算法

节点集形成算法就是从所有节点中筛选出某个特定层级的节点，形成待挖掘的节点集。主要难度在于，各个节点存储的上下游节点都是level-1的节点，为了在筛选中保持图的结构，需要将上下游节点重定向至特定层级的节点，需要递归的查询各个节点的符合要求的祖先。

右侧给出了算法伪代码：
![bg right:35% 85%](https://loopyme.gitee.io/mindspore_subgraph_detection/_images/algorithm-1.png)

---

### 3.2. 子图挖掘算法

本项目所使用的子图挖掘算法，基于Apriori思想进行改进，使用“自下而上”的方法，首先生成单节点的核，核进行多次生长，每次生长后移除原来的核，当所有核都生长为子图集后，算法终止。

右侧给出了算法伪代码：
![bg right:40% 80%](https://loopyme.gitee.io/mindspore_subgraph_detection/_images/algorithm-2.png)

---
## 4. 实现

以下简述算法的实现的相关思路和方式

---

### 4.1. 数据结构

#### 4.1.1 节点

**节点**由`SNode`类所定义的数据结构存储，每个节点既保存了上下游节点（保存计算图中的联系），也保存了自己所属的命名空间（保存节点树中的联系）。`Scope`类继承自`SNode`，在其基础上，`Scope`还额外存储了自己的组成节点。为了避免循环引用，`SNode`和`Scope`在存储其它节点时，只存储了ID，具体节点信息需要到`SMSGraph`中去查询。同时，它们的设计采用了鸭子类型的风格，在子图挖掘阶段，两种类型的实例是等效的。

---
#### 4.1.2 计算图
**计算图**由`SMSGraph`类所定义的数据结构存储，它保存了一个计算图中的全部信息，并提供了丰富的接口以接受各种查询。它的另一个主要功能是解析`MSGraph`对象，将其数据按本项目设计的方式整理和清洗。

**子图的层次结构**没有专门的类定义存储，而是使用节点存储的信息结合计算图的查询接口完成查询。在需要递归建树或查询时，使用猴子补丁的方式，将函数临时附加到节点类上，以完成需要的功能。

---
#### 4.1.4 子图
**子图**相关信息由`Subgraph`类所定义的数据结构存储（生长时为`SubgraphCore`），与大多数编码实现不同，本项目中的图没有保存边的关系，而是按序保存了节点的关系，即子图模式。在子图挖掘中，核（子图核集）为核心数据结构，所有调度和控制都围绕着核展开。比如`Node1(biaAdd)->Node2(Conv2D)`与`Node3(biaAdd)->Node4(Conv2D)`同构，则子图核集保存的是`{pattern:['biaAdd', 'Conv2D']，nodes:[(1,2), (3,4)]}`，ID为`hash("1-2")`.为了保证核模式在遍历时只遍历边界点，并减少功能耦合，核可以作为迭代器，只有边界点会被遍历取得。

---
### 4.2. 并行调度与执行

为了充分使用硬件并加快速度，子图挖掘部分算法执行采用Map-Reduce机制，执行器中的计算池存储着前一生长周期被提出的核，通过线程池调度的方式进行生长计算，从而获得下一生长周期的新计算池。每个核的生命周期都为一个生长周期，在生长周期结束后可能被销毁或拷贝提交。同时使用了子图核集的注册机制以避免冗余计算，每一子图核集(除单节点的外)在被提出时都需要通过执行器进行注册(注册需要是线程安全的)，冗余的核将不被加入计算池。

---
### 4.3. 用户使用

- 本项目可以按照Python Package的标准进行分发和安装，具体安装方式见[安装指南](https://loopyme.gitee.io/mindspore_subgraph_detection/pages/usage.html#id2)
 - 同时也已实现命令行运行接口，具体安装方式见[使用指南](https://loopyme.gitee.io/mindspore_subgraph_detection/pages/usage.html#id3)

![bg left:40% 80%](https://loopyme.gitee.io/mindspore_subgraph_detection/_images/console_run.png)

---
## 5. 其他相关链接

- [API文档](https://loopyme.gitee.io/mindspore_subgraph_detection/pages/api/detect_subgraph.html)
- [更新日志](https://loopyme.gitee.io/mindspore_subgraph_detection/pages/changelog.html)
- [测试结果](https://loopyme.gitee.io/mindspore_subgraph_detection/pages/result.html)
- [参考文献](https://loopyme.gitee.io/mindspore_subgraph_detection/pages/reference.html)
- [外部依赖](https://loopyme.gitee.io/mindspore_subgraph_detection/pages/dependencies.html)
