# Chapter01 图机器学习导论
## 1.1 图论 (Graph Theory)
- **图论**是数学的一个分支。它以图作为研究对象。
- **图**(graph) $G = (V, E)$ 是一个二元组 $(V, E)$ 使得 $E{\in}[V]^2$，所以 $E$ 的元素是 $V$ 的2-元子集。我们总是默认 $V{\cap}E=\varnothing$。
- 集合 $V$ 中的元素称为图 $G$ 的**顶点**(vertex)或**节点**(node)，而集合 $E$ 的元素称为**边**(edge)或**线**(line)。$G$ 的顶点集记为 $V(G)$ ，边集记为 $E(G)$。
### 1.1.1 有向图、无向图
- **有向图**是由一对不相交的集合 $(V,E)$ 以及两个映射 ${\rm{init}}:E{\rightarrow}V$ 和 ${\rm{ter}}:E{\rightarrow}V$ 组成的，其中 $\rm{init}$ 把每条边 $e$ 映到了一个初始点 ${\rm{init}}(e)$ 上，而 $\rm{ter}$ 把每条边 $e$ 映到一个终点 ${\rm{ter}}(e)$ 上。我们称边 $e$ 是从 ${\rm{init}}(e)$ **指向** ${\rm{ter}}(e)$ 的。
- 对于有向图 $D$ 和(无向)图 $G$ ，如果 $V(D)=V(G)$ ，$E(D)=E(G)$ ，且对每条边 $e=xy$ 有 $\{{\rm{init}}(e),{\rm{ter}}(e)\}=\{x,y\}$ ，则称 $D$ 是 $G$ 的一个**定向**。
### 1.1.2 顶点度
- 如果 $\{x,y\}$ 是 $G$ 的一条边，则称两个顶点 $x$ 和 $y$ 是**相邻**的或**邻点**。
- 若 $G$ 的所有顶点都是两两相邻的，则称 $G$ 是**完全的**。
- 设 $G=(V,E)$ 是一个非空图，$G$ 中顶点 $v$ 的邻点集记为 $N_G(v)$ ，简记为 $N(v)$ ， 顶点 $v$ 的**度**(degree) $d_G(v)=d(v)$ 是指关联 $v$ 的边数 ${\lvert}E(v){\rvert}$ 。由图的定义，它等于 $v$ 的邻点的个数。
- $G$ 的**平均度**(average degree)定义为 
$$d(G):=\frac{1}{{\lvert}V{\rvert}}\sum\limits_{v{\in}V}d(v)$$
### 1.1.3 路和圈
- **路**(path) $P=(V,E)$ 是一个非空图，其**顶点集**和**边集**分别为
$$V=\{x_0,x_1,\dotsc,x_k\},{\qquad}E=\{x_0x_1,x_1x_2,\dotsc,x_{k-1}x_k\}$$
这里所有的 $x_i$ 均互不相同，顶点 $x_0$ 和 $x_k$ 由路 $P$ 连接(link)，并称它们为路的**端点**(endvertex)，其余的点称为 $P$ 的**内部**顶点。
- 若 $P=x_0{\dotsc}x_{k-1}$ 是一条路且 $k\geqslant3$ ，则称图 $C:=P+x_{k-1}x_0$ 为**圈**(cycle)。
### 1.1.4 连通图
- 如果非空图 $G$ 中的任意两个顶点之间均有一条路相连，我们称 $G$ 是**连通的**(connected)。若 $U{\subseteq}V(G)$ 且 $G[U]$ 是连通的，则称(在 $G$ 中) $U$ 本身是连通的。
### 1.1.5 邻接矩阵
- 设图 $G=(V,E)$ 的顶点集是 $V=\{v_1,\dotsc,v_n\}$ ，而边集是 $E=\{e_1,\dotsc,e_m\}$ ，则它在 $\mathbb{F}_2$ 上的**关联矩阵**(incidence matrix) $B=(b_{ij})_{n{\times}m}$ 定义为
$$b_{ij}:=\left\{
    \begin{array}{lcl}
    1,{\quad}v_i{\in}e_j,\\
    0,{\quad}\rm{otherwise}\\
\end{array} \right.$$
- 图 $G$ 的**邻接矩阵**(adjacency matrix) $A=(a_{ij})_{n{\times}n}$ 定义为
$$a_{ij}:=\left\{
    \begin{array}{lcl}
    1,{\quad}v_iv_j{\in}E,\\
    0,{\quad}\rm{otherwise}\\
\end{array} \right.$$

## 1.2 图数据挖掘常见任务介绍
### 1.2.1 常见任务
- 路径选择问题<br>
例如：地图导航选择最短路径
- 度中心性评价<br>
例如：用PageRank算法对搜索结果排序
- 社群检测<br>
例如：对网络中虚假、伪造信息的监测
-连接预测<br>
例如：推荐系统
- 节点相似度<br>
例如：分析不同年代历史人物相似性
- 图嵌入<br>
例如：NodetoVec
- 图生成<br>
例如：设计新的药物分子
- 图的演化<br>
例如：粒子模拟
 
### 1.2.2 典型方法
- 将实际问题转化为图论语言，运用数学方法将其转化为矩阵和向量信息，然后训练图神经网络来解决相关问题。
- 关键在于要找到一个好的图嵌入，把每个节点信息转化为低维向量并尽量减少信息的流失。
- 然后构造高效高质量的图神经网络来预测结果。

### 1.2.3 应用场景举例
**AlphaFold预测蛋白质空间结构**
- 传统方法：在实验室中使用冷冻电镜、核磁共振等技术手动的测定蛋白质的空间结构。要花费大量的人力物力财力以及时间成本。
- 人工智能辅助：输入多肽链的一级结构就可以预测出蛋白质的三维结构。<br>
- AlphaFold团队目前已经预测了人类约98.5%的蛋白质结构，其在GDT测试中的分数也远超其他模型，置信度非常高。


## 1.3 图神经网络常见模型和编程工具
### 1.3.1 图神经网络常见模型
- 传统模型：Graphlets， Graph Kernels
- 图嵌入模型：DeepWalk， Node2Vec
- 图神经网络：GCN， GraphSAGE， GAT， Theory of GNNs
- 知识图谱和推理：TransE， BetaE
- 深度图生成模型：GraphRNN
- 生物医学、AI for Science、 产业界的应用

### 1.3.2 编程工具及可视化引擎
- PyG（PyTorch Geometric）: https://www.pyg.org/
- SNAP.PY: https://snap.stanford.edu/snappy/
- NetworkX: https://networkx.org/
- DGL(Deep Graph Library): https://www.dgl.ai/
- AntV可视化(g6引擎): https://g6.antv.antgroup.com/zh
- Echarts可视化(GL关系图): https://echarts.apache.org/examples/zh/index.html#chart-type-graphGL
- GraphXR可视化: https://www.kineviz.com/graphxr
- Neo4j(图数据库): https://neo4j.com/

## 1.4 本章总结
&emsp;&emsp;本章主要学习图论的基础概念，图机器学习主要解决哪些实际问题，以及常见的技术手段及方法。