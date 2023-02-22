# Chapter06 PageRank
## 6.1 PageRank简介
- PageRank是Google专有的算法，用于衡量特定网页相对于搜索引擎索引中的其他网页而言的重要程度。它由Larry Page 和 Sergey Brin在20世纪90年代后期发明。
- PageRank算法的基本想法是在有向图上定义一个随机游走模型，沿着有向图随机访问各个结点。迭代优化后收敛，得到一个平稳分布，这时各个结点的概率值就是其PageRank值，代表了各个结点的重要度。
### 6.1.1 PageRank的历史地位
- PageRank造就了传奇的谷歌公司。
- 搜索引擎、信息检索、图机器学习、图数据挖掘的必读论文。
- 线性代数的应用典范。
### 6.1.2 PageRank的思想
- 将互联网当做一个整体的系统，网页与网页之间存在关联，不是孤立的个体。把整体看作一个图来处理。
- 2000年左右的互联网是一个较为静态的网络，而现如今的互联网则在不断的更新，有很多随时生成的网页，并且还有很多是"暗节点"(非公用网络)，因此现在的网页不再是一个固定不变的图。

## 6.2 PageRank算法概述
- 网络是一个有向图，网页是节点，网页间的链接是节点间的连接。
### 6.2.1 PageRank节点重要度的几种算法
- PageRank的基本假设是：每个网站的In-link是不同的，能够反映一个网站的重要程度；同时，由更重要的网站所引出的In-link则会提升更多的网站重要度。
- 问题描述：将网络抽象为一个图 $G=(V,E)$ ，$V$ 是节点集，$E$ 是边集。
- 目的是求解每个节点的节点重要度，记作PR值(PageRank值)。
#### 求解线性方程组算法
- 定义某个节点的PR值等于所有指向该节点的节点的PR值加权求和：
$${\rm{PR}}_u=\sum\limits_{v{\in}N(u)}\lambda_{vu}{\rm{PR}}_v$$
- 对于每个节点，可以根据上述假设，得到一个关于该节点PR值的方程。
- 联立网络中所有节点PR值方程，以及下述方程：
$$\sum\limits_{u{\in}V}{\rm{PR}}_u=1$$
通过求解线性方程组，得到所有节点的PR值。
- 由于网络过于庞大，使得需要联立的方程过多，求解困难，无法实际采用。
#### 幂迭代算法
- 1.同样的，对于一个有向图，每个节点的PR值等于其邻点的PR值加权求和：
$${\rm{PR}}_u=\sum\limits_{v{\in}N(u)}\lambda_{vu}{\rm{PR}}_v$$
- 2.假设当用户停留在某节点时，跳转到其他节点的概率相同，那么平均分配给每个outlink相同的权重，用权重矩阵 $M$ (Stochastic adjacency matrix)来表示所有节点的权重图。
- 3.为每个节点设定初值构建pagerank向量 $r=(r_1,\cdots)$，$\sum\limits_{i}r_i=1$，左乘权重矩阵则会得到一次迭代的PR值，多次迭代收敛到最终结果($t$为迭代次数)：
$$r_{\rm{ter}} = M^{t}r$$
#### 随机游走算法
- 假设从一个节点遍历图，引入阻尼系数，以$\alpha$的概率沿有向边游走，以$1-\alpha$的概率随机游走到任意节点，得到所有节点随机游走的概率分布。
- 最终如果收敛，那么收敛的概率分布结果就是pagerank值，算法数学表达式与幂迭代相同，同样是求权重矩阵 $M$ 的特征向量。
#### 马尔科夫链算法
- 每个节点是一个状态，节点之间的连接对应状态转移，应用求解马尔科夫链的方法也可以同样求解。
### 6.2.2 Pagerank节点重要度算法分析
- 在实际求解网页pagerank时，随机游走算法需要模拟大量的游走，马尔科夫链方法同样需要求解大量的转移矩阵，而采样幂迭代方法则只需要计算机最擅长的矩阵乘法。
- 因此实际上所采用的就是幂迭代算法。
#### PageRank收敛性分析
- 收敛性问题：
    - 是否能收敛至稳定值？
    - 不同初始值，是否收敛至同一个结果？
    - 收敛的结果在数学上是否有意义？
    - 收敛的结果是否真的代表节点重要度？
- 问题解答：
    - 互联网满足遍历定理(Ergodic Theorem)：对于不可约的(Irreducible)非周期的(Aperiodic)马尔科夫链，存在一个唯一的稳定分布 $\pi$ ，并且所有的初始分布 $\pi_0$ 都收敛到 $\pi$ 。
#### 幂迭代算法特殊情况
- 幂迭代终止点问题：实际应用中，有一些网页不指向任何网页，会导致迭代收敛到0。
- 幂迭代陷阱问题：如果一个网页不存在指向其他网页的链接，但存在指向自己的链接，也会导致迭代问题。
- 幂迭代算法改进：引入基尼系数$\alpha$，每个节点有$1-\alpha$的概率跳转到其他任意节点：
$$r^\prime={\alpha}Mr+(1-\alpha)\left[\frac{1}{N}\right]_{N{\times}N}$$
$N$ 为节点总数。
### 6.2.3 计算节点相似度
#### Personalized PageRank
- 个性化PageRank被用于节点被分为多类的情况。例如，在推荐系统中，全图拥有用户节点和商品节点两类节点，假设同一个用户买到的商品具有一定的相似度，那么可以通过随机传送回用户节点，来计算商品节点的相似度。
#### Random Walk with Restarts
- 想要计算 $Q$ 节点和其他节点的相似度，可以从 $Q$ 节点出发，模拟多次的随机游走，并且具有一定概率随机传送回 $Q$ 节点，再通过随机游走的访问次数来反映其他节点和 $Q$ 节点之间的亲疏远近。

## 6.3 代码实战
### 6.3.1 四大名著数据集读取
- 导入工具包
```python
import networkx as nx                               # 图数据挖掘
import numpy as np                                  # 数据分析
import random                                       # 随机数
import pandas as pd
# 数据可视化
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei']          # 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus']=False            # 用来正常显示负号
```
- [OpenKG-四大名著人物关系知识图谱和OWL本体](http://www.openkg.cn/dataset/ch4masterpieces)
```python
# 导入 csv 文件定义的有向图
# df = pd.read_csv('data/西游记/triples.csv')
df = pd.read_csv('data/三国演义/triples.csv')
```
### 6.3.2 生成有向图
```python
edges = [edge for edge in zip(df['head'], df['tail'])]
G = nx.DiGraph()
G.add_edges_from(edges)
```
- 常规可视化
```python
plt.figure(figsize=(15,14))
pos = nx.spring_layout(G, iterations=3, seed=5)
nx.draw(G, pos, with_labels=True)
plt.show()
```
### 6.3.3 计算每个节点PageRank重要度
- pagerank函数计算有向图的节点重要度，无向图会自动转为双向图再进行计算
```python
pagerank = nx.pagerank(G,                           # NetworkX graph 有向图，如果是无向图则自动转为双向有向图
                       alpha=0.85,                  # Damping Factor
                       personalization=None,        # 是否开启Personalized PageRank，随机传送至指定节点集合的概率更高或更低
                       max_iter=100,                # 最大迭代次数
                       tol=1e-06,                   # 判定收敛的误差
                       nstart=None,                 # 每个节点初始PageRank值      
                       dangling=None,               # Dead End死胡同节点
                      )
```
### 6.3.4 用节点尺寸可视化PageRank值
- 设置节点尺寸和节点颜色
```python
# 节点尺寸
node_sizes = (np.array(list(pagerank.values())) * 8000).astype(int)
# 节点颜色
M = G.number_of_edges()
edge_colors = range(2, M + 2)
```
- 绘制图像
```python
plt.figure(figsize=(15,14))
# 绘制节点
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_sizes)
# 绘制连接
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,                           # 节点尺寸
    arrowstyle="->",                                # 箭头样式
    arrowsize=20,                                   # 箭头尺寸
    edge_color=edge_colors,                         # 连接颜色
    edge_cmap=plt.cm.plasma,                        # 连接配色方案，可选：plt.cm.Blues
    width=4                                         # 连接线宽
)
# 设置每个连接的透明度
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])
ax = plt.gca()
ax.set_axis_off()
plt.show()
```

## 6.4 本章总结
本章主要学习了Google的PageRank算法，主要包括：
- PageRank节点重要度幂迭代算法
- 随机游走算法和马尔科夫链算法
- PageRank收敛性讨论
- 幂迭代的特殊情况
- 四大名著数据集节点重要度代码实战