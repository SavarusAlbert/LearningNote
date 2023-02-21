# Node2Vec论文精读
## 5.6 Node2Vec前瞻
- DeepWalk用完全的随机游走来训练图嵌入向量，仅能反映相邻节点的社群相似信息，无法反映距离较远但功能结构相似的节点信息。
- Node2Vec采用有偏的随机游走，通过预先设定两个参数，来控制随机游走的偏向性。</br>
1.Return参数 $p$ ：以 $\frac{1}{p}$ 的概率返回到上一个节点</br>
2.In-out参数 $q$ ：以 $\frac{1}{q}$ 的概率走向更远的节点</br>
3.以"1"的概率走向其它节点，再对概率进行归一化，得到最终的概率，采样出下一个随机游走。
- $p$ 大 $q$ 小对应于DFS，探索全局宏观区域；$q$ 大 $p$ 小对应于BFS，探索局部微观区域。
- DeepWalk是Node2Vec在 $p=1,q=1$ 的特例。

## 5.7 Node2Vec优劣
### 5.7.1 优点
- 通过调节 $p$ 、 $q$ 值，实现有偏随机游走，探索节点社群、功能等不同属性
- 首次把节点分类用于Link Prediction
- 可解释性、可扩展性好，性能卓越
### 5.7.2 缺点
- 需要采样大量的随机游走序列进行训练
- 距离较远的两个节点无法直接相互影响。看不到全图信息
- 无监督，仅编码图的连接信息，没有利用节点的属性特征
- 没有真正用到神经网络和深度学习

## 5.8 Node2Vec文章部分
**Scalable Feature Learning for Networks**
- 可扩展的图嵌入表示学习算法
### 5.8.1 摘要
- Node2Vec是学习网络中节点的连续特征表示的算法框架。
- 目的是学习一个映射，将每个节点变为一个低维连续稠密的向量。
- 基本原理是使用极大似然估计来进行模型的训练。
- 定义了一个灵活的有偏随机游走的范式，可以探索网络中多样化的属性和特征。
- 在多类别分类任务和连接预测任务上都得到了最好的实验效果。

### 5.8.2 问题描述
- $G = (V, E)$ 是一个给定的网络，$V$ 是节点集， $E$ 是边集。
- 目标是学习一个从节点到特征表示空间的映射：
$$f:V\rightarrow\mathbb{R}^d$$
其中 $d$ 是特征空间的维数。
- 定义 $N_S(u)$ 为节点 $u$ 在采样策略 $S$ 下的邻域。
- 目的是迭代优化下面的函数：
$$\mathop{\rm{max}}\limits_{f}\quad\sum\limits_{u{\in}V}{\rm{log}}{\rm{Pr}}(N_S(u)|f(u))$$
用中间节点预测周围节点，使预测的概率最大化。

### 5.8.3 问题转化
- 为了让问题更容易处理，作以下两个假设：</br>
1.条件独立假设。周围节点互不影响，等价于马尔可夫假设，从而将问题转化为连乘的形式：
$${\rm{Pr}}(N_S(u)|f(u))=\prod\limits_{n_i{\in}N_S(u)}{\rm{Pr}}(n_i|f(u))$$
2.对称性假设。两个节点之间相互影响的程度一样。</br>
由假设2，可以将传统的似然函数求解，转变为求下列的softmax分类：
$${\rm{Pr}}(n_i|f(u))=\frac{{\rm{exp}}(f(n_i){\cdot}f(u))}{\sum_{v{\in}V}{\rm{exp}}(f(v){\cdot}f(u))}$$
联立上式并进行变换后得到：
$$\mathop{\rm{max}}\limits_{f}\quad\sum\limits_{u{\in}V}[-{\rm{log}}Z_u+\sum\limits_{n_i{\in}N_S(u)}f(n_i){\cdot}f(u)]$$
其中$Z_u=\sum_{v{\in}V}{\rm{exp}}(f(v){\cdot}f(u))$。</br>
- 可以通过分层softmax来降低算法复杂度，从而计算 $Z_u$ 。
- 整个随机过程基于Skip-gram架构

### 5.8.4 BFS和DFS
一般来说，对于邻域集 $N_S$ 有两种采样方式
- 广度优先搜索(BFS)：邻域 $N_S$ 优先选取目标节点的近邻，提供了图的微观视角。
- 深度优先搜索(DFS)：邻域 $N_S$ 会在目标节点越来越远的位置逐步采样，提供了图的宏观视角。

### 5.8.5 Node2Vec
基于上述的讨论，Node2Vec设计了一种灵活的采样方式，可以方便在BFS和DFS之间进行调节。
#### 随机游走
- 令 $c_i$ 表示第 $i$ 个随机游走节点，给定起始节点 $c_0=u$ ，当前节点为 $v$ ，上一个节点 $t$ ，下一节点为 $x$ 的概率服从下列分布函数：
$$P(c_i=x|c_{i-1}=v)=\left\{
    \begin{array}{lcl}
    \frac{\pi_{vx}}{Z} &{\quad}{\rm{if}}(v,x){\in}E\\
    0 &{\quad}\rm{otherwise}\\
\end{array} \right.$$
其中 $\frac{\pi_{vx}}{Z}$ 是转移概率，$Z$ 是归一化的常数。

#### 搜索偏重 $\alpha$
- 最简单的采样策略是将权重作为偏向：
$$\pi_{vx}=w_{vx}$$
但这种方式无法对策略进行灵活的调节。
- 一阶随机游走：下一个节点仅与当前节点有关。(DeepWalk、PageRank)
- 二阶随机游走：下一节点不仅与当前节点有关，还与上一节点有关。(Node2Vec)
- 用两个参数 $p$ 和 $q$ 来控制随机游走策略：
$$\alpha_{pq}(t,x)=\left\{
    \begin{array}{lcl}
    \frac{1}{p} &{\quad}{\rm{if}}d_{tx}=0\\
    1 &{\quad}{\rm{if}}d_{tx}=1\\
    \frac{1}{q} &{\quad}{\rm{if}}d_{tx}=2\\
\end{array} \right.$$
其中 $d_{tx}$ 是上一节点 $t$ 和下一节点 $x$ 的最短路径长度。
- 再点乘上权重并进行归一化，得到最终的随机游走概率：
$$\frac{\pi_{vx}}{Z}=\frac{\alpha_{pq}(t,x){\cdot}w_{vx}}{Z}$$

#### 二阶随机游走算法复杂度
- 空间上，要存储当前节点和上一节点的邻点信息，因此空间复杂度为 $O(a^2|V|)$ ，其中 $a$ 是平均连接数。
- 时间上，所有节点都被采样的时间复杂度为 $O(\frac{1}{k(l-k)})$ 。

### 5.8.6 算法伪代码
#### 算法1 The node2vec algorithm

> **LearnFeatures** (${\rm{Graph}}\ G=(V,E,W),\ {\rm{Dimensions}}\ d,\ {\rm{Walks\ per\ node}}\ r$,  
&emsp;&emsp;${\rm{Walk\ length}}\ l,\ {\rm{Context\ size}}\ k,\ {\rm{Return}}\ p,\ {\rm{In}}$-${\rm{out}}\ q$)  
&emsp;&emsp;$\pi={\rm{PreprocessModifiedWeights}}(G,p,q)$&emsp;&emsp;(生成随机游走采样策略)  
&emsp;&emsp;$G^\prime=(V,E,\pi)$  
&emsp;&emsp;${\rm{Initialize}}\ walks\ {\rm{to\ Empty}}$&emsp;&emsp;(初始化)  
&emsp;&emsp;**for** $iter=1$ **to** $r$ **do**&emsp;&emsp;(每个节点生成 $r$ 个随机游走序列)  
&emsp;&emsp;&emsp;&emsp;**for all** ${\rm{nodes}}\ u{\in}V$ **do**  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$walk={\rm{node2vecWalk}}(G^\prime,u,l)$&emsp;&emsp;(生成1个随机游走序列)  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;${\rm{Append}}\ walk\ {\rm{to}}\ walks$  
&emsp;&emsp;$f={\rm{StochasticGradientDescent}}(k,d,walks)$&emsp;&emsp;(Skip-Gram训练得到节点嵌入表)  
&emsp;&emsp;**return** $f$

#### 算法2 node2vecWalk

> **node2vecWalk**(${\rm{Graph}}\ G^\prime=(V,E,\pi),\ {\rm{Start\ node}}\ u,\ {\rm{Length}}\ l$)  
&emsp;&emsp;${\rm{Inititalize}}\ walk\ {\rm{to}}\ [u]$&emsp;&emsp;(生成1个随机游走序列)  
&emsp;&emsp;**for** $walk{\_}iter=1$ **to** $l$ **do**&emsp;&emsp;($l$ 个节点)  
&emsp;&emsp;&emsp;&emsp;$curr=walk[-1]$&emsp;&emsp;(当前节点)  
&emsp;&emsp;&emsp;&emsp;$V_{curr}={\rm{GetNeighbors}}(curr,G^\prime)$&emsp;&emsp;(当前节点的邻点)  
&emsp;&emsp;&emsp;&emsp;$s={\rm{AliasSample}}(V_{curr},\pi)$&emsp;&emsp;(根据采样策略，找到下一个节点)  
&emsp;&emsp;&emsp;&emsp;${\rm{Append}}\ s\ {\rm{to}}\ walk$  
&emsp;&emsp;**return** $walk$

## 5.9 代码实战
### 5.9.1 悲惨世界人物关系数据集
- 安装和导入工具包
```python
!pip install node2vec                               # 安装node2vec工具包
import networkx as nx                               # 图数据挖掘
import numpy as np                                  # 数据分析
import random                                       # 随机数
# 数据可视化
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei']          # 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus']=False            # 用来正常显示负号
```
- 导入数据集
```python
G = nx.les_miserables_graph()                       # 《悲惨世界》人物数据集
```
### 5.9.2 构建Node2Vec模型
```python
from node2vec import Node2Vec
# 设置node2vec参数
node2vec = Node2Vec(G, 
                    dimensions=32,                  # 嵌入维度
                    p=1,                            # 回家参数
                    q=3,                            # 外出参数
                    walk_length=10,                 # 随机游走最大长度
                    num_walks=600,                  # 每个节点作为起始节点生成的随机游走个数
                    workers=4                       # 并行线程数
                   )

# p=1, q=0.5, n_clusters=6  # DFS深度优先搜索，挖掘同质社群
# p=1, q=2, n_clusters=3    # BFS宽度优先搜索，挖掘节点的结构功能

# 训练Node2Vec
model = node2vec.fit(window=3,                      # Skip-Gram窗口大小
                     min_count=1,                   # 忽略出现次数低于此阈值的节点（词）
                     batch_words=4                  # 每个线程处理的数据量
                    )
X = model.wv.vectors
```
### 5.9.3 节点聚类算法
```python
# KMeans聚类
from sklearn.cluster import KMeans
import numpy as np
cluster_labels = KMeans(n_clusters=3).fit(X).labels_
# 将词汇表的节点顺序转为networkx中的节点顺序
colors = []
nodes = list(G.nodes)
for node in nodes:                                  # 按 networkx 的顺序遍历每个节点
    idx = model.wv.key_to_index[str(node)]          # 获取这个节点在 embedding 中的索引号
    colors.append(cluster_labels[idx])              # 获取这个节点的聚类结果
```
- 作图
```python
plt.figure(figsize=(15,14))
pos = nx.spring_layout(G, seed=10)
nx.draw(G, pos, node_color=colors, with_labels=True)
plt.show()
```
### 5.9.4 Embedding降维
```python
# 将Embedding用PCA降维到2维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embed_2d = pca.fit_transform(X)
# 将Embedding用TSNE降维到2维
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, n_iter=5000)
embed_2d = tsne.fit_transform(X)
```
- 查看相似度节点
```python
model.wv.get_vector('Napoleon')                     # 查看某个节点的Embedding
model.wv.most_similar('Napoleon')                   # 查找 Napoleon 节点的相似节点
model.wv.similarity('Napoleon', 'Champtercier')     # 查看任意两个节点的相似度
```
### 5.9.5 对连接(Edge)做图嵌入
```python
from node2vec.edges import HadamardEmbedder
# Hadamard 二元操作符：两个 Embedding 对应元素相乘
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
# 计算所有 Edge 的 Embedding
edges_kv = edges_embs.as_keyed_vectors()
# 查看与某两个节点最相似的节点对
edges_kv.most_similar(str(('Bossuet', 'Valjean')))
```

## 5.10 本章总结
本章主要对Node2Vec文章进行精读，主要包括：
- 二阶随机游走概念
- BFS和DFS的算法借鉴
- 有偏随机游走算法设计
- Node2Vec算法伪代码
- 悲惨世界人物关系图嵌入代码实战