# chapter02 图的基本表示和特征工程
## 2.1 图的基本表示
### 2.1.1 图的本体设计
- 图 $G = (V, E)$ 包含了顶点集 $V(G)$ 和边集记为 $E(G)$ ，在面对实际问题时，首先就要设计好图包括哪些节点以及这些节点之间有哪些关系，也就是设计顶点集和边集。
- 图的本体设计取决于"想要通过图来解决什么样的问题"。

### 2.1.2 图的分类
- 有向图与无向图：(见1.1.1)
- 异质图：有不同类型的节点、不同类型的边的图</br>
二分图：由两类节点组成的异质图
- 简单图与多重图：不存在重复边，不存在节点到自身的边的图称为简单图；简单图以外的图称为多重图
- 连通图：(见1.1.4)
- 非连通图：由1个以上的连通域(connected component)组成的图</br>
非连通图的邻接矩阵是分块对角阵
- 强连通有向图：有向图中，任意两节点都存在双向的连通路</br>
强连通域(SCCs)：强连通的子图；指向SCC的称为In-component，指出SCC的称为Out-component
- 弱连通有向图：有向图中，不考虑边的方向，由一个连通域组成的图
- 带权重的图：边上带有权重的图

### 2.1.3 图的基本表示
- 邻接矩阵</br>
邻接矩阵(见1.1.5)，在实际应用中，节点的个数往往是远多于连接的个数，因此邻接矩阵很可能被大量的"0"填充，是一个稀疏矩阵，不利于模型的训练。
- 连接列表和邻接列表</br>
**连接列表**：只记录存在连接的节点对，相当于边集 $E(G)$ 。</br>
数据形式是 $m$ 个三元组的形式( $m$ 是边的个数)： $(v_i, v_j, e_{ij})$ 。</br>
**邻接列表**：每个节点 $v_i$ 记录一个与之连接的节点列表。</br>
数据形式是 $n$ 个列表( $n$ 是节点的个数)： $(v_i: v_{i1},\dotsc,v_{ik})$

## 2.2 传统图机器学习的特征工程
传统图机器学习：通过人工设计特征，将节点、边或全图特征编码为低维向量，再进行后续机器学习的预测。
### 2.2.1 节点层面特征工程
- 属性特征：节点自身所带有的信息</br>
例如：节点的权重；节点的类型；节点自身信息
- 连接特征：图中某个节点与其他点的连接关系</br>
例如：节点是桥接，是枢纽，还是边缘节点
- 常用的节点特征</br>
1.度中心性(Degree Centrality)</br>
2.特征向量中心性(Eigenvector Centrality)($\lambda$ 是用来归一化的常数)：</br>
$$e_u = \frac{1}{\lambda}\sum\limits_{v{\in}N(v)}\mathcal{A}[u,v]e_v$$
节点 $u$ 的**特征向量中心性**衡量了节点 $u$ 附近节点的重要程度。</br>
重写上述公式可以得出，特征向量中心性等价于是求矩阵 $\mathcal{A}$ 的特征向量：
$${\lambda}e=\mathcal{A}e$$
(其中 $\mathcal{A}$ 是图的邻接矩阵)</br>
3.中间中心性/中介中心性(Betweenness Centrality)：</br>
$$c_u=\sum\limits_{s{\neq}u{\neq}t}\frac{\#(\rm{shortest}\ \rm{paths}\ \rm{between}\ s\ \rm{and}\ t\ \rm{that}\ \rm{contain}\ v)}{\#(\rm{shortest}\ \rm{paths}\ \rm{between}\ s\ \rm{and}\ t)}$$
4.紧密中心性(Closeness Centrality)：
$$c_u=\frac{1}{\sum\limits_{v{\neq}u}\rm{shortest}\ \rm{path}\ \rm{length}\ \rm{between}\ v\ \rm{and}\ u}$$
5.集聚系数/群聚系数(Clustering coefficient):
$$e_u=\frac{\#(\rm{edges}\ \rm{among}\ \rm{neighboring}\ \rm{nodes})}{k_v \choose 2}\in[0,1]$$
集聚系数反映了节点 $u$ 的相邻节点之间联系的紧密程度。</br>
4.Graphlets：</br>
由 $n$ 个节点组成的子图称为 $n$-node graphlets：</br>
$\quad$</br>
![](./img/graphlets.png ':size=60%')</br>
$\quad$</br>
Graphlet Degree Vector(GDV)：由节点 $u$ 及其它节点共同构成的子图个数。


### 2.2.2 连接层面特征工程
