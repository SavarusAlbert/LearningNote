# NetworkX代码实战
## 3.1 安装配置相关环境
- 相关包安装
```python
!pip install numpy pandas matplotlib tqdm networkx -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 导入工具包
```python
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
```
- 设置matplotlib中文字体(windows系统)
```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号
# 测试效果
plt.plot([1,2,3], [100,500,300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()
```

## 3.2 图的基础操作
- 画图举例
```python
import networkx as nx
import matplotlib.pyplot as plt
G = nx.les_miserables_graph()           # 雨果-悲惨世界 人物关系图
plt.figure(figsize = (12, 10))
pos = nx.spring_layout(G, seed = 10)    # 布局方式
nx.draw(G, pos, with_labels = True)     # pos参数可选，with_labels参数展示标签
plt.show()
```
- NetworkX 提供的画图函数
```python
draw（G，[pos,ax,hold]）
draw_networkx(G，[pos,with_labels])
draw_networkx_nodes(G,pos,[nodelist])   # 绘制G的节点图
draw_networkx_edges(G,pos[edgelist])    # 绘制G的边图
```
- nx.draw基础参数
```python
nx.draw(G,node_size = 30, with_label = False)
```
node_size: 指定节点的尺寸大小</br>
with_labels: 节点是否带标签(默认为True)</br>
pos: 图像的布局</br>
- 常用布局方式
```python
pos=nx.circular_layout(G)               # 生成圆形节点布局
pos=nx.random_layout(G)                 # 生成随机节点布局
pos=nx.shell_layout(G)                  # 生成同心圆节点布局
pos=nx.spring_layout(G)                 # 利用Fruchterman-Reingold force-directed算法生成节点布局
pos=nx.spectral_layout(G)               # 利用图拉普拉斯特征向量生成节点布局
pos=nx.kamada_kawai_layout(G)           # 使用Kamada-Kawai路径长度代价函数生成布局
```
- 基础操作
```python
G = nx.DiGraph()                                    # 无多重边有向图
G.add_node(2)                                       # 添加一个节点
G.add_nodes_from([3, 4, 5, 6, 8, 9, 10, 11, 12])    # 添加多个节点
G.add_cycle([1, 2, 3, 4])                           # 添加环
G.add_edge(1, 3)                                    # 添加一条边
G.add_edges_from([(3, 5), (3, 6), (6, 7)])          # 添加多条边
G.remove_node(8)                                    # 删除一个节点
G.remove_nodes_from([9, 10, 11, 12])                # 删除多个节点
print(G.nodes())                                    # 输出所有的节点
print(G.edges())                                    # 输出所有的边
G.number_of_edges()                                 # 边的条数
G.degree                                            # 返回节点的度
G.in_degree                                         # 返回节点的入度
G.out_degree                                        # 返回节点的出度
len(G)                                              # 查看全图节点数
G.size()                                            # 查看全图连接数
G.nodes                                             # 查看所有的节点
G.edges                                             # 查看所有的连接
G.is_directed()                                     # 是否为有向图
G.graph['Name'] = 'HelloWorld'                      # 给整张图添加特征属性
G.neighbors(node_id)                                # 指定节点的邻点集
```

## 3.3 经典图结构
- 基础图形
```python
m = 7                                   # 节点个数
G = nx.complete_graph(m)                # 全连接无向图
G = nx.complete_graph(m, nx.DiGraph())  # 全连接有向图
G = nx.cycle_graph(m)                   # 环状图
G = nx.ladder_graph(m)                  # 梯状图
G = nx.path_graph(m)                    # 线性串珠图
G = nx.star_graph(m)                    # 星状图
G = nx.wheel_graph(m)                   # 轮辐图
G = nx.binomial_tree(m)                 # 二项树
```
- 多维栅格图
```python
G = nx.grid_2d_graph(3, 5)              # 二维矩形网格图
G = nx.grid_graph(dim = (2, 3, 4))      # 多维矩形网格图
G = nx.triangular_lattice_graph(2, 5)   # 二维三角形网格图
G = nx.hexagonal_lattice_graph(2, 3)    # 二维六边形蜂窝图
G = nx.hypercube_graph(4)               # n维超立方体图
```
- NetworkX内置图
```python
G = nx.diamond_graph()
G = nx.bull_graph()
G = nx.frucht_graph()
G = nx.house_graph()
G = nx.house_x_graph()
G = nx.petersen_graph()
G = nx.krackhardt_kite_graph()
```
- 随机生成
```python
# 随机图：n为节点个数，p为创建边的概率(float)
G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
# 无标度有向图
G = nx.scale_free_graph(n)
```
- 数据集生成图
```python
G = nx.karate_club_graph()              # 空手道俱乐部数据集
G.nodes[k]["club"]                      # 查看编号为k的节点的club属性
G = nx.les_miserables_graph()           # 雨果-悲惨世界 人物关系图
G = nx.florentine_families_graph()      # Florentine families graph
```
- 其他
```python
G = nx.caveman_graph(4, 3)              # 社群聚类图
tree = nx.random_tree(n=10, seed=0)
print(nx.forest_str(tree, sources=[0])) # 随机生成一个树
```

## 3.4 由连接表和邻接表创建图
- 由三元组连接表构建图
```python
url = 'triples.csv'                                 # 三国演义人物关系知识图谱
df = pd.read_csv(url)
# 提取head列和tail列作为连接，添加进空的图中
G = nx.DiGraph()
edges = [edge for edge in zip(df['head'], df['tail'])]
G.add_edges_from(edges)
```
- 导出导入邻接表
```python
# 将邻接表导出为本地文件 grid.edgelist
nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
# 从本地文件 grid.edgelist 读取邻接表
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")
```

## 3.5 创建节点和连接
- 节点可以为任意的可哈希的对象，如字符串、图像、XML对象、另一个图、自定义的节点对象
```python
# 创建空图
G = nx.Graph()
# 添加单个字符串节点
G.add_node('刘备')
# 添加多个节点
G.add_nodes_from(['诸葛亮', '曹操'])
G.add_nodes_from(range(100, 105))
# 用元组的方式添加带属性特征的节点
G.add_nodes_from([
    ('关羽',{'武器': '青龙偃月刀','武力值':90,'智力值':80}),
    ('张飞',{'武器': '丈八蛇矛','武力值':85,'智力值':75}),
    ('吕布',{'武器':'方天画戟','武力值':100,'智力值':70})
])
H = nx.path_graph(10)
G.add_nodes_from(H)             # 将另一个图的节点添加进来
G.add_node(H)                   # 将另一个图本身作为一个节点添加到G中
```
- 创建节点，并添加特征属性
```python
# 创建单个节点
G.add_node(0, feature=5, label=0, notebook=2)
# 创建多个节点
G.add_nodes_from([
  (1, {'feature': 1, 'label': 1, 'notebook':3}),
  (2, {'feature': 2, 'label': 2, 'notebook':4})
])
# 通过设置data=True查看节点属性
G.nodes(data=True)
```
- 创建连接，并添加特征属性
```python
# 创建单个连接
G.add_edge(0, 1, weight=0.5, like=3)
# 创建多个连接
G.add_edges_from([
  (1, 2, {'weight': 0.3, 'like':5}),
  (2, 0, {'weight': 0.1, 'like':8})
])
# 通过设置data=True查看连接属性
G.edges(data=True)
```

## 3.6 nx.draw可视化函数
- nx.draw常用参数
```python
nx.draw(
    G,
    pos,
    node_color='#A0CBE2',                           # 节点颜色
    edgecolors='red',                               # 节点外边缘的颜色
    edge_color="blue",                              # edge的颜色
    # edge_cmap=plt.cm.coolwarm,                    # 配色方案
    node_size=800,
    with_labels=False,
    width=3,                                        # edge线宽
)
```
- 其他参数
```python
# 设置每个节点可视化时的坐标
pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}
# 设置其它可视化样式
options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black", 
    "linewidths": 5,                                # 节点线宽
    "width": 5,
}
nx.draw_networkx(G, pos, **options)
ax = plt.gca()
ax.margins(0.20)                                    # 在图的边缘留白，防止节点被截断
plt.axis("off")                                     # 去掉坐标轴
plt.show()
```
- 分别绘制节点和连接
```python
nx.draw_networkx_nodes(G, pos)                      # 绘制节点
nx.draw_networkx_edges(G, pos)                      # 绘制连接
```
- 可视化函数模板
```python
def draw(G, pos, measures, measure_name):  
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)

    # plt.figure(figsize=(10,8))
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()
```

## 3.7 PageRank节点重要度
- pagerank函数计算有向图的节点重要度，无向图会自动转为双向图再进行计算</br>
返回一个字典，包含节点和节点重要度
```python
G = nx.star_graph(7)
pagerank = nx.pagerank(                             # 迭代的计算PR值
  G,                                                # 有向图，无向图会转为双向图
  alpha=0.85,                                       # 浮点型，阻尼参数，默认0.85
  personalization=None,                             # 字典形式，自定义节点的PR值分配，默认为均匀分配
  max_iter=100,                                     # 最大迭代次数
  tol=1e-06,                                        # 迭代阈值，停止迭代的差值
  nstart=None,                                      # 自定义PR值初值
  weight='weight',                                  # 权重
  dangling=None,
  )
```
### 幂迭代算法
- 1.对于一个有向图，每个节点的PR值等于其邻点的PR值加权求和：
$${\rm{PR}}_u=\sum\limits_{v{\in}N(u)}\lambda_{vu}{\rm{PR}}_v$$
- 2.假设当用户停留在某节点时，跳转到其他节点的概率相同，那么平均分配给每个outlink相同的权重，用权重矩阵 $M$ (Stochastic adjacency matrix)来表示所有节点的权重图。
- 3.为每个节点设定初值构建pagerank向量 $r=(r_1,\cdots)$，$\sum\limits_{i}r_i=1$，左乘权重矩阵则会得到一次迭代的PR值，多次迭代收敛到最终结果($t$为迭代次数)：
$$r_{\rm{ter}} = M^{t}r$$
### 随机游走算法
- 假设从一个节点遍历图，引入阻尼系数，以$\alpha$的概率沿有向边游走，以$1-\alpha$的概率随机游走到任意节点，得到所有节点随机游走的概率分布。
- 最终如果收敛，那么收敛的概率分布结果就是pagerank值，算法数学表达式与幂迭代相同，同样是求权重矩阵 $M$ 的特征向量。
### 马尔科夫链算法
- 每个节点是一个状态，节点之间的连接对应状态转移，应用求解马尔科夫链的方法也可以同样求解。
### Pagerank节点重要度算法
- 在实际求解网页pagerank时，随机游走算法需要模拟大量的游走，马尔科夫链方法同样需要求解大量的转移矩阵，而采样幂迭代方法则只需要计算机最擅长的矩阵乘法。
- 幂迭代终止点问题：实际应用中，有一些网页不指向任何网页，会导致迭代收敛到0
- 幂迭代陷阱问题：如果一个网页不存在指向其他网页的链接，但存在指向自己的链接，也会导致迭代问题
- 幂迭代算法改进：引入基尼系数$\alpha$，每个节点有$1-\alpha$的概率跳转到其他任意节点：
$$r^\prime={\alpha}Mr+(1-\alpha)\left[\frac{1}{N}\right]_{N{\times}N}$$
$N$ 为节点总数

## 3.8 特征工程相关API
- 基础API
```python
nx.connected_components(G)                          # 连通子域
nx.radius(G)                                        # 半径
nx.diameter(G)                                      # 直径
nx.eccentricity(G)                                  # 偏心度：每个节点到图中其它节点的最远距离
nx.center(G)                                        # 中心节点，偏心度与半径相等的节点
nx.periphery(G)                                     # 外围节点，偏心度与直径相等的节点
nx.single_source_shortest_path_length(G, node)      # node节点所有通路中最短路径长度
nx.triangles(G)                                     # 经过节点的三角形个数，返回一个字典
```
- 图密度(Graph density)(n为节点个数，m为连接个数)：</br>
无向图
$$density=\frac{2m}{n(n-1)}$$
有向图
$$density=\frac{m}{n(n-1)}$$
无连接图的density为0，全连接图的density为1，Multigraph（多重连接图）和带self loop图的density可能大于1
```python
nx.density(G)                                       # 图密度
```
- 特征工程常用API
```python
nx.degree_centrality(G)                             # 无向图 Degree Centrality
nx.in_degree_centrality(G)                          # 有向图入度 Indegree Centrality
nx.out_degree_centrality(G)                         # 有向图出度 Outdegree Centrality
nx.eigenvector_centrality(G)                        # 无向图 Eigenvector Centrality
nx.eigenvector_centrality_numpy(G)                  # 有向图 Eigenvector Centrality
nx.betweenness_centrality(G)                        # Betweenness Centrality 必经之地
nx.closeness_centrality(G)                          # Closeness Centrality 去哪儿都近
nx.pagerank(G, alpha=0.85)                          # PageRank
nx.katz_centrality(G, alpha=0.1, beta=1.0)          # Katz Centrality
nx.clustering(G)                                    # Clustering Coefficient 社群系数
nx.bridges(G)                                       # 如果某个连接断掉，会使连通域个数增加，则该连接是bridge
nx.common_neighbors(G, m, n)                        # m和n节点的共同邻点集
nx.jaccard_coefficient(G, ebunch=None)              # 计算ebunch中所有节点对的jaccard系数(交并比)
nx.adamic_adar_index(G, ebunch=None)                # 计算ebunch中所有节点对的Adamic-Adar指数(共同邻点的连接数倒数)
```
- Katz Index(节点u到节点v，路径长度为k的路径个数)：
$$S=\sum\limits_{i=1}\limits^{\infty}\beta^i(\mathcal{A}^i)=(I-\beta\mathcal{A})^{-1}-I$$
```python
import networkx as nx
import numpy as np
from numpy.linalg import inv
G = nx.karate_club_graph()
# 计算主特征向量
L = nx.normalized_laplacian_matrix(G)
e = np.linalg.eigvals(L.A)
# 折减系数(最大特征值 max(e))
beta = 1/max(e)
# 创建单位矩阵
I = np.identity(len(G.nodes))
# 计算 Katz Index
S = inv(I - nx.to_numpy_array(G)*beta) - I
```
- 计算全图Graphlet个数
```python
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
import itertools
G = nx.karate_club_graph()
# 指定Graphlet为三个节点的完全图
target = nx.complete_graph(3)
# 计算Graphlet个数
num = 0
for sub_nodes in itertools.combinations(G.nodes(), len(target.nodes())):  # 遍历全图中，符合graphlet节点个数的所有节点组合
    subg = G.subgraph(sub_nodes)                                          # 从全图中抽取出子图
    if nx.is_connected(subg) and nx.is_isomorphic(subg, target):          # 如果子图是完整连通域，并且符合graphlet特征，输出原图节点编号
        num += 1
        print(subg.edges())
```
- 拉普拉斯矩阵特征值分解：</br>
1.拉普拉斯矩阵 $L$ (Laplacian Matrix)：
$$L=D-A$$
$D$ 为节点degree对角矩阵，$A$ 为邻接矩阵</br>
2.归一化拉普拉斯矩阵(Normalized Laplacian Matrix)：
$$L_n = D^{-\frac{1}{2}}LD^{-\frac{1}{2}}$$
```python
L = nx.laplacian_matrix(G)                          # 拉普拉斯矩阵
A = nx.adjacency_matrix(G)                          # 邻接矩阵
D = L + A                                           # 节点degree对角矩阵
L_n = nx.normalized_laplacian_matrix(G)             # 归一化拉普拉斯矩阵
# 特征值分解
import numpy.linalg                                 # 线性代数包
e = np.linalg.eigvals(L_n.A)                        # 特征值
# 特征值分布直方图模板
plt.figure(figsize=(12,8))
plt.hist(e, bins=100)
plt.xlim(0, 2)                                      # eigenvalues between 0 and 2
plt.title('Eigenvalue Histogram', fontsize=20)
plt.ylabel('Frequency', fontsize=25)
plt.xlabel('Eigenvalue', fontsize=25)
plt.tick_params(labelsize=20)                       # 设置坐标文字大小
plt.show()
```

## 3.9 本章总结
本章主要学习NetworkX工具包的使用，包括
- 运用NetworkX和matplotlib进行画图
- nx.draw可视化方案
- PageRank节点重要度的算法和代码演示
- 特征工程API的学习