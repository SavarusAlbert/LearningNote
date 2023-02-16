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

## 3.7 