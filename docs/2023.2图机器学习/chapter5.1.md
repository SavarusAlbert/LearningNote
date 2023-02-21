# DeepWalk论文精读
## 5.1 DeepWalk前瞻
- DeepWalk首个将深度学习和自然语言处理的思想用于图机器学习，并且表明这两个领域可以互相影响共同进步。
- 在稀疏标注的节点分类场景下，嵌入性能卓越，并且具有很强的可扩展性。

### 5.1.1 自然语言处理Word2Vec
- Word2Vec是用来生成词向量的工具
- 通过词嵌入将单词转换为向量，再通过masked任务来进行预测
- 如果将图对应于文章，随机游走序列对应于句子，可以类比词嵌入来进行图嵌入操作。

## 5.2 DeepWalk优劣
### 5.2.1 优点
- 在线学习，可以对新加入的节点数据进行增量训练
- 可以将随机游走当做句子，使用自然语言处理领域的语言模型
- 效果很好，实现容易
- 可并行计算，效果仍能保持
### 5.2.2 缺点
- 完全的均匀随机游走，没有偏向(不像Node2Vec有BFS和DFS)
- 需要采样大量的随机游走序列进行训练
- 由于基于随机游走，因此很难分析全图信息，距离较远的节点很难相互关联
- 没有用到节点的属性信息
- 没有真正用到神经网络和深度学习

## 5.3 DeepWalk文章部分
**Online Learning of Social Representations**
- 用于图节点嵌入的在线机器学习算法
### 5.3.1 摘要
- DeepWalk是一个学习"网络中节点"的隐式表征的表示学习方法。
- 这个隐式的表征可以将网络中节点的"连接"编码映射到一个低维稠密的向量空间，方便进行后续的统计学习。
- DeepWalk将自然语言处理和无监督学习(或深度学习)领域的模型进行了推广。
- 通过"有截断的随机游走序列"去学习图的局部信息，将随机游走序列视作句子来进行机器学习。
- DeepWalk算法是一个可扩展的线上机器学习算法，可以进行增量学习，并且是可并行的。

### 5.3.2 问题描述
- 图$G = (V, E)$ ，$V$ 是节点集， $E$ 是边集， $E{\subseteq}(V{\times}V)$ ，$G_L=(V,E,X,Y)$ 是一个社交网络。
- 其中 $X\in\mathbb{R}^{|V|{\times}S}$ 是 $S$ 维的特征向量组成的特征空间，$Y
\in\mathbb{R}^{|V|\times|\mathcal{Y}|}$ 是由 $|\mathcal{Y}|$ 维的标签组成。
- 在传统的机器学习算法中，我们的目的是拟合一个映射，输入特征 $X$ 输出标签 $Y$ ，而对于图机器学习算法来说，我们可以充分利用节点之间的关联信息(连接)。
- 不同于传统算法，将问题描述为一个马尔科夫链的状态转移过程去处理，DeepWalk采用无监督的算法(随机游走序列)去根据连接信息预测结果。
- 将连接信息和标签信息分开，防止了误差累积问题(一步错步步错)。
- 目标是将稀疏的特征空间 $X$ 转换为一个稠密矩阵 $X_E\in\mathbb{R}^{|V|{\times}d}$ ，再进行后续的机器学习。

### 5.3.3 嵌入表示学习
- DeepWalk希望学习到的嵌入具有以下特性：</br>
1.灵活可变，弹性扩容</br>
2.反应社群聚类信息</br>
3.维度较低，便于训练</br>
4.嵌入表示，连续稠密
#### 随机游走算法：
- 从节点 $v_i$ 出发进行随机游走 $\mathcal{W}_{v_i}$ ，随机游走序列 $\mathcal{W}_{v_i}^1,\mathcal{W}_{v_i}^2,\cdots,\mathcal{W}_{v_i}^k$ ，第 $k+1$ 个点 $\mathcal{W}_{v_i}^{k+1}$ 是在第 $k$ 个点 $v_k$ 的邻域内选取。
- 这样的随机游走可以提取网络中的局部信息
- 可以并行的同时进行采样和数据处理
- 可以对子图构建随机游走，由子图再迭代到全图
#### 幂律分布
- 分布密度函数是幂函数的分布，变量 $x$ 服从参数为 $\alpha$ 的幂律分布，其概率密度函数为：
$$f(x)=cx^{-\alpha-1},x\rightarrow\infty$$
- 俗称二八定律，例如：20%的网站占据了80%的点击量，文章中有一些单词出现频率非常高。
- 自然语言处理和图机器学习任务都符合幂律分布，因此可以互相借鉴。

#### 语言模型
- 一般情况下，语言模型通过句子中的前 $n-1$ 个词来预测第 $n$ 个词，最大化下面这个概率：
$${\rm{Pr}}(w_n|w_0,w_1,\cdots,w_{n-1})$$
- DeepWalk也通过随机游走的前 $i-1$ 个节点来预测第 $i$ 个节点：
$${\rm{Pr}}(v_i|(v_1,v_2,\cdots,v_{i-1}))$$
- 由于节点并不是计算机能够理解的语言，所以需要通过图嵌入将其映射到特征空间中：
$$\Phi:v{\in}V\mapsto\mathbb{R}^{|V|{\times}d}$$
- 问题转化为求下面这个似然函数：
$${\rm{Pr}}(v_i|(\Phi(v_1),\Phi(v_2),\cdots,\Phi(v_{i-1})))$$
然而随着随机游走长度的增加，这个概率值会越来越小，会趋近于0导致不能进行预测。
- 类比语言模型中的方法：用一个词的上下文来预测mask掉的中间词，问题转化为求解下列的函数，用前后 $w$ 个随机游走来预测中间节点的概率，避免了趋于0的情况：
$$\mathop{\rm{minimize}}\limits_{\Phi}\qquad-{\rm{log}}{\rm{Pr}}(\{v_{i-w},\cdots,v_{i+w}\}{\setminus}v_i|\Phi(v_i))$$

### 5.3.4 算法伪代码
#### 算法1 DEEPWALK$(G,w,d,\gamma,t)$

> **Input：** graph $G(V,E)$  
&emsp;&emsp;window size $w$  
&emsp;&emsp;embedding size $d$  
&emsp;&emsp;walks per vertex $\gamma$&emsp;&emsp;(每个节点作为起始节点生成随机游走的次数)  
&emsp;&emsp;walk length $t$ &emsp;&emsp;(随机游走的最大长度)  
**Output：** matrix of vertex representations $\Phi\in\mathbb{R}^{|V|{\times}d}$  
1: Initialization: Sample $\Phi$ from $\mathcal{U}^{|V|{\times}d}$&emsp;&emsp;(随机初始化 $\Phi$ )  
2: Build a binary Tree $T$ from $V$  
3: **for** $i=0$ to $\gamma$ **do**  
4: &emsp;&emsp;$\mathcal{O}$=Shuffle$(V)$&emsp;&emsp;(随机打乱节点顺序)  
5: &emsp;&emsp;**for each** $v_i\in\mathcal{O}$ **do**&emsp;&emsp;(遍历图中每个节点)  
6: &emsp;&emsp;&emsp;&emsp;$\mathcal{W}_{v_i}=RandomWalk(G,v_i,t)$&emsp;&emsp;(生成一个随机游走序列)  
7: &emsp;&emsp;&emsp;&emsp;SkipGram$(\Phi,\mathcal{W}_{v_i},w)$&emsp;&emsp;(由中心节点预测周围节点)  
8: &emsp;&emsp;**end for**  
9: **end for**

#### 算法2 SkipGram $(\Phi,\mathcal{W}_{v_i},w)$
> 1: **for each** $v_j\in\mathcal{W}_{v_i}$ **do**&emsp;&emsp;(遍历当前随机游走序列里的每个节点)  
2: &emsp;&emsp;**for each** $u_k\in\mathcal{W}_{v_i}[j-w:j+w]$ **do**&emsp;&emsp;(遍历该节点周围窗口中的每个节点)  
3: &emsp;&emsp;&emsp;&emsp;$J(\Phi)=-{\rm{log}}{\rm{Pr}}(u_k|\Phi(v_j))$&emsp;&emsp;(计算损失函数)  
4: &emsp;&emsp;&emsp;&emsp;$\Phi=\Phi-\alpha\times\frac{{\partial}J}{\partial\Phi}$&emsp;&emsp;(梯度下降反向传播)  
5: &emsp;&emsp;**end for**  
6: **end for**

#### 分层Softmax(Hierarchical Softmax)
- 在反向传播做预测的时候，由于类别过于多，做softmax极大的消耗算力，因此考虑用分层Softmax的思想将复杂度由 $O(|V|)$ 降低到 $O({\rm{log}}|V|)$ 。
- 霍夫曼树编码：8分类变为3个2分类，降低复杂度。

#### 多线程异步并行
- 加速训练仍能保持性能

### 5.3.5 算法变种
- 在未知全图时，直接在子图中采样随机游走序列进行训练。
- 不随机的自然游走：用户可能有一定倾向性，在游走过程中可以加入一定的偏向。

## 5.4 代码实战
### 5.4.1 维基百科网页引用关联数据读取
```python
# 导入工具包
import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
# 数据可视化
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei']      # 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus']=False        # 用来正常显示负号
# 读取tsv文件
df = pd.read_csv("seealsology-data.tsv", sep = "\t")
```
### 5.4.2 生成随机游走序列
- 输入起始节点和路径长度，生成随机游走节点序列
```python
def get_randomwalk(node, path_length):
    random_walk = [node]
    for i in range(path_length-1):
        # 汇总邻接节点
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break
        # 从邻接节点中随机选择下一个节点
        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
    return random_walk
```
- 获取随机游走序列
```python
gamma = 10                                  # 每个节点作为起始点生成随机游走序列个数
walk_length = 5                             # 随机游走序列最大长度

random_walks = []
for n in tqdm(all_nodes):                   # 遍历每个节点
    for i in range(gamma):                  # 每个节点作为起始点生成gamma个随机游走序列
        random_walks.append(get_randomwalk(n, walk_length))
```
### 5.4.3 训练Word2Vec模型
- 训练DeepWalk就等同于训练Word2Vec
```python
from gensim.models import Word2Vec          # 自然语言处理
model = Word2Vec(vector_size=256,           # Embedding维数
                 window=4,                  # 窗口宽度
                 sg=1,                      # Skip-Gram
                 hs=0,                      # 不加分层softmax
                 negative=10,               # 负采样
                 alpha=0.03,                # 初始学习率
                 min_alpha=0.0007,          # 最小学习率
                 seed=14                    # 随机数种子
                )
# 用随机游走序列构建词汇表
model.build_vocab(random_walks, progress_per=2)
# 模型训练（耗时1分钟左右）
model.train(random_walks, total_examples=model.corpus_count, epochs=50, report_delay=1)
```
- 查看结果
```python
# 查看某个节点的Embedding
model.wv.get_vector('random forest').shape
model.wv.get_vector('random forest')
# 找相似词语
model.wv.similar_by_word('decision tree')
```

### 5.4.4 降维可视化
- 将Embedding用PCA降维到2维
```python
X = model.wv.vectors
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embed_2d = pca.fit_transform(X)
```
- 将Embedding用TSNE降维到2维
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, n_iter=1000)
embed_2d = tsne.fit_transform(X)
```
- 作图
```python
plt.figure(figsize=(14,14))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1])
plt.show()
```

## 5.5 本章总结
本章主要对DeepWalk文章进行精读，主要包括：
- Word2Vec思想的借鉴
- 随机游走序列的原理
- 图嵌入的理论支撑
- DeepWalk算法伪代码
- 维基百科网页引用代码实战