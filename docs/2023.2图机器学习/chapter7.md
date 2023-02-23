# Chapter07 半监督节点分类
## 7.1 基本概念
### 7.1.1 半监督学习
- 半监督学习：不依赖于外界信息，利用未标记的样本自动的提升学习性能，就是半监督学习。
- 半监督学习分为纯半监督学习(pure semi-supervised learning)和直推式半监督学习(transductive semi-supervised learning)。前者假定训练数据中的未标记样本并非待预测的数据，后者则假定学习过程中所考虑的未标记样本恰是待预测样本，学习的目的就是在这些未标记样本上获得最优泛化性能。
### 7.1.2 直推式学习与归纳式学习
- 直推式学习(transductive)：由当前学习的知识直接推广到给定的数据上(未知标签的部分参与训练，没有新的节点加入)。
- 归纳式学习(inductive)：从已有数据中归纳出模式来，应用于新的数据和任务(由已知部分训练预测新加入的节点)。
### 7.1.3 半监督节点分类方法
- Label Propagation(Relational Classification)
- Iterative Classification
- Correct & Smooth
- Belief Propagation
- Masked Label Prediction

## 7.2 半监督节点分类算法
- 大自然对图的基本假设
    - Homophily：具有相似属性特征的节点更可能相连且具有相同类别。(不是一家人不进一家门)
    - Influence：节点的社群连接会影响该节点。(物以类聚人以群分)

- 如何利用网络中的相关信息进行半监督的分类：
    - KNN最近邻分类
    - 利用节点自身的属性特征以及邻域节点的类别和属性特征

### 7.2.1 标签传播(Label Propagation)
- 基本算法
    - 1.初始化：已知标签按照标签标注(0、1标注)，未知标签初始化固定初值(0.5)
    - 2.按照节点顺序分别对所有未知标签计算概率值(周围节点标签概率加权平均)：
$$P(Y_v=c)=\frac{1}{\sum_{(v,u){\in}E}A_{v,u}}\sum\limits_{(u,v){\in}E}A_{v,u}P(Y_u=c)$$
    - 3.迭代步骤2至收敛，得到最终的标签概率(不一定收敛，可以预置迭代次数)
    - 4.将大于0.5的置位1，小于0.5的置位0
- 算法缺点
    - 不一定收敛
    - 没有用到节点属性特征

- 社群检测算法代码
```python
from networkx.algorithms import community
communities = community.label_propagation_communities(G)        # G 是任意的图
```

### 7.2.2 Iterative Classification
- 属性特征和连接特征
    - $f_v$ ：节点 $v$ 的属性特征
    - $z_v$ ：节点 $v$ 在邻域 $N_v$ 中的连接信息，通过人为构造：
        - 邻域内不同标签节点的直方图信息
        - 邻域内最多的标签
        - 邻域内类别数
        - 其他构造
- 基本算法
    - 1.使用已标注数据训练两个分类器：</br>
        base classifier：$\phi_1(f_v)$ </br>
        relational classifier：$\phi_2(f_v,z_v)$
    - 2.用 $\phi_1$ 预测未知标签节点的标签 $Y_v$ ，迭代以下步骤直至收敛(不一定收敛，预置迭代次数)：
        - 用 $Y_v$ 计算 $z_v$
        - 用 $\phi_2(f_v,z_v)$ 更新所有节点的 $Y_v$

### 7.2.3 Correct & Smooth
- 基本算法
    - 1.在有标签的节点上训练一个基础的分类器1。
    - 2.用分类器1预测所有节点(包括有标签节点)的分类，得到每个节点的soft label：
        $${\rm{soft\ label}}=({\rm{Pr}}_1,{\rm{Pr}}_0)$$
        其中 ${\rm{Pr}}_1$ 是类别为1的概率，${\rm{Pr}}_0$ 是类别为0的概率。
    - 3.后处理Correct & Smooth
        - Correct step(对不确定性较大的节点进行处理)：
            - 计算所有有标签节点的error：
            $${\rm{error}}=({\rm{label}}_1-{\rm{Pr}}_1,{\rm{label}}_0-{\rm{Pr}}_0)$$
            其中 ${\rm{label}}=({\rm{label}}_1,{\rm{label}}_0)$ 是有标签节点的标签信息，为$(1,0)$ 或 $(0,1)$。
            - 未知标签节点error初值置为 $(0,0)$ ，将所有节点的error作为行向量组成一个矩阵 $E^{(0)}$。
            - 迭代计算第 $t+1$ 步的error矩阵(可以证明不会发散)，得到Diffused training errors：
            $$E^{(t+1)}=(1-\alpha){\cdot}E^{(t)}+\alpha{\cdot}\tilde{A}E^{(t)}$$
            其中 $\alpha$ 是超参数，$\tilde{A}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ 是度归一化邻接矩阵，$A$ 是邻接矩阵，$D$ 是度矩阵。
            - 输出最终的error：
            $${\rm{Output\ error}}={\rm{Soft\ labels}}-s\cdot{\rm{Diffused\ training\ errors}}$$
            其中 $s$ 是超参数，最后进行一下归一化。
        - Smooth step(类似标签分类)：
            - 构造矩阵 $Z^{(0)}$ ，用有标签节点的 ${\rm{label}}$ 和无标签节点的 ${\rm{Output\ error}}$ 作为行向量组成的矩阵。
            - 迭代计算第 $t+1$ 步的 $Z$ 矩阵，收敛后得到每个点的预测信息：
            $$Z^{(t+1)}=(1-\alpha){\cdot}Z^{(t)}+\alpha{\cdot}\tilde{A}Z^{(t)}$$

### 7.2.4 Loopy Belief Propagation
类似于消息传递，下一时刻的状态仅取决于上一时刻，当所有节点达到共识时，可得最终结果，是一种动态规划算法。
- 定义
    - Label-label potential matrix $\psi$ ，$\psi(Y_i,Y_j)$ 定义为：节点 $i$ 为 $Y_i$ 类别时节点 $j$ 为 $Y_j$ 类别的概率。
    - Prior belief $\phi$ ，$\phi(Y_i)$ 定义为：节点 $i$ 为 $Y_i$ 类别的概率。
    - Messages $m_{i{\rightarrow}j}(Y_j)$ ：节点 $i$ 认为节点 $j$ 是 $Y_j$ 类别的概率。
- 基本算法
    - 初始化所有Messages为1。
    - 迭代计算：
    $$m_{i{\rightarrow}j}(Y_j)=\sum\limits_{Y_i{\in}\mathcal{L}}\psi(Y_i,Y_j)\phi(Y_i)\prod\limits_{k{\in}N_i{\setminus}j}m_{k{\rightarrow}i}(Y_i),\quad{\forall}Y_j\in\mathcal{L}$$
    其中 $\mathcal{L}$ 是所有类别组成的集合。
    - 收敛后得到节点 $i$ 为 $Y_i$ 类别的概率：
    $$b_i(Y_i)=\phi_i(Y_i)\prod\limits_{j{\in}N_i}m_{j{\rightarrow}i}(Y_i),\quad{\forall}Y_i\in\mathcal{L}$$
- 如果图中有环，则会出现不收敛情况。

### 7.2.5 Masked Label Prediction
灵感来源于Bert
- 基本思路
    - mask掉部分的标签，然后预测mask的部分，来训练出一个分类器。
    - 用训练好的分类器分类未知标签的节点。
## 7.3 本章总结
本次主要学习了半监督节点分类的相关算法，主要包括：
- 半监督学习的基本概念
- 半监督节点分类的5种基本算法