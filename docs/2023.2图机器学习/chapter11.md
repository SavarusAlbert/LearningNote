# Chapter11 GraphSAGE论文精读
大规模图上的归纳表示学习。
## 11.1 GraphSAGE前瞻
- GraphSAGE从两个方面对传统的GCN做了改进：
    - 在训练时，将GCN的全图采样优化到在节点的邻域内抽样，从而可以进行小批量训练，这使得大规模图数据的分布式训练成为可能。
    - 通过一个新定义的损失函数，使网络可以预测没有见过的节点，增加了模型的泛化能力，这使得GraphSAGE可以用来做归纳学习(Inductive Learning)。
## 11.2 GraphSAGE优劣
### 11.2.1优点
- 通过邻域采样的方式解决了GCN内存爆炸的问题，适用于大规模图训练。
- 将直推式学习(transductive)转化为归纳式(inductive)学习，避免节点的特征多次重复训练。
- GraphSAGE将模型目标定于学习一个聚合器而不是为每个节点学习一个表示，从而提升了模型的灵活性和泛化能力。
### 11.2.2缺点
- 只能用于无向图。
- 随机采样导致模型训练阶段同一节点embedding特征不稳定。
- 采样数目限制会导致部分节点的重要局部信息丢失。
## 11.3 GraphSAGE论文
### 11.3.1 GraphSAGE嵌入生成(前向传播)算法
我们首先假设GraphSAGE神经网络模型参数已经训练好，来讨论嵌入生成算法。
#### 算法1 GraphSAGE embedding generation algorithm
- 假定 ${\rm{AGGREGATE}}_k,{\forall}k\in\{1,\cdots,K\}$ 是已经训练好的 $K$ 个聚合函数，通过一组权重矩阵 ${\mathbf{W}}^k,{\forall}k\in\{1,\cdots,K\}$ 来聚合节点邻域信息。

> **Input：** 图 $\mathcal{G}(\mathcal{V},\mathcal{E})$  
&emsp;&emsp;特征 $\{x_v,{\forall}v\in\mathcal{V}\}$  
&emsp;&emsp;深度 $K$  
&emsp;&emsp;权重矩阵 ${\mathbf{W}}^k,{\forall}k\in\{1,\cdots,K\}$  
&emsp;&emsp;非线性函数 $\sigma$  
&emsp;&emsp;可微聚合函数 ${\rm{AGGREGATE}}_k,{\forall}k\in\{1,\cdots,K\}$  
&emsp;&emsp;邻域函数 $\mathcal{N}:v{\rightarrow}2^{\mathcal{V}}$  
**Output：** 向量表示 $z_v,{\forall}v\in\mathcal{V}$  
1: Initialization: $h_v^0{\leftarrow}x_v,{\forall}v\in\mathcal{V}$ &emsp;&emsp;(第0层的输入是节点的属性特征向量)    
2: **for** $k=1{\cdots}K$ **do**&emsp;&emsp;($K$层神经网络)  
3: &emsp;&emsp;**for** $v\in\mathcal{V}$ **do**&emsp;&emsp;(遍历每个节点 $v$)    
4: &emsp;&emsp;&emsp;&emsp;$h_{\mathcal{N}(v)}^k\leftarrow{\rm{AGGREGATE}_k}(\{h_u^{k-1},{\forall}u\in\mathcal{N}(v)\})$;&emsp;&emsp;(聚合所有邻域节点前一层嵌入)  
5: &emsp;&emsp;&emsp;&emsp;$h_v^k\leftarrow\sigma\left(\mathbf{W}^k\cdot{\rm{CONCAT}}(h_v^{k-1},h_{\mathcal{N}(v)}^k)\right)$&emsp;&emsp;(自注意力操作及模型参数训练)  
6: &emsp;&emsp;**end**  
7: &emsp;&emsp;$h_v^k{\leftarrow}h_v^k/||h_v^k||_2,{\forall}v\in\mathcal{V}$&emsp;&emsp;(归一化)  
8: **end**  
9: $z_v{\leftarrow}h_v^K,{\forall}v\in\mathcal{V}$&emsp;&emsp;(最后一层嵌入向量作为结果)

#### 算法2 GraphSAGE minibatch forward propagation algorithm
- 为了进行小批量计算，我们在固定大小的邻域中做均匀采样，并且在每个迭代epoch重新采样，将算法复杂度从 $\mathcal{O}(|\mathcal{V}|)$ 降低到 $\mathcal{O}(\prod_{i=1}^KS_i)$，$S_i$ 是可设置的常数，表示采样的邻域大小。

> **Input：** 图 $\mathcal{G}(\mathcal{V},\mathcal{E})$  
&emsp;&emsp;特征 $\{x_v,{\forall}v\in\mathcal{B}\}$  
&emsp;&emsp;深度 $K$  
&emsp;&emsp;权重矩阵 ${\mathbf{W}}^k,{\forall}k\in\{1,\cdots,K\}$  
&emsp;&emsp;非线性函数 $\sigma$  
&emsp;&emsp;可微聚合函数 ${\rm{AGGREGATE}}_k,{\forall}k\in\{1,\cdots,K\}$  
&emsp;&emsp;邻域采样函数 $\mathcal{N}_k:v{\rightarrow}2^{\mathcal{V}},{\forall}k\in\{1,\cdots,K\}$  
**Output：** 向量表示 $z_v,{\forall}v\in\mathcal{B}$  
1: $\mathcal{B}^K\leftarrow\mathcal{B}$&emsp;&emsp;(初始化)  
2: **for** $k=K{\cdots}1$ **do**&emsp;&emsp;(逆遍历$K$层神经网络)  
3: &emsp;&emsp;$\mathcal{B}^{k-1}\leftarrow\mathcal{B}^k$;&emsp;&emsp;  
4: &emsp;&emsp;**for** $u\in\mathcal{B}^k$ **do**&emsp;&emsp;(遍历每个节点 $u$)  
5: &emsp;&emsp;&emsp;&emsp;$\mathcal{B}^{k-1}\leftarrow\mathcal{B}^{k-1}\cup\mathcal{N}_k(u)$;&emsp;&emsp;(并上每个节点的邻域)  
6: &emsp;&emsp;**end**  
7: **end**  
8: $h_u^0{\leftarrow}x_v,{\forall}v\in\mathcal{B}^0$;&emsp;&emsp;(初始化为属性特征向量)  
9: **for** $k=1{\cdots}K$ **do**&emsp;&emsp;($K$层神经网络)  
10: &emsp;&emsp;**for** $u\in\mathcal{B}^k$ **do**&emsp;&emsp;(遍历每个节点 $u$)  
11: &emsp;&emsp;&emsp;&emsp;$h_{\mathcal{N}(u)}^k\leftarrow{\rm{AGGREGATE}}_k(\{h_{u^\prime}^{k-1},{\forall}u^\prime\in\mathcal{N}_k(u)\})$;&emsp;&emsp;(聚合每个点和其邻点嵌入)  
12: &emsp;&emsp;&emsp;&emsp;$h_u^k\leftarrow\sigma(\mathbf{W}^k\cdot{\rm{CONCAT}}(h_u^{k-1},h_{\mathcal{N}(u)}^k))$;&emsp;&emsp;(自注意力操作及模型参数训练)  
13: &emsp;&emsp;&emsp;&emsp;$h_u^k{\leftarrow}h_u^k/||h_u^k||_2$;&emsp;&emsp;(归一化)  
14: &emsp;&emsp;**end**  
15: **end**  
16: $z_u{\leftarrow}h_u^K,{\forall}u\in\mathcal{B}$&emsp;&emsp;(最后一层嵌入向量作为结果)

- 注意到我们的采样算法是逆向采样，每一层的数据都要多于下一层，在算法12、13行中，我们之所以能够在for loop中计算 $k$ 次迭代时 $\mathcal{B}^k$ 中所有的节点表示，是因为在 $k-1$ 次迭代中已经计算过这些节点了，避免了计算minibatch中不存在的节点。

### 11.3.2 GraphSAGE参数训练
- 为了学习到能够预测未知节点的嵌入表示，GraphSAGE采用了一个graph-based loss function来输出节点嵌入 $z_u$，通过随机梯度下降来训练权重矩阵和聚合函数参数。
- graph-based loss function能够使相近的节点具有相似的表示，同时也可以使分离的节点具有非常不同的表示：
$$J_{\mathcal{G}}(z_u)=-{\rm{log}}\left(\sigma(z_u^{\top}z_v)\right)-Q\cdot\mathbb{E}_{v_n{\sim}P_n(v)}{\rm{log}}\left(\sigma(z_u^{\top}z_{v_n})\right)$$
其中节点 $v$ 和节点 $u$ 是在同一个固定长随机游走序列中，$\sigma$ 是sigmoid函数，$P_n$ 是负采样分布，$Q$ 是负样本数。
- 与其他算法的损失函数不同的是，我们输入损失函数的特征 $z_u$ 是由节点的一个局部邻域生成的，而不是每个节点训练一个唯一的嵌入。
- 在特定的下游任务中，这个损失函数也可以换成适合特定任务的损失函数(例如交叉熵损失)。

### 11.3.3 聚合函数
- 图数据中，一个节点的邻域内没有自然的顺序，聚合函数需要作用在一个无序的向量集上，所以需要满足置换不变性。我们介绍以下3种聚合函数：
#### 均值聚合
- 通过均值来聚合信息，替换算法1的4、5行：
$$h_v^k\leftarrow\sigma\left(\mathbf{W}\cdot{\rm{MEAN}}(\{h_v^{k-1}\}\cup\{h_u^{k-1},{\forall}u\in\mathcal{N}(v)\})\right)$$

#### LSTM聚合
- 与均值聚合相比，LSTM聚合具有更大的表达能力。但由于LSTM并不是对称的(不满足置换不变性)，因此我们将通过对节点邻域进行随机置换来构造一个无序集，然后作用LSTMs。

#### 池化聚合
- 池化聚合既是对称的也是可训练的。通过一个全连接神经网络来进行最大池化：
$${\rm{AGGREGATE}}_k^{\rm{pool}}={\rm{max}}\left(\{\sigma(\mathbf{W}_{\rm{pool}}h_{u_i}^k+b),{\forall}u_i\in\mathcal{N}(u)\}\right)$$
其中 ${\rm{max}}$ 是逐元素最大算子。

### 11.3.4 理论基础
- **定理1** 图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$，$U$ 是 $\mathbb{R}^d$ 中任意的紧子集，$x_v{\in}U,{\forall}v\in\mathcal{V}$ 是算法1的输入。假定存在一个常数 $C\in\mathbb{R}^+$，使得对于任意的节点对都有$||x_v-x_{v^\prime}||_2>C$。</br>
那么，对于 $\forall\epsilon>0$，都存在一个算法1的参数集 $\Theta^*$，使得 $K=4$ 次迭代后：
$$|z_v-c_v|<\epsilon,{\forall}v\in\mathcal{V}$$
其中 $z_v\in\mathbb{R}$ 是算法1最后的输出，$c_v$ 是节点聚集系数。
- 定理1指出，对于任意的图，算法1都是可解的。如果每个节点的特征均不相同，且模型维度足够高，那么它可以以任意精度求解该图的聚类系数。

## 11.4 本章总结
本章学习了GraphSAGE的基础理论，主要包括：
- GraphSAGE前向传播算法以及小批量训练算法
- 基于图的损失函数的定义
- 三种聚合函数的比较