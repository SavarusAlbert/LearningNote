# Chapter08-09 图神经网络概述
图神经网络Graph neural networks(GNNs)是基于深度学习方法解决图机器学习问题的理论。
- 图神经网络中的很多设计都来源于卷积神经网络CNN和循环神经网络RNN，它们最大的不同在于图神经网络的研究对象是在非欧空间中。
## 8.1 图神经网络分类和基本模块
-图神经网络一般可以分为以下几种：循环图神经网络(recurrent graph neural networks)，卷积图神经网络(convolutional graph neural networks)，图自动编码器(graph autoencoders)和时序图神经网络(spatial-temporal graph neural networks)。
- 图神经网络模型一般会有三种类型的模块，分别是：传播模块(Propagation Module)、采样模块(Sampling Module)和池化模块(Pooling Module)。
    - 传播模块：用于聚合节点之间的特征和拓扑信息，在模块中，卷积算子(convolution)和循环算子(recurrent operator)通常用来聚合信息，跳跃连接算子(skip connection)通常用来提取历史信息并缓解过度平滑问题。
    - 采样模块：通常用在图比较大的情况，与聚合模块结合使用。
    - 池化模块：当需要提取高维子图信息时，会用到池化操作。
![](./img/GNN1.png ':size=100%')

## 8.2 图神经网络模型部分理论基础
### 8.2.1 信息传递框架(Message Passing Framework)
信息传递框架是近年来大部分GNN工作所采用的设计，主要包含了两个部分：聚合操作(aggregation)和更新操作(updating)。
- 在一个GNN的信息传递过程中，节点 $u{\in}V$ 对应的隐藏嵌入 $h_u^{(k)}$ 是由其邻域 $N(u)$ 的信息共同得到：
$$\begin{aligned}
h_u^{(k+1)}&={\rm{UPDATE}}^{(k)}(h_u^{(k)},{\rm{AGGREGATE}}^{(k)}(\{h_v^{(k)},{\forall}v\in\mathcal{N}(u)\}))\\
&={\rm{UPDATE}}^{(k)}(h_u^{(k)},m_{\mathcal{N}(u)}^{(k)})
\end{aligned}$$
其中，${\rm{UPDATE}}$ 和 ${\rm{AGGREGATE}}$ 是任意的可微函数，$m_{\mathcal{N}(u)}$ 是从节点 $u$ 的邻域 $\mathcal{N}(u)$ 收集的 ${\rm{message}}$。
- 在 $k=0$ 时，$h_u^{(0)}=x_u,{\forall}u\in\mathcal{V}$ ，$k$ 次迭代后，我们用最后的输出来定义每个节点的嵌入：
$$z_u=h_u^{(K)},{\forall}u\in\mathcal{V}$$
- 由于聚合函数(AGGREGATE)的输入是节点的集合 $\{h_v,{\forall}v{\in}\mathcal{N}(u)\}$，对于节点的邻点顺序没有做要求，这就说明聚合函数满足**置换不变性**。

### 8.2.2 基本信息传递框架
- 常用的最基础的信息传递框架定义为：
$$h_u^{(k)}=\sigma\left(W_{\rm{self}}^{(k)}h_u^{(k-1)}+W_{\rm{neigh}}^{(k)}\sum\limits_{v{\in}\mathcal{N}(u)}h_v^{(k-1)}+b^{(k)}\right)$$
其中 $W_{\rm{self}}^{(k)},\ W_{\rm{neigh}}^{(k)}\in\mathbb{R}^{d^{(k)}{\times}d^{(k-1)}}$ 是可训练的参数，$\sigma$ 是非线性函数，$b^{(k)}$ 是偏置项。
- 上述定义等价于以下两式：
$$m_{\mathcal{N}(u)}=\sum\limits_{v{\in}\mathcal{N}(u)}h_v$$
$${\rm{UPDATE}}(h_u,m_{\mathcal{N}(u)})=\sigma(W_{\rm{self}}h_u+W_{\rm{neigh}}m_{\mathcal{N}(u)})$$

### 8.2.3 聚合操作
#### 邻域归一化
- 由于聚合运算需要邻域内所有节点的信息，并不稳定，其结果对节点高度敏感。
- 最简单的方式是求均值：
$$m_{\mathcal{N}(u)}=\frac{\sum_{v\in\mathcal{N}(u)}h_v}{|\mathcal{N}(u)|}$$
- 也可以做对称归一化：
$$m_{\mathcal{N}(u)}=\sum\limits_{v\in\mathcal{N}(u)}\frac{h_v}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}}$$
- 图卷积网络(Graph convolutional networks)采用对称归一化聚合和自循环更新方法：
$$h_u^{(k)}=\sigma\left(W^{(k)}\sum\limits_{v{\in}\mathcal{N}(u){\cup}\{u\}}\frac{h_v}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}}\right)$$
#### 池化操作
- 池化操作通过对特征进行降维，来获取高维表示信息。由于聚合函数满足置换不变性，所以选取的池化操作也要满足，最简单的可以通过一个MLP来进行池化操作：
$$m_{\mathcal{N}(u)}={\rm{MLP}}_\theta\left(\sum\limits_{v\in\mathcal{N}(u)}{\rm{MLP}}_\phi(h_v)\right)$$
通常来说池化操作通过牺牲一些信息来避免过拟合，因此不需要很深的MLP(通常一两层就可以满足要求)。
- Janossy pooling采用另一种思路，对置换后的邻点集，作用一个置换敏感函数再取平均：
$$m_{\mathcal{N}(u)}={\rm{MLP}}_\theta\left(\frac{1}{|\Pi|}\sum\limits_{\pi\in\Pi}\rho_\phi(h_{v_1},h_{v_2},\cdots,h_{v_{|\mathcal{N}(u)|}})_{\pi_i}\right)$$
其中 $\Pi$ 是置换的集合，$\rho_\phi$ 是一个置换敏感函数，实际操作时 $\rho_\phi$ 通常使用LSTM。
- 如果 $\Pi$ 为所有可能的置换，那么上式也是一个广义近似逼近(universal function
approximator)，然而对所有可能的情况进行计算是困难的，因此，实践中通常采用下述策略之一：
    - 在置换集合中随机采样出一个子集进行计算
    - 在邻点集的一个正则序列上计算，例如按照节点度降序排列并随机断开连接。
#### 邻域注意力机制
- 另一种提升聚合层的方法是利用注意力机制，用注意力权重来定义 ${\rm{message}}$：
$$m_{\mathcal{N}(u)}=\sum\limits_{v\in\mathcal{N}(u)}\alpha_{u,v}h_v$$
其中 $\alpha_{u,v}$ 表示聚合 $u$ 时 邻域 $v\in\mathcal{N}(u)$ 上的注意力权重。
- 在最初的GAT论文中，注意力权重定义为：
$$\alpha_{u,v}=\frac{{\rm{exp}}(a^\top[Wh_u{\oplus}Wh_v])}{\sum_{v^\prime\in\mathcal{N}(u)}{\rm{exp}}(a^\top[Wh_u{\oplus}Wh_{v^\prime}])}$$
其中 $a$ 是可训练的向量，$W$ 是可训练的矩阵，$\oplus$ 是concat操作。
- 其他的注意力权重定义方式：
$$\alpha_{u,v}=\frac{{\rm{exp}}(h_u^{\top}Wh_v)}{\sum_{v^\prime\in\mathcal{N}(u)}{\rm{exp}}(h_u^{\top}Wh_{v^\prime})}$$
$$\alpha_{u,v}=\frac{{\rm{exp}}({\rm{MLP}}(h_u,h_v))}{\sum_{v^\prime\in\mathcal{N}(u)}{\rm{exp}}({\rm{MLP}}(h_u,h_{v^\prime}))}$$
- 同时可以设计多头注意力
$$\begin{aligned}
m_{\mathcal{N}(u)}&=[a_1{\oplus}a_2{\oplus}\cdots{\oplus}a_K]\\
a_k&=W_k\sum\limits_{v\in\mathcal{N}(u)}\alpha_{u,v,k}h_v
\end{aligned}$$
其中 $\alpha_{u,v,k}$  可以使用前面定义的任意一种注意力权重。

### 8.2.4 更新操作
- 在图神经网络的训练过程中，随着网络层数的增加和迭代次数的增加，同一连通分量内的节点的表征会趋向于收敛到同一个值，出现过度平滑现象。更新操作可以帮助解决这一问题。
#### 拼接和跳跃连接(Concatenation and Skip-Connections)
过度平滑是GNN的核心问题，更新节点表示过于依赖前面层对于邻域信息的聚合，为了缓解这个问题，可以类比ResNet的残差连接方法：
- 拼接跳跃连接(Concatenation)，在信息传递过程中通过将上一层的特征拼接起来，来保留前面层的更多信息：
$${\rm{UPDATE}}_{\rm{concat}}(h_u,m_{\mathcal{N}(u)})=[{\rm{UPDATE}}_{\rm{base}}(h_u,m_{\mathcal{N}(u)}){\oplus}h_u]$$
其中 ${\rm{UPDATE}}_{\rm{base}}$ 是前面定义的基础UPDATE函数。
- 线性插值法：
$${\rm{UPDATE}}_{\rm{interpolate}}(h_u,m_{\mathcal{N}(u)})=\alpha_1\circ{\rm{UPDATE}}_{\rm{base}}(h_u,m_{\mathcal{N}(u)})+\alpha_2{\odot}h_u$$
其中 $\alpha_2=1-\alpha_1$ ，"$\odot$" 是逐元素相乘。

#### 门控单元
- 通常来说，在RNN模型中使用的门控参数更新方法都可以类比到GNN中，在实践中经常使用LSTM的方法，有时还会使用共享参数的技巧。

#### 跳跃知识连接(Jumping Knowledge Connections)
由于我们定义节点嵌入为最后一层的输出：$$z_u=h_u^{(K)},{\forall}u{\in}\mathcal{V}$$，不可避免的要使用残差网络和门控单元来避免过度平滑。
- 另一种方式是将每一层的输出都考虑进来：
$$z_u=f_{\rm{JK}}(h_u^{(0)}{\oplus}h_u^{(u)}{\oplus}\cdot{\oplus}h_u^{({\mathcal{K}})})$$
其中 $f_{\rm{JK}}$ 是任意的可微函数。
- $f_{\rm{JK}}$ 函数通常使用恒等映射，也可以用其他的方法，例如最大池化方法或者LSTM注意力层等。

### 8.2.5 多关系GNN(Multi-relational GNNs)
#### 关系图神经网络(Relational Graph Neural Networks)
- 最初由RGCN提出，这种方法中，我们通过一个可分的转移矩阵来定义聚合函数，以适应多种的关系类型：
$$m_{\mathcal{N}(u)}=\sum\limits_{\tau\in\mathcal{R}}\sum\limits_{v\in\mathcal{N}_{\tau(u)}}\frac{W_{\tau}h_v}{f_n(\mathcal{N}(u),\mathcal{N}(v))}$$
其中 $f_n$ 是归一化函数。
#### 共享参数
- 当图中节点和连接增加时，模型参数增多，这会导致训练变慢以及过拟合问题。这时候可以采用共享参数的方式：
$$W_\tau=\sum\limits_{i=1}^b\alpha_{i,\tau}B_i$$
在这种方法中，所有的参数通过 $b$ 个矩阵 $B_i$ 来表示。
- 这样我们可以重写聚合函数：
$$m_{\mathcal{N}(u)}=\sum\limits_{\tau\in\mathcal{R}}\sum\limits_{v\in\mathcal{\tau(u)}}\frac{\alpha_\tau\times_1\mathcal{B}\times_2{h_v}}{f_n(\mathcal{N}(u),\mathcal{N}(v))}$$
其中 $\mathcal{B}=(B_1,\cdots,B_b)$ 是由矩阵 $B$ 构成的向量，$\alpha_\tau=(\alpha_{1,\tau},\cdots,\alpha_{b,\tau})$ ，$\times_i$ 是模式 $i$ 下的张量积。

### 8.2.6 全图嵌入
- 前面的特征提取操作是用来提取节点嵌入，还需要提取图的整体嵌入：
#### 图池化方法
- 第一种方法是对节点求和(或均值)：
$$z_\mathcal{G}=\frac{\sum_{v{\in}V}z_u}{f_n(|\mathcal{V}|)}$$
其中 $f_n$ 是归一化函数。
- 第二种方法是用LSTM和注意力机制来池化，通过下述方程来迭代：
$$\begin{aligned}
q_t&={\rm{LSTM}}(o_{t-1},q_{t-1}),\\
e_{v,t}&=f_a(z_v,q_t),{\forall}v{\in}\mathcal{V},\\
a_{v,t}&=\frac{{\rm{exp}}(e_{u,i})}{\sum_{u{\in}\mathcal{V}}{\rm{exp}}(e_{u,t})},{\forall}v{\in}\mathcal{V},\\
o_t&=\sum\limits_{v{\in}\mathcal{V}}a_{v,t}z_v.
\end{aligned}$$
初始化 $q_0$ 和 $o_0$ 为0向量，经过迭代后计算全图嵌入：
$$z_{\mathcal{G}}=o_1{\oplus}o_2{\oplus}\cdots{\oplus}o_T$$

#### 图聚类或粗化(Graph coarsening)
- 前一种池化方法的局限在于没有使用图的结构信息，一个流行的算法是通过图的聚类或粗化来代替均值从而进行池化。
- 假定我们有一些聚类函数将图的节点映射到c聚类：
$$f_c:\rightarrow\mathcal{G}\times\mathbb{R}^{|V|{\times}d}\rightarrow\mathbb{R}^{+|V|{\times}c}$$
并且通过映射 $f_c$ 得到矩阵 $S$：
$${\rm{S}}=f_c(\mathcal{G},{\rm{Z}})$$
其中 ${\rm{S}}[u,i]\in\mathbb{R}^+$ 表示节点 $u$ 和聚类 $i$ 距离。
- 通过 $S$ 来计算一个新的粗邻接矩阵以及节点特征：
$${\rm{A^{new}}}={\rm{S^{\top}AS}}\in\mathbb{R}^{+c{\times}c}$$
$${\rm{X^{new}}}={\rm{S^{\top}X}}\in\mathbb{R}^{c{\times}d}$$
这样这些新的邻接矩阵就可以携带图聚类的信息。然后继续通过GNN网络来进行训练。
- 这个方法能够提升训练性能，但也会使训练变得不稳定和困难。有时候删除一些节点信息可以在计算复杂性方面带来一些好处。

### 8.2.7 广义信息传递(Generalized Message Passing)
- GNN信息传递框架也可以用来收集边和图层面的信息，通过下述方程定义信息传递框架：
$$\begin{aligned}
h_{(u,v)}^{(k)}&={\rm{UPDATE_{edge}}}(h_{(u,v)}^{(k-1)},h_u^{(k-1)},h_v^{(k-1)},h_{\mathcal{G}}^{(k-1)})\\
m_{\mathcal{N}(u)}&={\rm{AGGREGATE_{node}}}(\{h_{(u,v)}^{(k)},{\forall}v{\in}\mathcal{N}(u)\})\\
h_u^{(k)}&={\rm{UPDATE_{node}}}(h_u^{(k-1)},m_{\mathcal{N}(u)},h_{\mathcal{G}}^{(k-1)})\\
h_{\mathcal{G}}^{(k)}&={\rm{UPDATE_{graph}}}(h_{\mathcal{G}}^{(k-1)},\{h_u^{(k)},{\forall}u{\in}\mathcal{V}\}\{h_{(u,v)}^{(k)},\forall(u,v)\in\mathcal{E}\})
\end{aligned}$$

## 8.3 图神经网络训练部分理论基础
为了提高训练的精度和性能，我们从以下几个方面进行讨论。
### 8.3.1 损失函数
- 现如今，GNNs主要应用在以下三类任务中：节点分类、图分类和关系预测。针对不同的下游人物，需要采用不同的损失函数。
#### 节点分类任务
- GNNs在节点分类任务上的标准做法是完全监督训练，其中损失函数使用分类任务常用的softmax和似然函数方式：
$$\mathcal{L}=\sum\limits_{u\in\mathcal{V}_{\rm{train}}}-{\rm{log}}({\rm{softmax}}(z_u,y_u))$$
其中，我们假定 $y_u\in\mathbb{Z}^c$ 是一个独热编码的向量，用来表示训练节点 $u\in\mathcal{V}_{\rm{train}}$ 的类。
- ${\rm{softmax}}(z_u,y_u)$ 是节点属于 $y_u$ 类的可能性：
$${\rm{softmax}}(z_u,y_u)=\sum\limits_{i=1}^cy_u[i]\frac{e^{z_u^{\top}w_i}}{\sum_{j=1}^ce^{z_u^{\top}w_j}}$$
$w_i$ 是可训练的参数。
#### 图分类任务
- 历史上，图分类任务通常使用核方法来解决，在这些任务中，分类 ${\rm{softmax}}$ loss也经常使用。
- 近些年，在涉及图数据的回归任务中，GNNs也取得了成功，尤其是在预测蛋白质分子结构的任务，这些任务通常使用平方误差损失：
$$\mathcal{L}=\sum\limits_{\mathcal{G}_i\in\mathcal{T}}||{\rm{MLP}}(z_{\mathcal{G}_i})-y_{\mathcal{G}_i}||_2^2$$
其中，${\rm{MLP}}$ 是一个单变量输出的稠密连通神经网络，$y_{\mathcal{G}_i}\in\mathbb{R}$ 是训练集 $\mathcal{G}_i$ 的目标值。
#### 关系预测任务
- GNNs也用于关系预测任务，例如推荐系统和知识图谱任务等。
- 这种任务中一般使用成对节点嵌入损失函数，
#### 预训练
- 预训练方法在GNNs任务中作用并不明显，随机初始化甚至优于邻域重建损失的预训练，即使如此，使用其他预训练策略仍取得了积极结果，例如最大化 $z_u$ 和 $z_{\mathcal{G}}$ 之间的相互信息:
$$\mathcal{L}=-\sum\limits_{u\in\mathcal{V}_{\rm{train}}}\mathbb{E}_{\mathcal{G}}{\rm{log}}(D(z_u,z_{\mathcal{G}}))+\gamma\mathbb{E}_{\mathcal{G}}{\rm{log}}(1-D(\tilde{z_u},z_{\mathcal{G}}))$$
其中，$z_u$ 是由原图 $\mathcal{G}$ 生成的节点嵌入，$\tilde{z_u}$ 是 $\mathcal{G}$ 经过修改或破坏后生成的，这种修改或破坏通常是以某种随机方式修改节点或邻接矩阵。$D$ 是判别函数，训练来预测节点嵌入是前面两类的哪一类。

### 8.3.2 效率问题和节点采样
#### 图级实现
- 图级实现通过矩阵乘法来加快计算效率，例如，基础GNN的图级方程定义为：
$$\mathbf{H}^{(k)}=\sigma\left(\mathbf{AH}^{(k-1)}\mathbf{W}_{\rm{neigh}}^{(k)}+\mathbf{H}^{(k-1)}\mathbf{W}_{\rm{self}}^{(k)}\right)$$
其中 $\mathbf{H}^{(k)}$ 是包含所有节点的 $k$ 层嵌入的矩阵。
- 这种方法的优势在于只需要进行一次的矩阵乘法，但是需要将图中所有的信息一次运算，没办法进行minibatch训练。
#### 子采样和mini-batch
- 为了进行小批量训练，我们需要处理子图，但随机采样很可能会丢失重要信息，甚至会取到不连通的子图。
- Hamilton等人提出，可以先选取一组目标节点作为一个batch，再对其邻点集进行递归采样，保证连通性的同时以固定的样本大小来进行运算。

### 8.3.3 参数共享和正则化
正则化是机器学习领域的重要方法，像L2正则、dropout和layer norm就被广泛的应用
#### 层之间的参数共享
- 在具有多层消息传递的GNNs中经常采用的一种策略。核心思想是在GNN中的所有聚合和更新函数中使用相同的参数。一般来说，这种方法在六层以上的GNN中最有效，并且经常与门控单元结合使用。
#### Edge Dropout
- 另一种在GNN训练时常用的策略为边缘Dropout。我们在训练期间随机删除(或屏蔽)邻接矩阵中的某些边，这种方法可以使模型不容易过拟合。

## 8.4 本章总结
本章学习了图神经网络的基础理论，主要包括：
- 信息传递框架
- 聚合和更新操作
- 多关系GNN和图池化
- 损失函数定义和一些优化方法