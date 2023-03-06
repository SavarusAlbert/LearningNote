# Chapter12 GAT论文精读
图注意力网络
## 12.1 GAT前瞻
- 自然语言处理领域率先提出的注意力机制，在各个数据集上都获得很好的效果，通过将注意力集中在更有价值的信息上，来有选择性的处理信息，获得更好的结果。
- GAT将注意力机制引入到图中，通过自注意力来对图进行邻域聚合，设置若干个功能相同的图注意力层，来提取特征信息。利用多头注意力来更稳定的进行训练。

## 12.2 GAT优劣
### 12.2.1 优点
- GAT模型的自注意力层能够很好的并行计算，无需做矩阵的特征分解，模型效率高。
- 可以处理有向图。
- 可以进行归纳学习，处理未知节点。
### 12.2.2 缺点
- 没有利用边的信息，只利用了连接信息。
- 只利用了1跳的信息，多跳后会出现过度平滑。

## 12.3 GAT论文-图的注意力机制
### 12.3.1 层的输入输出
- 层的输入为一组节点特征：
$$\mathbf{h}=\{\vec{h}_1,\vec{h}_2,\cdots,\vec{h}_N\},\vec{h}_i\in\mathbb{R}^F$$
其中 $N$ 是节点数，$F$ 是节点的特征维度。
- 层的输出为新的节点特征：
$$\mathbf{h}^\prime=\{\vec{h}_1^\prime,\vec{h}_2^\prime,\cdots,\vec{h}_N^\prime\},\vec{h}_i^\prime\in\mathbb{R}^{F^\prime}$$

### 12.3.2 自注意力机制
- 自注意力机制 $a:\mathbb{R}^{F^\prime}\times\mathbb{R}^{F^\prime}\rightarrow\mathbb{R}$：
$$e_{ij}=a(\mathbf{W}\vec{h}_i,\mathbf{W}\vec{h}_j)$$
其中 $\mathbf{W}\in\mathbb{R}^{F^\prime{\times}F}$ 是权重矩阵。
- 我们通过mask掉非邻域内的节点，只考虑 $j\in\mathcal{N}_i$ 的自注意力 $e_{ij}$，来引入图的结构信息。对于有向边 ${i{\rightarrow}j}$，mask掉 $e_{ji}$ 就可以引入方向信息。
- 为了比较不同节点之间的权重系数，我们进行归一化操作：
$$\alpha_{ij}={\rm{softmax}}_j(e_{ij})=\frac{{\rm{exp}}(e_{ij})}{\sum_{k\in\mathcal{N}_i}{\rm{exp}}(e_{ik})}$$

### 12.3.3 模型最终形式
- 复合上述函数，得到：
$$\alpha_{ij}=\frac{{\rm{exp}}\left({\rm{LeakyReLU}}(a^\top[\mathbf{W}\vec{h}_i\oplus\mathbf{W}\vec{h}_j])\right)}{\sum_{k\in\mathcal{N}_i}{\rm{exp}}\left({\rm{LeakyReLU}}(a^\top[\mathbf{W}\vec{h}_i\oplus\mathbf{W}\vec{h}_k])\right)}$$
其中 $a$ 是可训练的注意力机制，$\oplus$ 是concatenation算子。
- 然后通过下式输出最终的节点特征：
$$\vec{h}_i^\prime=\sigma\left(\sum\limits_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}\vec{h}_j\right)$$
- 引入多头注意力，能够让训练更稳定：
$$\vec{h}_i^\prime=\mathop{\oplus}\limits_{k=1}^K\sigma\left(\sum\limits_{j\in\mathcal{N}_i}\alpha_{ij}^k\mathbf{W}^k\vec{h}_j\right)$$
- 在模型的最后一层我们需要做预测输出特征，所以我们将非线性函数放在最外层，然后不再使用concatenation算子，而是用均值操作代替：
$$\vec{h}_i^\prime=\sigma\left(\frac{1}{K}\sum\limits_{k=1}^{K}\sum\limits_{j\in\mathcal{N}_i}\alpha_{ij}^k\mathbf{W}^k\vec{h}_j\right)$$

## 12.4 本章总结
本章主要学习了GAT的注意力机制模型，多头注意力机制，在图上很好的聚合了邻域信息。