# Chapter10 GCN论文精读
用于半监督节点分类的图卷积神经网络GCN。
## 10.1 GCN前瞻
- GCN提供了一种半监督学习的可扩展的图数据挖掘方法，它是卷积神经网络的一个变体，通过学习隐藏层的参数来表征图的局部结构和节点特征。
- 传统的卷积学习的是欧式空间中的数据模型，但是在非欧几何数据中却没法实现，很重要的一个原因就是传统的卷积在非欧几何中无法保持“平移不变性”。
- 而GCN则将传统的卷积理念推广到非欧几何的图数据上。

## 10.2 图上的卷积定义
### 10.2.1 卷积、傅里叶变换和拉普拉斯矩阵
在进行GCN论文精读前，我们先来看一下卷积、傅里叶变换和拉普拉斯矩阵的关系。
#### 卷积
- 在数学中，两个函数 $f,g:\mathbb{R}^n\rightarrow\mathbb{R}$ 之间的卷积定义为：
$$(f*g)(x)=\int_{\mathbb{R}^n}f(z)g(x-z)dz$$
- 可以将其理解为：当把一个函数"翻转"并"移位" $x$ 后，$f$和$g$之间的重叠。
- 例如，如果是 $\mathbb{Z}$ 上的平方可加的无限维向量集合中抽取向量，可以得到如下定义：
$$(f*g)(i)=\sum\limits_af(a)g(i-a)$$
- 卷积的平移不变性：
$$f(t+a)*g(t)=f(t)*g(t+a)=(f*g)(t+a)$$
- 由上式可以得到推论：
$${\Delta}f(t)*g(t)=f(t)*{\Delta}g(t)={\Delta}(f*g)(t)$$
其中 ${\Delta}f(t)=f(t+1)-f(t)$ 是离散单变量信号的拉普拉斯算子。

#### 傅里叶变换
- 可积函数 $f:\mathbb{R}\rightarrow\mathbb{C}$ 的(连续)傅里叶变换定义为：
$$\mathcal{F}[f(x)]=\hat{f}(\xi)=\int_{-\infty}^{\infty}f(x)e^{-2{\pi}ix{\xi}}dx$$
其中， $\xi$ 为任意实数，其定义域称作频域。
- 如果约定自变量 $x$ 表示时间(以秒为单位)，变换变量 $\xi$ 表示频率(以赫兹为单位)。在适当条件下，$\hat{f}$ 可由逆傅里叶变换得到 $f$ ：
$$\mathcal{F}^{-1}[\hat{f}(\xi)]=f(x)=\int_{-\infty}^{\infty}\hat{f}(\xi)e^{2{\pi}i{\xi}x}d\xi$$
其中，$x$ 为任意实数，其定义域称作时域。
- 傅里叶变换可以理解为函数 $f$ 在一组基 $e^{-2{\pi}i{\xi}}$ 上的投影分解。

#### 傅里叶变换的卷积特性
- 若函数 $f(x)$ 及 $g(x)$ 都在 $(-\infty,+\infty)$ 上绝对可积，则卷积函数 $f*g$ 的傅里叶变换存在，且有：
$$\mathcal{F}[f*g]=\mathcal{F}[f]\cdot\mathcal{F}[g]$$
- 稍微变换上式的形式我们可以得到，两函数的卷积是其傅里叶变换的逆变换：
$$f*g=\mathcal{F}^{-1}[\hat{f}\cdot\hat{g}]$$

### 10.2.2 图的拉普拉斯矩阵
#### 拉普拉斯算子
- 多元函数 $f(x_1,\cdots,x_n)$ 的拉普拉斯算子是所有自变量的二阶偏导之和：
$${\Delta}f=\sum\limits_{i=1}^n\frac{\partial^2f}{{\partial}x_i^2}$$
- 如果是二元函数 $f(x,y)$ ，其拉普拉斯算子可以写为如下形式：
$${\Delta}f=\sum\limits_{(k,l){\in}N(i,j)}\left(f(x_k,y_l)-f(x_i,y_j)\right)$$
其中 $N(i,j)$ 是 $(x_i,y_j)$ 的邻点集。
#### 拉普拉斯矩阵
- 对应到图中，如果将图的顶点处的值看作是函数值，那么在顶点处的拉普拉斯算子可以写为：
$${\Delta}f_i=\sum\limits_{j{\in}N_i}(f_i-f_j)$$
其中 $N_i$ 是节点 $i$ 的邻点集。
- 因为图的边带有权重，我们可以改写上式：
$${\Delta}f_i=\sum\limits_{j{\in}N_i}w_{ij}(f_i-f_j)$$
- 如果 $j$ 不是 $i$ 的邻点，我们可以令 $w_{ij}=0$ ，这样上式可以继续改写为：
$${\Delta}f_i=\sum\limits_{j{\in}\mathcal{V}}w_{ij}(f_i-f_j)=\sum\limits_{j{\in}\mathcal{V}}w_{ij}f_i-\sum\limits_{j{\in}\mathcal{V}}w_{ij}f_j=d_if_i-\mathbf{w}_i\mathbf{f}$$
其中 $d_i$ 是节点 $i$ 的加权度，$\mathbf{w}_i$ 是邻接矩阵的第 $i$ 行，$\mathbf{f}$ 是所有节点构成的列向量。
- 对于所有顶点有：
$${\Delta}f=\begin{gather*}
\begin{bmatrix}
{\Delta}f_1\\
\quad\\
\cdots\\
\quad\\
{\Delta}f_n
\end{bmatrix}
\end{gather*}=\begin{gather*}
\begin{bmatrix}
d_1f_1-\mathbf{w}_1\mathbf{f}\\
\quad\\
\cdots\\
\quad\\
d_nf_n-\mathbf{w}_n\mathbf{f}
\end{bmatrix}
\end{gather*}=\begin{gather*}
\begin{bmatrix}
d_1 & \cdots  &\cdots\\
\quad\\
\cdots & \cdots & \cdots\\
\quad\\
\cdots & \cdots & d_n
\end{bmatrix}
\end{gather*}\begin{gather*}
\begin{bmatrix}
f_1\\
\quad\\
\cdots\\
\quad\\
f_n
\end{bmatrix}
\end{gather*}-\begin{gather*}
\begin{bmatrix}
\mathbf{w}_1\\
\quad\\
\cdots\\
\quad\\
\mathbf{w}_n
\end{bmatrix}
\end{gather*}\begin{gather*}
\begin{bmatrix}
f_1\\
\quad\\
\cdots\\
\quad\\
f_n
\end{bmatrix}
\end{gather*}=(\mathbf{D}-\mathbf{W})\mathbf{f}
$$
#### 图的拉普拉斯矩阵
上面的结论启发我们，可以在邻接矩阵和加权度矩阵的基础上定义拉普拉斯矩阵。
- 图 $G$ 的邻接矩阵和加权度矩阵分别为 $\mathbf{A}$ 和 $\mathbf{D}$，如果对图的每个节点加入自环，那么新图的邻接矩阵和度矩阵为：
$$\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}$$
$$\tilde{\mathbf{D}}=\mathbf{D}+\mathbf{I}$$
其中 $\mathbf{I}$ 为单位阵。
- 图的拉普拉斯矩阵定义为：
$$\mathbf{L}=\mathbf{D}-\mathbf{A}$$
- 归一化拉普拉斯矩阵为：
$$\mathbf{L}_{\rm{sym}}=\mathbf{D}^{-\frac{1}{2}}\mathbf{L}\mathbf{D}^{-\frac{1}{2}}=\mathbf{I}-\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}$$
- 对于 $\forall\mathbf{f}\in\mathbb{R}^n$，有：
$$\mathbf{f}^\top\mathbf{L}_{\rm{sym}}\mathbf{f}=\frac{1}{2}\sum\limits_{i=1}^n\sum\limits_{j=1}^na_{ij}\left(\frac{f_i}{\sqrt{d_i}}-\frac{f_j}{\sqrt{d_j}}\right)^2$$
#### 邻接矩阵和拉普拉斯矩阵
- 归一化邻接矩阵为：
$$\mathbf{A}_{\rm{sym}}=\mathbf{D}^{-\frac{1}{2}}\mathbf{L}\mathbf{D}^{-\frac{1}{2}}=\mathbf{I}-\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}$$
- 根据定义我们有 $\mathbf{L}_{\rm{sym}}=\mathbf{I}-\mathbf{A}_{\rm{sym}}$，两个归一化矩阵的特征分解有下面的关系：
$$\mathbf{L}_{\rm{sym}}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$$
$$\mathbf{A}_{\rm{sym}}=\mathbf{U}(\mathbf{I}-\mathbf{\Lambda})\mathbf{U}^\top$$
这意味着两种矩阵可以互换使用，我们可以用拉普拉斯矩阵代替邻接矩阵进行图的特征提取。

### 10.2.3 图傅里叶变换
#### 傅里叶变换和拉普拉斯矩阵的关系
- 根据前面的讨论，我们知道，傅里叶变换可以理解为在一组基上的函数分解，拉普拉斯算子则相当于对其导数 ${\nabla}f(x)$ 求散度 $\nabla$，直觉上告诉了我们节点和其邻点集的平均差值。
- 根据下述式子：
$$-\Delta(e^{2{\pi}ist})=\frac{\partial^2(e^{2{\pi}ist})}{{\partial}t^2}=(2{\pi}s)^2e^{2{\pi}ist}$$
我们能够知道，拉普拉斯算子的特征方程对应于复指数 $e^{2{\pi}is}$，而其正是傅里叶变换函数分解的基。

#### 图傅里叶变换
上述关系帮助我们去定义图上的傅里叶变换。
- 考虑图拉普拉斯矩阵分解：
$$\mathbf{L}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$$
- 由于拉普拉斯算子和傅里叶变换的对应关系，我们将图拉普拉斯特征向量 $\mathbf{U}$ 定义为图的傅里叶模式，矩阵 $\mathbf{\Lambda}$ 的对角线上是特征值，这些特征值相当于图上的频率值。
- 一个信号或函数 $\mathbf{f}\in\mathbb{R}^{|\mathcal{V}|}$ 的傅里叶变换可以通过下式计算：
$$\hat{\mathbf{f}}=\mathbf{U}^\top\mathbf{f}$$
它的逆傅里叶变换为：
$$\mathbf{f}=\mathbf{U}\hat{\mathbf{f}}$$

### 10.2.4 图卷积
- 我们可以通过哈达玛积计算图卷积：
$$\mathbf{f}*_{\mathcal{G}}\mathbf{h}=\mathbf{U}(\mathbf{U}^\top\mathbf{f}\circ\mathbf{U}^\top\mathbf{h})$$
- 我们一般将 $\mathbf{f}$ 看作输入的Graph的节点特征，将 $\mathbf{h}$ 视为可训练且参数共享的卷积核来提取拓扑图的空间特征。我们用 $\theta_h$ 来表示 $\mathbf{U}^\top\mathbf{f}$，那么有：
$$\mathbf{f}*_{\mathcal{G}}\mathbf{h}=\mathbf{U}(\mathbf{U}^\top\mathbf{f}\circ\theta_h)=(\mathbf{U}{\rm{diag}}(\theta_h)\mathbf{U}^\top)\mathbf{f}$$
- 然而，这样定义并不能满足我们需要的很多卷积性质，为了保证滤波 $\theta_h$ 对应于一个有意义的图卷积，可以将其变换为拉普拉斯特征值组成的矩阵 $p_N(\mathbf{\Lambda})$，其特征值与 $\theta_h$ 的特征值共轭：
$$\mathbf{f}*_{\mathcal{G}}\mathbf{h}=(\mathbf{U}p_N(\mathbf{\Lambda})\mathbf{U}^\top)\mathbf{f}=p_N(\mathbf{L})\mathbf{f}$$
这样，我们就用拉普拉斯特征值组成的多项式来表示了图卷积。


## 10.3 GCN论文
- 多层图卷积网络
$$H^{(l+1)}=\sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$$
其中 $W^{(l)}$ 是一个可训练的权重矩阵，$\sigma$ 是一个非线性函数。</br>
下面我们将证明，上式可由图上的局部频谱的一阶逼近得到。

### 10.3.1 谱图卷积
- 根据10.2.4中的推导过程，考虑一个信号 $x\in\mathbb{R}^N$ 和一个滤波 $g_\theta={\rm{diag}}(\theta),\theta\in\mathbb{R}^N$，我们有：
$$g_\theta*x=\mathbf{U}g_\theta\mathbf{U}^{\top}x$$
其中 $\mathbf{L}=\mathbf{I}_N-\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}=\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$，$\mathbf{U}^{\top}x$ 是 $x$ 的图傅里叶变换。
- $g_\theta$ 可以看作 $\mathbf{L}$ 的特征向量上的函数，既 $g_\theta(\mathbf{\Lambda})$。
- 计算上述式子耗费很大，对于较大的图来说，求特征分解也是很难的事，为此，我们可以用切比雪夫多项式 $T_k(x)$ 来逼近 $g_\theta(\mathbf{\Lambda})$：
$$g_{\theta^\prime}(\mathbf{\Lambda})\approx\sum\limits_{k=0}^K\theta_k^{\prime}T_k(\tilde{\mathbf{\Lambda}})$$
其中 $\tilde{\mathbf{\Lambda}}=\frac{2}{\lambda_{\rm{max}}}\mathbf{\Lambda}-\mathbf{I}_N$，$\lambda_{\rm{max}}$ 是 $\mathbf{L}$ 的最大的特征值。
- 注：切比雪夫多项式是：$T_k(x)=2xT_{k-1}(x)-T_{k-2}(x)$，初值 $T_0(x)=1,T_1(x)=x$。
- 联立上述几式，我们得到：
$$g_{\theta^\prime}*x\approx\sum\limits_{k=0}^K\theta_k^{\prime}T_k(\tilde{\mathbf{L}})x$$
其中 $\tilde{\mathbf{L}}=\frac{2}{\lambda_{\rm{max}}}\mathbf{L}-\mathbf{I}_N$。

### 10.3.2 逐层线性模型
- 像社交网络、引文网络、知识图谱这样的真实世界图数据集，都具有非常广的节点度分布，我们希望图卷积模型能在处理这些图的局部邻域结构上缓解过拟合问题。因此，我们需要舍弃高维的切比雪夫逼近。
- 另一方面，逐层的线性模型能够方便我们构建深度模型。
- 由以上两点，考虑令 $K=1$，得到一个低维线性模型，然后通过这个线性模型来构建深度神经网络。更进一步，我们考虑 $\lambda_{\rm{max}}\approx2$，用来降低模型训练时的计算。此时我们可以改写模型为：
$$g_{\theta^\prime}*x\approx\theta_0^{\prime}x+\theta_1^\prime(\mathbf{L}-\mathbf{I}_N)x=\theta_0^{\prime}x-\theta_1^{\prime}\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}x$$
- 在深度模型训练过程中可以共享参数。
- 在实践中，还可以进一步降低参数的规模，令 $\theta=\theta_0^\prime=-\theta_1^\prime$，使用下式来训练：
$$g_{\theta^\prime}*x\approx\theta(\mathbf{I}_N+\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}})x$$
- 我们令 $\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}_N$，$\tilde{\mathbf{D}}_{ii}=\sum_j\tilde{\mathbf{A}_{ij}}$，使得：
$$\mathbf{I}_N+\mathbf{D}^{-\frac{1}{2}}\mathbf{A}\mathbf{D}^{-\frac{1}{2}}\rightarrow\tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}}$$
能够在一定程度上避免梯度消失问题。

### 10.3.3 模型最终形式
- 一个 $C$ 维信号 $X\in\mathbb{R}^{N{\times}C}$(既每个节点的 $C$ 维特征向量)，和一个特征映射：
$$\mathbf{Z}=\tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}}X\Theta$$
其中 $\Theta\in\mathbb{R}^{C{\times}F}$ 是滤波参数矩阵，$\mathbf{Z}\in\mathbb{R}^{N{\times}F}$ 是卷积信号矩阵。
- 算法复杂度为 $\mathcal{O}(|\mathcal{E}|FC)$

## 10.4 本章总结
本章学习了GCN的基础理论，主要包括：
- 卷积、傅里叶变换、拉普拉斯矩阵的概念以及三者之间的关系
- 图卷积的动机来源及公式推导
- 一维图卷积的简化方法
- 深度图卷积模型