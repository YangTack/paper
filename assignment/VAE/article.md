<font size=4dp>
<div style="5%;">

# VAE -- Variational Autoencoder 变分自编码器 报告


<p style="font-size: 20px" align="right">
杨少博
</p>

## 概念
由于传统自编码器编码出的隐变量分布的不确定性，没有办法使输出连续。VAE提出将隐变量空间假设成标准正态分布。

<center>

 $$ \mathcal{Z} \sim \mathcal{N}(0, 1) $$ 

 </center>

其中 $\mathcal{Z}$ 即是VAE中的隐变量

> *"可以理解为: 图片的各种特征，例如，亮度，棱角，高级特征如皱纹，肤色等（输入为图片的情况下），但实际并不一定是人类所感知的特征"*

使用一个$Encoder$将原始图片 $\mathcal{x}$ 映射到 $\mathcal{z} \sim \mathcal{Z}$ 中，再使用一个$Decoder$将图片还原成 $\mathcal{x^\prime}$ 

结构如下：

<div align="center">
<img src=img/struct.png width=500 />
</div>

## VAE 假设

VAE有如下假设:
- $z \sim \mathcal{N}(0, I)$，
隐变量服从标准正态分布(先验分布)
- $P(x|z) \sim Norm \; \text{or} \; Bern$, 其他研究发现换成别的影响不大。

## Variational Inference

变分推断是VAE的理论基础

首先对于对数似然：

$$
\begin{align*}
&E_{x \sim D} ( \log { ( p_\theta (x) ) } ) \\
&\approx N^{-1} * \sum_i^N { \log { ( p_\theta (x_i) ) } }
\end{align*}
$$

能取最大，其中$\theta$为模型参数

接下来对于单个样本$x$进行推导

$$

p( x ) = { p( x, z ) \over p( z | x ) } \\

$$

对于含有隐变量$z$的模型来说，$p( x, z ) = p( x | z ) * p( z )$，其中只有$p( x | z )$ 作为重构项带有模型参数，$p( z )$，没有模型参数。

换句话说$p( x, z )$是我们可以处理的，而后验概率$p( z | x )$不可解，因为其依赖于$p( x )$，而$p( x )$正是我们需要求解的。

Variational Inference 则是以一个简单分布$q( z )$代替$p( z | x )$

### 1. ELBO推导

$$ 

\begin{align*}

\log { ( p( x ) ) } &= \log{ \int_z p( x, z ) dz } \\

& = \log { \int_z q( z ) { p( x, z ) \over q( z ) } dz } \\

& =  \log { E_{ z \sim q( z ) } ( { p( x, z ) \over q( z ) }  ) } \\

& \geq  E_{ z \sim q( z ) } ( \log { p( x, z ) \over q( z ) }  ) \\

& = E_{ z \sim q( z ) } ( \log { p( x, z ) } ) + H( q( z ) ) \qquad \small { (ELBO) } \\

\end{align*}

$$

接下来，关注$q( z )$与$p( z | x )$之间的距离

$$

\begin{align*}
    
KL( q( z ) || p( z | x ) ) & = \int_z q( z ) \log { q( z ) \over p( z | x ) } dz  \\

& = \log p( x ) - H_{ z \sim q( z ) } - E_{ z \sim q( z ) } ( \log { p( x, z ) } ) \\

& = \log p( x ) - ELBO

\end{align*}

$$

所以，对于任取的$q( z )$而言，ELBO能取的最大值即是$\log p( x )$

以下即为实际优化任务
> - 最大化ELBO 
> - 使 $q( z )$与$p( z | x )$尽量接近 

### 2. $q( z )$ 分布

由上可知

$q( z )$分布可以取得尽量简单，使得我们可以处理

由于$q( z )$与$p( z | x )$需要尽量接近，$q( z )$最好也是需要依赖于$x$，
即$q( z ) \triangleq q( z | x )$

可以取$q( z | x ) = q_\phi( z | x )$对于每个真实数据采样$x_i$有不同的分布参数$\phi_i$，
每次更新模型时，先对每个$\phi_i$进行更新，之后对$\theta$进行更新。

但是这种方法需要的计算资源以及存储资源太大，为了简化，用神经网络定义函数$f_\lambda(x_i) \rightarrow \phi_i$，
进行取舍，其中$\lambda$为神经网络的参数，$q_\phi( z | x ) = q_{ f_\lambda } ( z | x )$

### 3. 优化ELBO

$$

\begin{align*}

ELBO &= E_{ z \sim q( z | x ) } ( \log { p( x, z ) } ) + H( q( z ) ) \\

& = E_{ z \sim q( z | x ) } ( \log { p( x | z ) } ) - KL( q( z | x ) || p( z ) )
    
\end{align*}

$$

ELBO对于$\theta$做梯度上升比较容易，期望项使用Monte Carlo采样，KL项不含$\theta$。
但是对于$\phi$也就是$\lambda$做梯度上升时直接使用Monte Carlo采样就会导致方差过大，因为$z \sim q( z | x )$ 依赖于 $\lambda$。
VAE中使用重采样的方法使得$E_{ z \sim q( z | x ) } ( \log { p( x | z ) } ) = E_\epsilon( \log {p( x | z( \epsilon ) )} )$，$\epsilon$是一个固定分布，
此时期望不再依赖于 $\lambda$ ，就可以做Monte Carlo采样。

<div align="center">
<img src=img/reparameterize.png heigh=300 />
</div>

接下来推导此项

### 4. VAE

VAE中假设$p( z )$是多维标准正态分布，由于ELBO中$KL( q( z | x ) || p( z ) )$要尽量小，所以$q( z | x )$可以采用多维正态分布，$f_\lambda$返回$q( z | x )$的均值$\mu( x )$和对数方差$\log \sigma^2( x )$

此时ELBO

$$

\begin{align*}
    
ELBO & = E_{ z \sim q( z | x ) } ( \log { p( x | z ) } ) - KL( q( z | x ) || p( z ) ) \\
& = E_{ z \sim q( z | x ) } ( \log { p( x | z ) } ) - 
\frac{1}{2}
\sum_{i=0}^{k}
\{
    \mu_i^2
    +\sigma_i^2
    -\log{\sigma_i^2}
    - 1
\} \\

\end{align*} \\

\begin{align*}
    
E_{ z \sim q( z | x ) } ( \log { p( x | z ) } ) 

& = E_\epsilon( \log p(x | x, \epsilon ) ) \\

& \approx \log ( p( x | x, \epsilon ) ) \\

& \propto -(\hat{x} - x)^2

\end{align*}

$$

其中$k$为$z$的维度

总结以上

变分推断是使用简单分布$q( z )$近似$p( z | x )$使得$p( x )$可以求出。

在VAE中用于近似$p( z | x )$的分布$q( x | z )$为多维正态分布，其参数由$f_\lambda$给出，
$f_\lambda$在VAE结构中也是Encoder的结构，而$p( x | z )$为Decoder。
