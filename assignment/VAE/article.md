<font size=5dp>
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


由上图可知，VAE的目标是得到真实世界的分布 $P(x)$ ，
由*Bayes*公式，$P(x) = \frac{P(x, z)}{P(z|x)}$，
要想知道 $P(x)$ ，就需要知道 $P(z|x)$, 
但如果要知道 $P(z|x)$，
同样由*Bayes*公式，$P(z|x) = \frac{P(x|z)}{P(z)/P(x)}$ ，就必须知道 $P(x)$ 。

因此VAE使用一个分布$q(z|x)$来近似$P(z|x)$

## VAE 假设

VAE有如下假设:
- $z \sim \mathcal{N}(0, I)$，
隐变量服从标准正态分布
- $P(x|z) \sim Norm \; \text{or} \; Bern$, 其他研究发现换成别的影响不大。

## VAE 推导

随机变量$x$为真实世界观察到的图像序列 $\{x_1, x_2, ...\}$ 

### 1. ELBO推导及理解
似然函数 $L = P(x)$

目标就是最大化 $L \Rightarrow max \; \log{L}$, 对数似然函数 

$$ 
\begin{align*}

\log{L} &= \log{P(x)}*\int_zq(z|x)  \\

&= \int_z\log{P(x)}*q(z|x) \quad \text{引入P(z|x)} \\

&= \int_zq(z|x)*\log{\frac{P(x,z)}{P(z|x)}} \quad \text{凑KL Divergence} \\

&= \int_zq(z|x)*\log{\frac{P(x,z)*q(z|x)}{(P(z|x)*q(z|x)}} \\

&= \int_zq(z|x)*\log{\frac{q(z|x)}{P(z|x}} + \int_zq(z|x)*\log{\frac{P(x,z)}{q(z|x)}} \\

&= KL(q(z|x)||P(z|x)) + \int_zq(z|x)*\log{P(x,z)/q(z|x)}

\end{align*}
$$

由于 $KL(q(z|x)||P(z|x)) \geq 0$ 当且仅当两个分布完全一样的时为0

得 $\log{L} \geq \int_zq(z|x)*\log{\frac{P(x,z)}{q(z|x)}}$ 
> 后者被称为*ELBO*--*Evidence Lower Bound*
>
> 由于 $\log{L}$ 有上界，当尽量增大*ELBO*时，$KL(q(z|x)||P(z|x))$ 会相应减小
>
> *An Introduction to Variational Autoencoders (2019, Diederik P. Kingma)* 论文中也给出结论，优化*ELBO*相当于优化两件事:
> - 优化 $\log{L} = \log{P(x)}$ 似然函数
> - 优化 $KL(q(z|x)||P(z|x))$ 使得$q(z|x)$ 与 $P(z|x)$更接近

### 2. ELBO拆分

由上可知

可设我们优化目标 $min(L) = min(-ELBO) = min(-\int_zq(z|x)*\log{\frac{P(x,z)}{q(z|x)}})$

$$
\begin{align*}
L &= -\int_zq(z|x)*\log{\frac{P(z)}{q(z|x)}} - \int_zq(z|x)*\log{P(x|z)}  \\
&= KL(q(z|x)||P(z)) - E_{z \sim q(z|x)}(\log{P(x|z)} \\
\end{align*}
$$
我们使用神经网络逼近$q(z|x) \rightarrow q_\theta(z|x)$ 与 $P(x|z) \rightarrow P_\phi(x|z)$，其中 $\theta$ 和 $\phi$ 为这两个神经网络的参数

> $q_\theta(z|x)$ 就是Encoder
>
> $P_\phi(x|x)$ 就是Decoder

此时

$$
\begin{align*}
L_{\theta,\phi} &= KL(q_\theta(z|x)||P(z)) - E_{z \sim q_\theta(z|x)}(\log{P_\phi(x|z)} \\
\end{align*}
$$

#### KL Divergence 部分推导

对于$KL(q_\theta(z|x)||P(z))$项，需要让其尽量小

根据之前的假设 $z \sim \mathcal{N}(0, I)$，
可知$q_\theta(z|x)$也为正态分布。

可设$q_\theta(z|x) \sim \mathcal{N}(\mu_\theta, \Sigma_\theta)$，
其中$\mu_\theta = (\mu_1, \mu_2, ..., \mu_k)^T$，
$\Sigma_\theta$为$z$的协方差矩阵，

即$\Sigma_{\theta\; i,j} = E((z_i - \mu_i)*(z_j - \mu_j)) = Cov(z_i, z_j)$，

又根据假设$z$各个维度相互独立，
$\Sigma_\theta = diag(\sigma_1, \sigma_2, ..., \sigma_k)$

所以可得

$$
\begin{align*}

KL(q_\theta(z|x)||P(z)) &=
 \int_{z}
 {
    q_\theta(z|x) * \log
    {
        \frac
        {
            \prod_{i=0}^{k}
            {
                \frac{1}{\sqrt{2\pi\sigma^2_i}}
                e^{-\frac{1}{2}(\frac{z_i-\mu_i}{\sigma_i})^2}
            }
        }
        {
            \prod_{i=0}^{k}
            {
                \frac{1}{\sqrt{2\pi}}
                e^{-\frac{1}{2}z_i^2}
            }
        }
    }
} \\



&= \int_{z}
{
    q_\theta(z|x)*
    \sum_{i=0}^{k}
    \{
        -\log{\sigma_i}
        - \frac{1}{2}
        [
            (\frac{z_i - \mu_i}{\sigma_i})^2
            - z_i^2
        ]
    \}
} \\


&= \sum_{i=0}^{k}
{
    \int_{z_i}
    {
        q_\theta(z|x)* 
        \{
            -\log{\sigma_i}
            - \frac{1}{2}
            [
                (\frac{z_i - \mu_i}{\sigma_i})^2
                - z_i^2
            ]
        \}
    }
} \\

&= \sum_{i=0}^{k}
{
    \int_{z_i}{
        q_\theta(z_i|x)* 
        \{
            -\log{\sigma_i}
            - \frac{1}{2}
            [
                (\frac{z_i - \mu_i}{\sigma_i})^2
                - z_i^2
            ]
        \}
    }
} \\

&= \sum_{i=0}^{k}
{
    -\log{\sigma_i}
    -\frac{1}{2}
    [
        \frac{1}{\sigma_i^2}
        D_{z_i \sim q_\theta(z_i|x)}(z_i)
        -E(z_i^2)
    ]
} \\

&= \sum_{i=0}^{k}
{
    -\log{\sigma_i}
    -\frac{1}{2}
    (
        1
        -\mu_i^2
        -\sigma_i^2
    )
} \\

&= 
\frac{1}{2}
\sum_{i=0}^{k}
\{
    \mu_i^2
    +\sigma_i^2
    -\log{\sigma_i^2}
    - 1
\}

\end{align*}
$$

$k$为隐变量的维度,

设计*Encoder*输出$\mu_\theta, \log{\sigma_\theta}$，

可得第一部分*loss*为
$$
\frac{1}{2}
\sum_{i=0}^{k}
\{
    \mu_i^2
    +\sigma_i^2
    -\log{\sigma_i^2}
    - 1
\}
$$

#### 重构误差部分推导

对于$- E_{z \sim q_\theta(z|x)}(\log{P_\phi(x|z)}$项，需要让其尽量小

由于这一项涉及到对$z$的期望计算，
并且对$z$积分是不可求的。

解决方法一般为蒙特卡罗方法，即对$z$进行采样，将采样值作为期望值，并且，蒙特卡洛方法是无偏的。

但是直接采样是无法进行反向传播更新$\phi$和$\theta$的，
所以注意到$q_\theta(z|x)$为正态分布，
从$q_\theta(z|x)$中采样相当于从$\mathcal{N}(0, I)$中采样$\epsilon \sim \mathcal{N}(0, I)$，
再$z = \epsilon * \sigma_\theta + \mu_\theta$作为采样值，
论文中将此技术称为重参数化技术(Reparameterization Trick)

<div align="center">
<img src=img/reparameterize.png heigh=300 />
</div>

接下来推导此项

$$
\begin{align*}

- E_{z \sim q_\theta(z|x)}(\log{P_\phi(x|z)}
&\approx
\log{P_\phi(x|z)} \quad \\
\text{z为采样值} \\
\text{根据假设，}P_\phi(x|z) = \mathcal{N}(x;x^{org}, I)\qquad * \\
&= 
\frac{1}{2}
(x - x^{org})^2
 - \log
 {\frac
    {1}
    {\sqrt{2\pi}}
 } \\


&=
\frac{1}{2}
(x - x^{org})^2
+C\\

\end{align*}
$$

> \* 处结论论文中并没有给出解释，查阅相关资料后
>
> 根据 https://stats.stackexchange.com/questions/540092/how-do-we-get-to-the-mse-in-the-loss-function-for-a-variational-autoencoder 的回答
>
> $P_\phi(x|z)$中的方差$\hat{\sigma}^2$是一个超参数，对结果影响不大，一般取$I$即可
>
> 而均值$\hat{\mu}$根据分布是从$z$重建$x$，理所当然应该为$x^{org}$，也可以根据概率公式$E(\hat{\mu}) = E_z(E_{x \sim P_\phi(x|z)}(x)) = E(x) = x^{org}$

### 结论

根据以上推导 

损失函数

$$
L_{\sigma, \phi} = 
\frac{1}{2}
\sum_{i=0}^{k}
\{
    \mu_i^2
    +\sigma_i^2
    -\log{\sigma_i^2}
    - 1
\}
+
\frac{1}{2}
(x - x^{org})^2
$$

其意义为既**减少编码器与$z$分布的误差**又**减少重构$x$时的误差**


</div>