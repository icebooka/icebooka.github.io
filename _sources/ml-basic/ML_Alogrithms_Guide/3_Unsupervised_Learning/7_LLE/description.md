## 算法说明

LLE 是以降维后依旧保持原始高维空间中的局部线性组合关系作为核心

LLE算法要求数据点由其近邻点的线性组合来表示

对于数据点 x1，以最接近 x1 的两个点 x2 和 x3 的线性组合来表示它。

$x_1=w_{12} x_2 + w_{13} x_3$

可以得到两个权值w12，w13.

目的是通过降维后，依旧保持这个线性组合关系，如下图所示

LLE将高维的曲折关系转化为邻近点的组合关系。

![image.png](images/2.png)

对于近邻点数量，是超参数，设定为k个。

步骤：

- 找到数据点 xi 的 k 个近邻点
- 求出由 k 个近邻点线性组合的权值
- 使用权值计算出低维度的 y_i

在确定近邻点数量后，首先，为了求出权重 wij，我们将 xi 和 其近邻点的线性组合 的误差表示为

$x_i - \sum_j w_{ij} x_j$

随着 wij 值的变化，这个误差会增大或减少。

通过计算所有  xi 和线性组合的差的平方和，我们将权重 wij  与误差之间的关系表示为以下误差函数： $e(W) = \sum_i \left| x_i - \sum_j w_{ij} x_j \right|^2$

除了 wij 近邻点之外，表达式中的值都是 0。另外还有一个约束条件：对于某个 i，有

$\sum_j w_{ij} = 1$

我们可以认为权重 wij 表示数据点 xi 及其近邻点之间的关系，而 LLE 通过这种关系在低维空间中也得以保持。在计算完权重后，我们计算低维空间中表示数据点的 y_i：

$\Phi(y) = \sum_i \left| y_i - \sum_j w_{ij} y_j \right|^2$

前面求出使误差最小的权重 wij，现在要做的是利用刚刚求出的 wij，求使误差最小的 y
