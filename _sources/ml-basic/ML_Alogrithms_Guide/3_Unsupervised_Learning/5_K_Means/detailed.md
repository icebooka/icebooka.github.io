## 详细说明

### k-means的评估

用于评估分类的结果好坏，可以使用WCSS（簇内平方和）。也就是每个簇内的所有数据到重心距离的平方和相加的值。

WCSS越小，说明聚类结果越好。

### Elbow方法确定簇数量

对于簇数量，是作为输入的超参数，也就是我们算法名称中的k，有些数据集很难定义合适的簇数量。

我们已经知道可以使用WCSS来确定优劣，因此对于不同的簇数量，我们都可以得到一个WCSS。

![image.png](images/3.png)

纵轴是WCSS，横轴是k，可以发现，随着k的变大，WCSS会递减，但是我们可以找到一个点 3，在这个点后，WCSS的变化值显著减少，也就是选择这个点是最适合的点，这个图像类似于一个肘部，因此称作肘方法。

注：

当没有很明确的理由来确定簇的数量或对数据知之甚少时，Elbow方法很有用。不过在实际的分析中，常常不会出现图中那样明显的“肘部”，所以Elbow方法的结果只不过是一种参考而已。