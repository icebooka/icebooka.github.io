## ʾ������

�����Ƕ��β�����ݼ�Ӧ��k-means�㷨�Ĵ��롣

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
data = load_iris()
n_clusters = 3  # ���ص���������Ϊ3
model = KMeans(n_clusters=n_clusters)
model.fit(data.data)
print(model.labels_)  # �����ݵ������Ĵ�
print(model.cluster_centers_)  # ͨ��fit()����õ�������
```
