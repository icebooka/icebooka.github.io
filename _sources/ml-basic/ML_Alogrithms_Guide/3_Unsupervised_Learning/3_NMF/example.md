## ʾ������

```python
from sklearn.decomposition import NMF
from sklearn.datasets.samples_generator import make_blobs
centers = [[5, 10, 5], [10, 4, 10], [6, 8, 8]]
V, _ = make_blobs(centers=centers)  # ��centersΪ������������
n_components = 2  # Ǳ�ڱ����ĸ���
model = NMF(n_components=n_components)
model.fit(V)
W = model.transform(V) # �ֽ��ľ���
H = model.components_
print(W)
print(H)
```
