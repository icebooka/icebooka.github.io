## ʾ������

```python
from sklearn.datasets import samples_generator
from sklearn.manifold import LocallyLinearEmbedding
data, color = samples_generator.make_swiss_roll(n_samples=1500)
n_neighbors = 12  # ���ڵ������
n_components = 2  # ��ά���ά��
model = LocallyLinearEmbedding(n_neighbors=n_neighbors,
n_components=n_components)
model.fit(data)
print(model.transform(data))  # �任�����
```
