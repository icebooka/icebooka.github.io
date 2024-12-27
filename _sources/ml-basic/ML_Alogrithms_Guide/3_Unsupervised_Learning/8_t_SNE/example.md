## ʾ������

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
data = load_digits()
n_components = 2  # ���ý�ά���ά��Ϊ2
model = TSNE(n_components=n_components)
print(model.fit_transform(data.data))
```
