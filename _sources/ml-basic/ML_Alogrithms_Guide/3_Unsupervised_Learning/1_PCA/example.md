## ʾ������

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
data = load_iris()
n_components = 2  #  �����ٺ��ά������Ϊ2
model = PCA(n_components=n_components)
model = model.fit(data.data)
print(model.transform(data.data))  # �任�������
```
