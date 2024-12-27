## ʾ������

```python
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
data = load_iris()
n_components = 3  # ��˹�ֲ�������
model = GaussianMixture(n_components=n_components)
model.fit(data.data)
print(model.predict(data.data))  # Ԥ�����
print(model.means_)  # ����˹�ֲ��ľ�ֵ
print(model.covariances_)  # ����˹�ֲ��ķ���
```
