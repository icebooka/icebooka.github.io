## ʾ������

����һ��ʹ��8��������=���ʵĸ�������ʾ�����ݼ�������2��Ǳ�ڱ���ȥ��ʾ����

```python
from sklearn.decomposition import TruncatedSVD
data = [[1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]]
n_components = 2  # Ǳ�ڱ����ĸ���
model = TruncatedSVD(n_components=n_components)
model.fit(data)
print(model.transform(data))  # �任�������
print(model.explained_variance_ratio_)  # ������
print(sum(model.explained_variance_ratio_))  # �ۼƹ�����
```

���⣬��PCAһ��������Ҳ���Լ��LSA�任��ľ����а�������ԭʼ��Ϣ��ʹ����scikit-learn����������������ۼƹ�����ԼΪ0.67��������2������������Լ67%��ԭʼ���ݵ���Ϣ��
