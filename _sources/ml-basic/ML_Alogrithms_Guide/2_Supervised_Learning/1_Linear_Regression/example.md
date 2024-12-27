## ʾ������

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
X = [[10.0], [8.0], [13.0], [9.0], [11.0], [14.0], [6.0], [4.0], [12.0], [7.0], [5.0]]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
# ����ѵ�����Ͳ��Լ���test_size ָ�����Լ��ı���
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.intercept_) # �ؾ�
print(model.coef_) # б��
y_pred = model.predict([[0], [1]])
print(y_pred) # ��x=0, x=1��Ԥ����
# �������ݼ�ɢ��ͼ
plt.scatter(X_train, y_train, color='blue', label='Data')
# ���ƻع���
y_pred = model.predict(X_train)
plt.plot(X_train, y_pred, color='red', label='Linear Regression Model')
plt.show()
```

![1.png](images/1.png)
