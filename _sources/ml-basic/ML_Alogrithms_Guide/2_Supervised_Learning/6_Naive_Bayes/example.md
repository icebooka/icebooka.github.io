## ʾ������

```python
from sklearn.naive_bayes import MultinomialNB
# ��������
X_train = [[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]]
y_train = [1, 1, 1, 0, 0, 0]
model = MultinomialNB()
# ѵ��
model.fit(X_train, y_train)
# Ԥ��������
y_pred = model.predict(X_train)
y_pred_prob = model.predict_proba(X_train)
model.predict([[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
```
