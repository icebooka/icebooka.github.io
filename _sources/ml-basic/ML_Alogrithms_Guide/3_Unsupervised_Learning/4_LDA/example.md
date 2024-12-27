## ʾ������

``` python

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# ʹ��removeȥ�������������Ϣ
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
max_features = 1000
# ���ı����ݱ任Ϊ����
tf_vectorizer = CountVectorizer(max_features=max_features,
stop_words='english')
tf = tf_vectorizer.fit_transform(data.data)
n_topics = 20
model = LatentDirichletAllocation(n_components=n_topics)
model.fit(tf)                                
print(model.components_)  # ����������ĵ��ʵķֲ�
print(model.transform(tf))  # ʹ�������������ı�

```

ʹ��scikit-learnʵ�ֻ���LDA������ģ�͵Ĵ�����ʹ����һ����Ϊ20 Newsgroups��
���ݼ���������ݼ���20��������������ı��ļ��ϣ�ÿ���ı�����һ�����⡣
