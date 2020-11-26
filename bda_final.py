import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('C:/pythondata/train.csv')
test = pd.read_csv('C:/pythondata/test.csv')

X_train = train['text']
y_train = train['sentiment']
X_test = test['text']
y_test = test['sentiment']

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stop_words)
X_train_seq = vectorizer.fit_transform(X_train)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=123)
clf.fit(X_train_seq, y_train)

X_test_seq = vectorizer.transform(X_test)
y_pred = clf.predict_proba(X_test_seq)[:,1]

from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(y_test=='pos', y_pred)
plt.plot(fpr,tpr, label='AUC = %0.3f' %metrics.auc(fpr,tpr))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
lab = 'asdfasd'
plt.legend(loc='lower right')