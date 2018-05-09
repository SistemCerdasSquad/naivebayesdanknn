
# coding: utf-8

# # Load Data Set

# In[ ]:

import pandas as pd
df = pd.read_csv('dataset.csv')


# # Save DataSet into Data Structure

# In[ ]:

X = []
y = []
for (name,mcg,gvh,lip,chg,acc,alm1,alm2,label) in zip(df['sequence_name'],df['mcg'],df['gvh'],df['lip'],df['chg'],df['aac'],df['alm1'],df['alm2'],df['label']):
    X.append([mcg,gvh,lip,chg,acc,alm1,alm2])
    y.append(label)


# # Membagi Data Testing dan Data Training

# In[ ]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Training Model

# In[ ]:

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

cls = GaussianNB()
cls.fit(X_train, y_train)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)


# # Testing Model

# In[ ]:

for test in X_test:
    result = cls.predict([test])
    print("nilai {nilai} memiliki label {label}".format(nilai=test,label=result[0]))


# In[ ]:

for test in X_test:
    result = clf.predict([test])
    print("nilai {nilai} memiliki label {label}".format(nilai=test,label=result[0]))


# # Akurasi

# In[ ]:

print("Akurasi Prediksi : ",cls.score(X_test,y_test))


# In[ ]:

print("Akurasi Prediksi : ",clf.score(X_test,y_test))

