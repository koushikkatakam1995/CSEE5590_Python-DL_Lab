import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing our cancer.csv dataset
dataset = pd.read_csv('cancer.csv')
X = dataset.iloc[:, 1:9].values
Y = dataset.iloc[:, 10].values
dataset.head()
print("Cancer data set dimensions : {}".format(dataset.shape))
dataset.isnull().sum()
print(dataset.isna().sum())

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

'''#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

# NB
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train, y_train)
print("\n----------GNB----------")
# GaussianNB(priors=None, var_smoothing=1e-09)
print('Accuracy of Naive Bayes GaussianNB on training set: {:.2f}'.format(GNB.score(X_train, y_train)))
# Evaluate the model on testing part
print('Accuracy of Naive Bayes GaussianNB on test set: {:.2f}'.format(GNB.score(X_test, y_test)))

# SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print("\n----------SVM----------")
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# test data set acc
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print("\n----------KNN----------")
print('Accuracy of KNN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
# test data set acc
print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))