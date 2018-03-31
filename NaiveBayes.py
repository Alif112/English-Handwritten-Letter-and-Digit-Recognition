import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import pandas as pd

from sklearn import metrics
from sklearn.metrics import accuracy_score

# test = pd.read_csv('test.csv')

# train = pd.read_csv('train.csv')

# y_train = train['label']
# X_train= train.drop('label', axis=1)

# y_test = test['label']
# X_test= test.drop('label', axis=1)

import numpy

data = numpy.loadtxt('main.csv' , delimiter = ',')

print(data.shape , type(data))

X = data[:, 0:784]
y = data[:, 784]
print(X.shape , y.shape)

alpha_X = []
alpha_y = []

dig_X = []
dig_y = []

for i in range(y.shape[0]):
	label = y[i]
	if(label > 25):
		dig_X.append(X[i])
		dig_y.append(y[i])
	else:
		alpha_X.append(X[i])
		alpha_y.append(y[i])


y_unique = []
for i in range(y.shape[0]):
	if(y[i] not in y_unique):
		y_unique.append(y[i])

print(y_unique)

# for i in range(X.shape[0]):
# 	for j in range(X.shape[1]):
# 		if(X[i][j] == 0):
# 			X[i][j] = 1
# 		else:
# 			X[i][j] = 0

X_train, X_test, y_train, y_test = train_test_split(dig_X, dig_y, test_size=0.2, random_state=1)

print('data loaded')


print("============Result========Naive Bayes=====")


# MOdel =========
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Digit Accuracy : ", str(accuracy_score(y_test, y_pred)*100.0)+"%")

X_train, X_test, y_train, y_test = train_test_split(alpha_X, alpha_y, test_size=0.2, random_state=1)

print('data loaded')


print("============Result========Naive Bayes=====")


# MOdel =========
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Alphabet Accuracy : ", str(accuracy_score(y_test, y_pred)*100.0)+"%")