import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics

mdf = pd.read_csv("D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\masterDF.csv")
print(mdf.head(20))

X = mdf[['HPCP', 'STAT_BIOME']]
y = mdf[['FIRE']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
# 70% training and 30% test_size

# Support vector machines
svmMchn = svm.SVC()

# Gaussian Classifier
clf = RandomForestClassifier(n_estimators=1000)

# Logistic Regression Classifier
logisticRegr = LogisticRegression()

#Training the model
svmMchn.fit(X_train,y_train)
clf.fit(X_train, y_train)
logisticRegr.fit(X_train, y_train)

svm_pred = svmMchn.predict(X_test)
y_pred = clf.predict(X_test)
logistic_pred = logisticRegr.predict(X_test)

print("Training has ended")
# feature importance
# feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
# print(feature_imp)
print("SVM Accuracy:",
metrics.accuracy_score(y_test, svm_pred))
print("RF Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("LReg Accuracy:", metrics.accuracy_score(y_test, logistic_pred))
print("Predict this: ")

print(svmMchn.predict([[0.1, 12]]))
print(clf.predict([[0.1, 12]]))
print(logisticRegr.predict([[0.1, 12]]))
