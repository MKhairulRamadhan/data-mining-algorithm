import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

bankdata = pd.read_csv("bill_authentication.csv")

# print(bankdata)

x = bankdata.drop('Class', axis=1)
y = bankdata['Class']

print(x)
print(y)

X_train, X_test , y_train, y_test = train_test_split(x, y, test_size = 0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))