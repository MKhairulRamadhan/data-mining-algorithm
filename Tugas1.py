# 
# 03-04-2020
# 	M Khairul Ramadhan
# 	1708107010006
# 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

#load dataset
bankdata = pd.read_csv("./bill_auth.csv")

#Menentukan variabel independen & dependen
x = bankdata.iloc[:, [2, 3]].values
y = bankdata.iloc[:, 4].values

#Menentukan data training(80%) dan testing(20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Memanggil fungsi klasifikasi Naive Bayes
model = GaussianNB()

#membuatkan model klasifikasi dari data training
model.fit(x_train, y_train)

#prediksi hasil test set
y_pred = model.predict(x_test)

#Memebuat Confusin Matrix
cm = confusion_matrix(y_test, y_pred)

print ("Hasil Confusion Matrix:\n",cm)
print("\nHasil Klasifikasi:\n",classification_report(y_test,y_pred))
