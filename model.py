import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv('pollutant-standards-index-southtangerang-2020-2022.csv')
print(df)

#Karena variabel O3 memiliki positive skewd, maka data kosong di isikan dengan median
fill=df["O3"].median()
df["O3"]=df["O3"].fillna(fill)

#Merubah tipe data O3
df["O3"]=df["O3"].astype("int64")

class_feature = ['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'NO2']
x = df[class_feature]
y = df['Category']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

print("Training Accuracy :",model.score(X_train,y_train))
print("Testing Accuracy :",model.score(X_test,y_test))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Akurasi:', accuracy)

classification_report = classification_report(y_test, y_pred)
print('Laporan Klasifikasi:\n', classification_report)

#Simpan Model

pickle.dump(model,open("model.pkl","wb"))

print(x)