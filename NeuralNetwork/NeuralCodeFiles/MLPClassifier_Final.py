import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

X = dataset.iloc[:, 1:-1].values
y = labels.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=25)

clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500,
                    solver='adam', verbose=True,  random_state=20, tol=0.000000001)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

detail = classification_report(y_test, y_pred, output_dict=True)
result = classification_report(y_test, y_pred)

print("\n Classification Report:")
print(result)

print("\n Breast Cancer:")
print("recall : ", detail['breast cancer']['recall'] , "precision : " 
, detail['breast cancer']['precision'] ,"f2-score : "
, ((5*detail['breast cancer']['recall']*detail['breast cancer']['precision'])/(4*detail['breast cancer']['precision']+detail['breast cancer']['recall'])))

print("\n Colon Cancer:")
print("recall : ", detail['colon cancer']['recall'] , "precision : " 
, detail['colon cancer']['precision'] ,"f2-score : "
, ((5*detail['colon cancer']['recall']*detail['colon cancer']['precision'])/(4*detail['colon cancer']['precision']+detail['colon cancer']['recall'])))

print("\n Lung Cancer:")
print("recall : ", detail['lung cancer']['recall'] , "precision : " 
, detail['lung cancer']['precision'] ,"f2-score : "
, ((5*detail['lung cancer']['recall']*detail['lung cancer']['precision'])/(4*detail['lung cancer']['precision']+detail['lung cancer']['recall'])))

print("\n Prostate Cancer:")
print("recall : ", detail['prosrtate cancer']['recall'] , "precision : " 
, detail['prosrtate cancer']['precision'] ,"f2-score : "
, ((5*detail['prosrtate cancer']['recall']*detail['prosrtate cancer']['precision'])/(4*detail['prosrtate cancer']['precision']+detail['prosrtate cancer']['recall'])))

print("\n")
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2, "\n")

