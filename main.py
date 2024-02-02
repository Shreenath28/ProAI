import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from PIL import ImageTk,Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from tkinter import *

import tkinter
df = pd.read_csv('C:\\Users\\Srinath\\OneDrive\\Others\\Desktop\\bit 2\\dataset.csv')
X = df[['Speed', 'Field Current']].values
y = df['Eg(output)'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X, y)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
