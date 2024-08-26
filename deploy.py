# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 07:33:47 2023

@author: Indra
"""
#import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler

import pickle
#load the csv file
data = pd.read_csv('_placement_dataset.csv')
print(data)
print(data.dtypes)
print(data.head())
print(data.describe())
data.isnull().sum()
#independent and dependent
x=data[["SSC Percentage","HSC Percentage","pg","ug","certification course"]]
y=data["Placed"]
le = LabelEncoder()
data['Placed'] = le.fit_transform(data['Placed'])
#select independent and dependent variable
X = data.drop('Placed', axis=1)
y = data['Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
print(X.shape,X_train.shape,X_test.shape)
#sc= StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test=sc.transform(X_test)
classifier = LogisticRegression()
# fit the model;
classifier.fit(X_train, y_train)
'''y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
new_student=[[45,55,56,60,2]]

prediction=model.predict(new_student)
print(prediction)
if prediction[0] == 1:
    print("***you are placed***")
else:
    print("***you are not placed***")'''
    # make pickle file of our model
pickle.dump(classifier,open("model.pkl","wb"))
    
    
