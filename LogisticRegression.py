# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:32:41 2018

@author: Ujjawal Panchal
"""
#Importing the necessary libraries.
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('preprocessed-cc-data - preprocessed-cc-data.csv' ,na_values = '?')
dataset = dataset.iloc[:,:-2]
dataset.iloc[:,10:23] = dataset.iloc[:,10:23].replace(to_replace = 'nan',value = 'XYZ')
dataset = dataset.replace(to_replace = 'nan',value = 'XYZ')
X = dataset.iloc[:, 1:24].values
Y = dataset.iloc[:,-1].values
Y = np.array(Y, dtype = int)
# Taking care of missing data of numeric types.
#from sklearn.preprocessing import Imputer
#
#numeric_imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#X[:,25:] = numeric_imputer.fit_transform(X[:,25:])

#Encoding the categorical data 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(X.shape[1]):
    if(type(X[1,i]) == str or type(X[0,i]) == str):
        for j in range(0,X.shape[0]):
            if(type(X[j,i]) == float):
                X[j,i] = "NANX"
        X[:,i] = le.fit_transform(X[:,i])
        print(dataset.iloc[:,1:24].columns[i])
        for index in range(len(le.classes_)):
            print(' ',le.classes_[index],':',index, end = ' , ' )
        print()
       
#Data Splitting.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0, test_size = 0.2)
X_val,X_test,Y_val,Y_test = train_test_split(X_test,Y_test,random_state = 0, test_size = 0.5)

# Taking care of missing data of numeric types.
#from sklearn.preprocessing import Imputer
#
#numeric_imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#X[:,25:] = numeric_imputer.fit_transform(X[:,25:])


#Classification using Logistic regression.
from sklearn.linear_model import LogisticRegression

#Hyperparam : Tuning.
"""Hyperparams"""
'''
C_list = [0.001,0.03,0.01,0.1, 0.3, 1] #4: add 0.001
solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']#5
max_iter_list = [500]#5 add : 200,250
random_grid = {
        'C':C_list,
        'solver':solver_list,
        'max_iter':max_iter_list
        }

"""/Hyperparams"""

from sklearn.model_selection import RandomizedSearchCV
clf_random = RandomizedSearchCV(estimator = LogisticRegression(n_jobs  = 3), 
                                param_distributions = random_grid, 
                                cv = 3, verbose=10, random_state=42, n_jobs = 1) 
clf_random.fit(X_val,Y_val)
print(clf_random.best_params_)

#Best Parameters : {'solver': 'liblinear', 'max_iter': 500, 'C': 0.1}
'''
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000, C = 1) 
lr.fit(X_train,Y_train)
#Prediction and Evaluation

print('Cross Validation Set Evaluation : ')

Y_pred = lr.predict(X_val)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy : ',accuracy_score(Y_val,Y_pred)*100,'%')
val_acc = accuracy_score(Y_val,Y_pred)*100
print ('Confusion Matrix')
print(pd.crosstab(Y_val,Y_pred, margins = True))

print()
print('Test Set Evaluation : ')

Y_pred = lr.predict(X_test)

print('Accuracy : ',accuracy_score(Y_test,Y_pred)*100,'%')
test_acc = accuracy_score(Y_test,Y_pred)*100

print ('Confusion Matrix')
print(pd.crosstab(Y_test,Y_pred, margins = True))

print('\nConclusive Accuracy :', ((val_acc+test_acc)/2),'%')

print('Precision Score : ', precision_score(Y_test, Y_pred))
print('Recall Score : ', recall_score(Y_test, Y_pred))
print('F1 Score : ', f1_score(Y_test, Y_pred))


