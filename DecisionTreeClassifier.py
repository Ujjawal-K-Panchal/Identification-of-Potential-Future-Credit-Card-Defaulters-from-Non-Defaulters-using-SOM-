# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 12:31:51 2018

@author: Uchiha Madara
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

#Classification using Logistic regression.
#from sklearn.linear_model import LogisticRegression

#Hyperparam : Tuning.

from sklearn.tree import DecisionTreeClassifier

crits = ['gini', 'entropy']
max_depths = [15,12,10, 18, 20]
random_grid = {
        'criterion' : crits,
        'max_depth' : max_depths
        }
from sklearn.model_selection import RandomizedSearchCV
clf_random = RandomizedSearchCV(estimator = DecisionTreeClassifier(), 
                                param_distributions = random_grid, 
                                cv = 3, verbose=10, random_state=42, n_jobs = 1)
clf_random = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5)
clf_random.fit(X_train,Y_train)

#Found gini and 15 depth to be the best parameters.

lr = DecisionTreeClassifier(criterion = 'gini', max_depth = 10)
lr.fit(X_train, Y_train)

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

'''
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from graphviz import Source
dot_data = StringIO()
export_graphviz(lr, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
col_names =  dataset.iloc[:, 1:25].columns
new_str = dot_data.getvalue().replace('<X<SUB>'+str(0)+'</SUB>' , '<'+col_names[0]+'<SUB>'+str(0)+'</SUB>')

for i in range(1,len(col_names)):
        new_str = new_str.replace('X<SUB>'+str(i)+'</SUB>' , col_names[i])
img = Source(new_str)
img.render('OptimalTree')
'''
#f = open('dtree.txt', 'wt')
#f.write(new_str)
#f.close()