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
#from sklearn.preprocessing import Imputer.
#numeric_imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#X[:,25:] = numeric_imputer.fit_transform(X[:,25:])

#Encoding the categorical data.
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


#Selecting the best Features.
'''
from sklearn.feature_selection import SelectPercentile,f_classif
selector= SelectPercentile(percentile = 90, score_func= f_classif)
selector.fit(X_train, Y_train)
X_train, X_val, X_test = selector.transform(X_train), selector.transform(X_val), selector.transform(X_test)
'''
#Hyperparam : Tuning.
'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 300, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth = [x for x in range(1,9)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

#Criterion
criterions = ['gini', 'entropy']

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterions
               }

from sklearn.model_selection import RandomizedSearchCV


clf_random = RandomizedSearchCV(estimator = RandomForestClassifier(n_jobs  = -1), 
                                param_distributions = random_grid, 
                                n_iter = 100, cv = 3, verbose=10, random_state=42, n_jobs = 1) 
clf_random.fit(X_train,Y_train)
print(clf_random.best_params_)
'''
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators= 300,
 min_samples_split= 5,
 min_samples_leaf= 4,
 max_features= 'log2',
 max_depth= None,
 bootstrap= False, 
 criterion = 'gini', random_state = 42)

clf.fit(X_train, Y_train)

print('Cross Validation Set Evaluation : ')

Y_pred = clf.predict(X_val)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy : ',accuracy_score(Y_val,Y_pred)*100,'%')
val_acc = accuracy_score(Y_val,Y_pred)*100
print ('Confusion Matrix')
print(pd.crosstab(Y_val,Y_pred, margins = True))

print()
print('Test Set Evaluation : ')

Y_pred = clf.predict(X_test)

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
from sklearn.tree import export_graphviz
from graphviz import Source
import os
os.chdir('RFTREES-IMGS')
dot_data = StringIO()
for i in range (len(clf.estimators_[:])):
    dot_data = StringIO()
    export_graphviz(clf.estimators_[i], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
    col_names =  dataset.iloc[:, 1:25].columns
    new_str = dot_data.getvalue().replace('<X<SUB>'+str(0)+'</SUB>' , '<'+col_names[0]+'<SUB>'+str(0)+'</SUB>')
    for j in range(1,len(col_names)):
        new_str = new_str.replace('X<SUB>'+str(j)+'</SUB>' , col_names[j])  
    img = Source(new_str)
    img.render('tree'+str(i))
''' 