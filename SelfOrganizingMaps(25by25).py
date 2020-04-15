# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:31:23 2018

@author: Ujjawal.K.Panchal
"""

#importing the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset and preprocessing.
dataset = pd.read_csv('preprocessed-cc-data - preprocessed-cc-data.csv' ,na_values = '?')
dataset = dataset.iloc[:,:-2]
dataset.iloc[:,10:23] = dataset.iloc[:,10:23].replace(to_replace = 'nan',value = 'XYZ')
dataset = dataset.replace(to_replace = 'nan',value = 'XYZ')
dataset['index'] = [i for i in range(len(dataset))]
X = dataset.iloc[:, 1:24].values
Y = dataset.iloc[:,-2].values
Y = np.array(Y, dtype = int)
X = np.concatenate( (X,dataset.iloc[:,-1:].values) , axis = 1 )

#Encoding the categorical data 
print('Encoding the Categorical Data')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(X.shape[1]):
    if(type(X[1,i]) == str or type(X[0,i]) == str):
        for j in range(0,X.shape[0]):
            if(type(X[j,i]) == float):
                X[j,i] = "NANX"
        X[:,i] = le.fit_transform(X[:,i])
        #print(dataset.iloc[:,1:24].columns[i])
        #for index in range(len(le.classes_)):
         #   print(' ',le.classes_[index],':',index, end = ' , ' )
       # print()


#Feature Scaling.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()  
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test  = train_test_split(X,test_size = 0.2, random_state = 0)
X_cv , X_test = train_test_split(X,test_size = 0.2, random_state = 0)

#MINI-SOM Implementation from : testpypi.python.org/pypi/MiniSom/1.0
from minisom import MiniSom
import pickle
n_neurons = int(np.round( np.sqrt( 5 * np.sqrt(X.shape[0]) ) ) ) # Calculated using n_neurons = 5* sqrt(data-points) 24.7 ~ 25

'''
from hyperopt import *
space = { 'sig' : hp.uniform('sig' , 0.01, n_neurons/2.01),
          'learning_rate' : hp.uniform('learning_rate' , 0.01, 5) 
        }

def som_fn(space):
    sig = space['sig']
    learning_rate = space['learning_rate']
    val = MiniSom(                             
            x = n_neurons,
            y = n_neurons,
            input_len= X.shape[1],
            sigma = sig, 
            learning_rate = learning_rate
            ).quantization_error(X_cv)
    print(val)
    return{'loss' : val , 'status' : STATUS_OK}
trials = Trials()
best = fmin(
                    fn = som_fn,
                    space = space,
                     algo = tpe.suggest,   
                    max_evals = 100,
                    trials = trials
                )
print(best)
''' 
#Tuned : {'learning_rate': 0.9669379851870759, 'sig': 12.163584472081842}
som = MiniSom(x = n_neurons , y = n_neurons , input_len = X.shape[1], sigma = 12.163584472081842 , learning_rate =  0.9669379851870759)
#Training.
#som.random_weights_init(X)
#som.train_random(X_train, num_iteration = 1500)
#trained_weights = som.weights
#Storing the SOM weights
#import pickle
#with open('som-model-25by25.p' , 'wb') as outfile:
#    pickle.dump(trained_weights, outfile)

'''getting pre-trained weights.'''
trained_weights = pickle.load(open('som-model-25by25.p' , 'rb'))
som.weights = trained_weights

#pickle.dump(som, open('som-model-25by25.sav', 'wb'), pickle.HIGHEST_PROTOCOL)

#Visualization of the created SOM.
'''
from pylab import bone, pcolor, colorbar, plot, show
%matplotlib qt
plt.title('Untrained Self Organizing Map for Credit Default Analysis')
bone()
pcolor((som.distance_map()).T)
colorbar()


#Plotting Markers.
markers = ['o', 's']
colors = ['r', 'g']

Dict1 = dict() #Making a Dictionary to get how many defaulters/non defaulters in a cluster.
for i in range(n_neurons): #This Dictionary will take all nodes of SOM into account.
    for j in range(n_neurons):
        Dict1[(i,j)] = [0,0] #[no of defaulters, no of non defaulters].

for i,x in enumerate(X):
    w = som.winner(x)
    #Assigning scores to each node of SOM.
    if Y[i] == 1:
        Dict1[ (w[0], w[1]) ] = [ (Dict1[ (w[0], w[1]) ])[0]+1, (Dict1[ (w[0], w[1]) ])[1] ]
    else:
        Dict1[ (w[0], w[1]) ] = [ (Dict1[ (w[0], w[1]) ])[0], (Dict1[ (w[0], w[1]) ])[1]+1 ]
    try:
        plot(w[0]+0.5, w[1]+0.5, markers[Y[i]], markeredgecolor  = colors[Y[i]], 
             markeredgewidth = 2, markerfacecolor = 'None')
    except : 
        continue
show()
'''
#finding the clusters where 90% constitute of defaulters.
def_index = [] #indexes of defaulter clusters with above given criterion.
for i in range(n_neurons):
    for j in range(n_neurons):     #checking if defaulters constitute 90% or greater of a cluster.
        if((Dict1[i,j])[0] >  ((Dict1[i,j])[1] + (Dict1[i,j])[0] )*0.70 ):      #and (som.distance_map())[i,j] < 0.9 ):
            def_index.append((i,j))

mappings = som.win_map(X)
defaulters_all = np.array(mappings[def_index[0]])
def_index = def_index[1:]
for k in def_index:
    defaulters_all = np.concatenate((defaulters_all, np.array(mappings[k])) , axis = 0)

#Getting actual column values of all points in the defaulter cluster. 
defaulters_all = sc.inverse_transform(defaulters_all)

#Getting new most probable defaulters using our SOM.
defaulters_probable = []
for i in range(len(defaulters_all)):
    if(Y[ int( defaulters_all[i,-1] ) ] == 0):
        defaulters_probable.append(defaulters_all[i,-1])
        
for i in range(len(defaulters_probable)):
    defaulters_probable[i] = int(defaulters_probable[i])
