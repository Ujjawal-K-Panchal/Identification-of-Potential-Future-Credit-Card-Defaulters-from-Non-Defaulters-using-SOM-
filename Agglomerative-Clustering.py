# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 22:20:02 2018

@author: uchih
"""

#importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset.
dataset = pd.read_csv('preprocessed-cc-data - preprocessed-cc-data.csv')
dataset = dataset.iloc[:,:-2]
dataset.iloc[:,11:24] = dataset.iloc[:,11:24].replace(to_replace = 'nan',value = 'XYZ')
dataset = dataset.replace(to_replace = 'nan',value = 'XYZ')
X = dataset.iloc[:, 1:25].values
Y = dataset.iloc[:,-1].values
Y = np.array(Y, dtype = int)


#Encoding the categorical data.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(X.shape[1]):
    if(type(X[1,i]) == str or type(X[0,i]) == str):
        for j in range(0,X.shape[0]):
            if(type(X[j,i]) == float):
                X[j,i] = "NANX"
        X[:,i] = le.fit_transform(X[:,i])

#Using the Dendrogram to find optimal number of clusters.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Modelling.
from sklearn.cluster import AgglomerativeClustering
ac= AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
Y_hc = ac.fit_predict(X)
Y_hc1 = Y_hc.reshape(Y_hc.shape[0], -1)
clustered_dataset = np.concatenate((X, Y_hc1), axis = 1)

#Putting the names and recreating a dataframe with clustered data.
names = dataset.iloc[:,1:25].columns
names = list(names)
names.append('Cluster')
clustered_dataset = pd.DataFrame(clustered_dataset, columns = names)

#Visualizing Clusters.
plt.scatter(X[Y_hc == 0, 4], X[Y_hc == 0, 9], s = 10, c = 'red', label = 'Cluster 0')
plt.scatter(X[Y_hc == 1, 4], X[Y_hc == 1, 9], s = 10, c = 'green', label = 'Cluster 1')
plt.scatter(X[Y_hc == 2, 4], X[Y_hc == 2, 9], s = 10, c = 'blue', label = 'Cluster 2')
plt.title('Cluster Assignments : '+str(names[4])+' vs. '+str(names[9]))
plt.xlabel(names[4])
plt.ylabel(names[9])
plt.legend()
plt.show()