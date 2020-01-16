#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:14:48 2019

@author: kiran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset with pandas
data_set=pd.read_csv('Mall_Customers.csv')
X=data_set.iloc[:,[3,4]].values

# dendogram to take optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.ylabel('eucledian distance')
plt.xlabel('customers')
plt.show()


#fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#plotting the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='cluster1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='green',label='cluster2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='blue',label='cluster3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='cluster4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='cluster5')

plt.xlabel('clusters')
plt.ylabel('annual income')
plt.legend()
plt.title('clustering of customers')
plt.show()
