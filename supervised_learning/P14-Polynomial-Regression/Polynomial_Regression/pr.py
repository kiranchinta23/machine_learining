#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:49:33 2019

@author: kiran
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fit linear regression model
from sklearn.linear_model import LinearRegression
regressor_l=LinearRegression()
regressor_l.fit(X,y)

#fit polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)

regressor_l2=LinearRegression()
regressor_l2.fit(X_poly,y)

#visualize the linear regression results
plt.scatter(X,y,color='red')
plt.plot(X,regressor_l.predict(X))
plt.title('linear regression model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualize the polynomial regression
plt.scatter(X,y,color='red')
plt.plot(X,regressor_l2.predict(X_poly))
plt.title('polynomial regression model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


# predicting linear regression model
regressor_l.predict(6.5)

#predicting polynomial regression model
regressor_l2.predict(poly.fit_transform(6.5))
