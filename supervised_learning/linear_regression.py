# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting simple linear regresstion to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X=X_train,y=y_train)

#predicting the results
y_pred=regressor.predict(X=X_test)

#visualize the results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.scatter(X_test,y_test,color='green')
plt.title('salary vs experience')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

