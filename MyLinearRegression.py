"""
Created on Wed Nov 13 22:42:53 2019

@author: supreeth
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:05:56 2019

@author: supreeth
"""

#importing the standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading dataset and seperating dependent and independent variables.
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Seperating test data and train data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#FeatureScaling
#from sklearn.preprocessing import StandardScaler
#scx = StandardScaler()
#X_train = scx.fit_transform(X_train)
#X_test = scx.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

#plt.scatter(X_train, Y_train, color = 'red')
#plt.plot(X_train, regressor.predict(X_train),color = 'blue')
#
#plt.title('Salary vs Experience (Training set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary($)')
#plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')

plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary($)')
plt.show()





