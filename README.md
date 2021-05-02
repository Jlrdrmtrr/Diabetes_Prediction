# Diabetes_Prediction

A program that could predict diabetes using the SciKit Learn Diabetes dataset.

from sklearn import datasets

diabetes = datasets.load_diabetes()

print(diabetes.feature_names)

X = diabetes.data
Y = diabetes.target
X.shape, Y.shape

X, Y = datasets.load_diabetes(return_X_y=True)
X.shape, Y.shape
import pandas as pd

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train.shape, Y_train.shape
X_test.shape, Y_test.shape

#Linear Regression

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#Prediction Result

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))

print(diabetes.feature_names)

#Scatter Plot

Y_test

import numpy as np
np.array(Y_test)

Y_pred

sns.scatterplot(Y_test, Y_pred)

sns.scatterplot(Y_test, Y_pred, marker="+")

sns.scatterplot(Y_test, Y_pred, alpha=0.5)
