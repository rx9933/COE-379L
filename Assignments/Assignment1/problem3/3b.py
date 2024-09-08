import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

import pandas as pd
from sklearn import datasets

def f(theta, X):
    # [your work!]
    return 0.4638 + theta[0] * X.iloc[:, 0] + theta[1] * X.iloc[:, 1]

def mean_squared_error(theta, X, y):
    # [your work!]
    y_pred = f(theta, X)
    return 1 / (2 * len(y)) * np.sum((y - y_pred) ** 2) 

def mse_gradient(theta, X, y):
    # [your work!]
    print("a",(f(theta, X)).shape)
    print("b",X.T.shape)
    print(((f(theta, X) - y) * X.T).shape)
    print(np.sum((f(theta, X) - y) * X.T, axis=1).values.shape)
    return np.sum((f(theta, X) - y) * X.T, axis=1).values # vs np.mean

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300
X = np.ones(((X_train.shape)[0], 6))
x1, x2 = X_train.values[:,0], X_train.values[:,1]
X[:, 1], X[:, 2], X[:, 3], X[:, 4],X[:, 5] = x1, x2, x1*x2,x1**2, x2**2, 


print(X.shape)
theta = np.linalg.inv(X.T@X) @ X.T @ y_train
print(theta)
"""
[  0.33898498   3.29212551   0.41478228 -21.41000838  29.55079269
  37.96660731]

X.T X THETA - x.t y = 0
theta = np.linalg.inv(X.T@X) @X.T @ y
X.T:
11
x1
x2
x1x2
x1**2
x2**2
"""