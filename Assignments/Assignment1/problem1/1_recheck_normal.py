import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

import pandas as pd
from sklearn import datasets

def f(theta, X):
    # [your work!]
    return 0.4638 + theta[0] * X.iloc[:, 0] + theta[1] * X.iloc[:, 1]


# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)

# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

# Add bias term (column of ones) to X
X_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train.values])

# Compute theta using the normal equation
theta_normal_eq = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y_train
print("Theta (Normal Equations with bias):", theta_normal_eq)


X = X_train.values
y = y_train.values
print(np.linalg.inv(X.T @ X))
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(theta)