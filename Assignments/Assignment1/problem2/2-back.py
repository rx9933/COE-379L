import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

import pandas as pd
from sklearn import datasets

def f(theta, X):
    return (theta[0] 
            + theta[1] * X.iloc[:, 0] 
            + theta[2] * X.iloc[:, 1] 
            + theta[3] * X.iloc[:, 0] * X.iloc[:, 1] 
            + theta[4] * X.iloc[:, 0]**2 
            + theta[5] * X.iloc[:, 1]**2)

def mean_squared_error(theta, X, y):
    y_pred = f(theta, X)
    return 1 / (2 * len(y)) * np.sum((y - y_pred) ** 2) 

def mse_gradient(theta, X, y):
    y_pred = f(theta, X)
    error = y_pred - y
 
    gradients = np.zeros(theta.shape)

    gradients[0] = np.mean(error)
    gradients[1] = np.mean(error * X.iloc[:, 0])
    gradients[2] = np.mean(error * X.iloc[:, 1])
    gradients[3] = np.mean(error * X.iloc[:, 0] * X.iloc[:, 1])
    gradients[4] = np.mean(error * X.iloc[:, 0]**2)
    gradients[5] = np.mean(error * X.iloc[:, 1]**2)

    return gradients

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

tolerance = 1e-6
step_size = 4e-1 
theta, theta_prev = np.array([0,0,0,0,0,0]), np.array([1,1,1,1,1,1])
iter = 0

while np.linalg.norm(theta - theta_prev) > tolerance:
    if iter > 200000:
        break
    theta_prev = theta
    gradient = mse_gradient(theta, X_train, y_train)
    theta = theta_prev - step_size * gradient
    iter += 1
print("Optimal Theta:", theta)
print("Iteration:", iter)
print(f"Final Mean Squared Error: {mean_squared_error(theta, X_train, y_train):.4f}")
print(f"Residual: {np.linalg.norm(theta - theta_prev):.4f}")