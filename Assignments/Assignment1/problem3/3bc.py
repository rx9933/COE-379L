import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

from sklearn import datasets

def f(theta, X):
    return 0.4638 + theta[0] * X[:, 1] + theta[1] * X[:, 2] + theta[2] * X[:, 3] + theta[3] * X[:, 4] + theta[4] * X[:, 5]

def mean_squared_error(theta, X, y):
    y_pred = f(theta, X)
    return 1 / (2 * len(y)) * np.sum((y - y_pred) ** 2) 

def mse_gradient(theta, X, y):
    y_pred = f(theta, X)
    error = y_pred - y
    gradient = (X.T @ error) / len(y)
    return gradient

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

# Design matrix X (including interaction terms and quadratic terms)
X_design = np.ones((X_train.shape[0], 6))  # 6 columns: [1, x1, x2, x1*x2, x1^2, x2^2]
x1, x2 = X_train.values[:, 0], X_train.values[:, 1]
X_design[:, 1], X_design[:, 2], X_design[:, 3], X_design[:, 4], X_design[:, 5] = x1, x2, x1 * x2, x1**2, x2**2

# Perform least squares estimation
theta = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y_train)
print("Theta:", theta)

# Check the mean squared error
mse = mean_squared_error(theta, X_design, y_train)
print("Mean Squared Error:", mse)
print("MSE gradient",mse_gradient(theta, X_design, y_train))
"""[ 3.58727465e-02 -3.30362548e-03  1.07121871e-03 -1.62327841e-04
 -1.32860642e-04 -6.75559160e-05]"""