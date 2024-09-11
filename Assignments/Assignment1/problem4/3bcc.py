import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

import pandas as pd
from sklearn import datasets

def create_polynomial_features(X):
    """
    Create polynomial features for the input matrix X.
    This includes x1, x2, x1*x2, x1^2, and x2^2.
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    X_poly = np.vstack([
        np.ones(x1.shape),
        x1,
        x2,
        x1*x2,
        x1**2,
        x2**2
    ]).T
    return X_poly

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
# X_train = X.iloc[:-20].loc[:, ['bmi', 'bp']].values
# y_train = y.iloc[:-20] / 300
X_train = X.iloc[:].loc[:, ['bmi', 'bp']].values
y_train = y.iloc[:] / 300

# Create polynomial features
X_train_poly = create_polynomial_features(X_train)

# Compute theta using the least squares solution
X_train_poly_transpose = X_train_poly.T
# theta = np.linalg.inv(X_train_poly_transpose.dot(X_train_poly)).dot(X_train_poly_transpose).dot(y_train)

theta = np.linalg.solve(np.matmul(X_train_poly.T, X_train_poly) + .1*np.eye(X_train_poly.shape[1]), np.matmul(X_train_poly.T, y_train))

print(f"Parameters obtained from least squares: {theta}")

# Predict y values
y_pred = X_train_poly.dot(theta)

# Compute Mean Squared Error
mse = np.mean((y_train - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

 
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
A = X_train_poly_transpose @ X_train_poly
print(is_pos_def(A))
print(np.linalg.cholesky(A))
