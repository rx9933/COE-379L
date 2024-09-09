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
        x1 * x2,
        x1 ** 2,
        x2 ** 2
    ]).T
    return X_poly

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']].values
y_train = y.iloc[-20:] / 300

# Create polynomial features
X_train_poly = create_polynomial_features(X_train)

# Compute theta using the least squares solution
X_train_poly_transpose = X_train_poly.T
# theta = np.linalg.inv(X_train_poly_transpose.dot(X_train_poly)).dot(X_train_poly_transpose).dot(y_train)
theta = np.linalg.solve(np.matmul(X_train_poly.T, X_train), np.matmul(X_train_poly.T, y_train))

print(f"Parameters obtained from least squares: {theta}")

# Predict y values
y_pred = X_train_poly.dot(theta)

# Compute Mean Squared Error
mse = np.mean((y_train - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

# Plotting results (if needed)
plt.plot(range(len(y_train)), y_train, 'bo', label='True values')
plt.plot(range(len(y_train)), y_pred, 'r-', label='Fitted values')
plt.xlabel('Data Point')
plt.ylabel('Normalized Response')
plt.title('True vs Fitted Values')
plt.legend()
plt.savefig("x.png")
plt.show()
"""

Iteration 200000. MSE: 0.008673 TOL: 0.0000451
Optimal Theta: [ 0.40321129  3.27370785  0.50343709  8.68341456  8.08531226 10.40964703]

Parameters obtained from least squares: [  0.33898498   3.29212551   0.41478228 -21.41000838  29.55079269
  37.96660731]
Mean Squared Error: 0.015592009853843144
"""