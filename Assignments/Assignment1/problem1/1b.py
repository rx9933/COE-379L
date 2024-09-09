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
    return np.mean((f(theta, X) - y) * X.T, axis=1).values # vs np.sum

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

tolerance = 1e-6
step_size = 4e-1
theta =  np.array([0,0]) # [your work!] np.array([0,0]) or np.array([4,4]) for (1.c)
theta_prev = np.array([1,1])
opt_pts = [theta]
opt_grads = []
# [your work!]
iter = 0
while np.linalg.norm(theta - theta_prev) > tolerance:
    # [your work!]
    if iter % 100 == 0:
        print('Iteration %d. MSE: %.6f' % (iter, mean_squared_error(theta, X_train, y_train)))
    if iter == 10:
        theta_mid = theta
    theta_prev = theta
    gradient = mse_gradient(theta, X_train, y_train)
    theta = theta_prev - step_size * gradient
    opt_pts += [theta]
    opt_grads += [gradient]
    iter += 1
print("Optimal Theta:", theta)

theta0_grid = np.linspace(0,4,101)
theta1_grid = np.linspace(0,4,101)


t0,t1 = np.meshgrid(theta0_grid, theta1_grid)
# Initialize J_grid to store the MSE values
J_grid = np.zeros(t0.shape)

# Compute the MSE for each combination of theta0 and theta1
for i in range(t0.shape[0]):
    for j in range(t0.shape[1]):
        theta = np.array([t0[i, j], t1[i, j]])
        J_grid[i, j] = mean_squared_error(theta, X_train, y_train)
"""
contours = plt.contour(t0,t1, J_grid, 10)
print(J_grid)
# plt.clabel(contours)
plt.axis('equal')
plt.show()
"""

# Create the contour plot
contours = plt.contour(t0, t1, J_grid, 10, cmap='viridis')

# Add labels to the contours
plt.clabel(contours, inline=True, fontsize=8)

# Ensure equal aspect ratio for the axes
plt.axis('equal')

# Display the plot
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Contour plot of Mean Squared Error')
plt.savefig('1b.png')
plt.show()
