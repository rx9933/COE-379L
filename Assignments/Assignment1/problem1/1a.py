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
    if iter == 10: # to confirm algo is somewhat improving fit. red represents the incorrect plot, green represents the optimized best fit plane
        theta_mid = theta
    theta_prev = theta
    gradient = mse_gradient(theta, X_train, y_train)
    theta = theta_prev - step_size * gradient
    opt_pts += [theta]
    opt_grads += [gradient]
    iter += 1


# 3D Plotting
fig = plt.figure()
ax = plt.axes(projection='3d')

# Scatter plot for the data points
ax.scatter3D(X_train['bmi'], X_train['bp'], y_train, color='b', marker='*')

# Create a meshgrid for BMI and BP
bmi_vals = np.linspace(X_train['bmi'].min(), X_train['bmi'].max(), 20)
bp_vals = np.linspace(X_train['bp'].min(), X_train['bp'].max(), 20)
bmi_grid, bp_grid = np.meshgrid(bmi_vals, bp_vals)

# Calculate corresponding predictions
z_vals = 0.4638 + theta[0] * bmi_grid + theta[1] * bp_grid
print("THETA", theta)

ax.plot_surface(bmi_grid, bp_grid, z_vals, color='green', alpha=0.1)

# Plot the regression plane
print("Theta mid", theta_mid)
z_vals = 0.4638 + theta_mid[0] * bmi_grid + theta_mid[1] * bp_grid
ax.plot_surface(bmi_grid, bp_grid, z_vals, color='red', alpha=0.1)

# Set axis labels
ax.set_xlabel('Body Mass Index (BMI)')
ax.set_ylabel('Blood Pressure (BP)')
ax.set_zlabel('Diabetes Risk')

plt.show()
X_train.to_excel("input.xlsx", engine='xlsxwriter')  
y_train.to_excel("output.xlsx", engine='xlsxwriter')  

                 