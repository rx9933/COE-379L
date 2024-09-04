import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

import pandas as pd
from sklearn import datasets

def f(theta, X):
    # [your work!]
    return theta[0] + theta[1] * X.iloc[:, 0] + theta[2] * X.iloc[:, 1] + \
        theta[3] * X.iloc[:, 0] * X.iloc[:, 1] + theta[4] * X.iloc[:, 0]**2 + theta[5] * X.iloc[:, 1]**2

def mean_squared_error(theta, X, y):
    # [your work!]
    y_pred = f(theta, X)
    return 1 / (2 * len(y)) * np.sum((y - y_pred) ** 2) 

def mse_gradient(theta, X, y):
    # [your work!] n
    # return np.sum((f(theta, X) - y) * X.T, axis=1).values # vs np.mean
    n = np.size(y)
    x0 = np.mean(X.iloc[:,0])
    x1 = np.mean(X.iloc[:,1]) # dimension d  
    
    # print("Aa",np.mean(X, axis = 0))
    # print("b",np.sum((f(theta, X) - y).values))
    # print("aa",np.sum((f(theta, X) - y).values) *  np.mean(X.values, axis = 0))
    return np.mean((f(theta, X) - y).values) *  np.array([0, x0, x1, x0*x1, x0**2, x1**2])
 
# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Collect 20 data points and use bmi and bp dimension
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

tolerance = 1e-6
step_size = 4e-1
theta, theta_prev = np.array([0,0,0,0,0,0]), np.array([1,1,1,1,1,1])
mse_gradient(theta, X_train, y_train)


iter = 0
# [your work!]
opt_pts = [theta]
opt_grads = []
while np.linalg.norm(theta - theta_prev) > tolerance:
    if iter > 200000:
        break
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