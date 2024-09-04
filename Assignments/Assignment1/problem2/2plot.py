import numpy as np
import matplotlib.pyplot as plt
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
    gradients = np.zeros_like(theta)

    gradients[0] = np.mean(error)
    gradients[1] = np.mean(error * X.iloc[:, 0])
    gradients[2] = np.mean(error * X.iloc[:, 1])
    gradients[3] = np.mean(error * X.iloc[:, 0] * X.iloc[:, 1])
    gradients[4] = np.mean(error * X.iloc[:, 0]**2)
    gradients[5] = np.mean(error * X.iloc[:, 1]**2)
    
    return gradients

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

def grad_descent(n):
    tolerance = 1e-6
    step_size = 4e-1 *n
    theta, theta_prev = np.zeros(6), np.ones(6)

    iter = 0
    opt_pts = [theta]
    opt_grads = []

    while np.linalg.norm(theta - theta_prev) > tolerance:
        if iter > 200000:
            break
        if iter % 100 == 0:
            print('Iteration %d. MSE: %.6f TOL: %.7f' % (iter, mean_squared_error(theta, X_train, y_train),np.linalg.norm(theta - theta_prev)))
        theta_prev = theta
        gradient = mse_gradient(theta, X_train, y_train)
        theta = theta_prev - step_size * gradient
        opt_pts += [theta]
        opt_grads += [gradient]
        iter += 1

print("Optimal Theta:", grad_descent(1))
