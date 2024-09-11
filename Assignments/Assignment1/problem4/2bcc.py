import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def f(theta, X):
    return (theta[0] 
            + theta[1] * X.iloc[:, 0] 
            + theta[2] * X.iloc[:, 1] 
            + theta[3] * X.iloc[:, 0] * X.iloc[:, 1])

def mean_squared_error(theta, X, y):
    y_pred = f(theta, X)
    return 1 / (2 * len(y)) * np.sum((y - y_pred) ** 2)  + .01/2*np.linalg.norm(theta)**2

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

    
    return gradients + .01*theta

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
X_train = X.iloc[-20:].loc[:, ['bmi', 'bp']]
y_train = y.iloc[-20:] / 300

tolerance = 1e-6
step_size = 4e-4
theta, theta_prev = np.zeros(6), np.ones(6)

iter = 0
opt_pts = [theta]
opt_grads = []

while np.linalg.norm(theta - theta_prev) > tolerance:
    # step_size = 4e-4 ##
    if iter > 20000:
        break
    if iter % 100 == 0:
        print('Iteration %d. MSE: %.6f TOL: %.7f' % (iter, mean_squared_error(theta, X_train, y_train),np.linalg.norm(theta - theta_prev)))
        # print(theta)
    # if iter >0:
    #     while mean_squared_error(theta, X, y) >= mean_squared_error(theta_prev, X, y):
    #         print(1)
    #         step_size /=2
    #         gradient = mse_gradient(theta, X_train, y_train)
    #         theta = theta_prev - step_size * gradient
    # print('Iteration %d. MSE: %.6f TOL: %.7f' % (iter, mean_squared_error(theta, X_train, y_train),np.linalg.norm(theta - theta_prev)))
    theta_prev = theta
    gradient = mse_gradient(theta, X_train, y_train)
    theta = theta_prev - step_size * gradient
    opt_pts += [theta]
    opt_grads += [gradient]
    iter += 1

print("Optimal Theta:", theta)
print(mean_squared_error(theta, X_train, y_train))
print(mean_squared_error(theta, X_train, y_train) - .01/2 * np.linalg.norm(theta)**2)

# print('Iteration %d. MSE: %.6f TOL: %.7f' % (iter, mean_squared_error(theta, X_train, y_train),np.linalg.norm(theta - theta_prev)))

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
        x1**2,
        x2**2
    ]).T
    return X_poly

# Create polynomial features
X_train_poly = create_polynomial_features(X_train.values)
y_pred = X_train_poly.dot(theta)
# Plotting results (if needed)
plt.plot(range(len(y_train)), y_train, 'bo', label='True values')
plt.plot(range(len(y_train)), y_pred, 'r-', label='Fitted values')
plt.xlabel('Data Point')
plt.ylabel('Normalized Response')
plt.title('True vs Fitted Values')
plt.legend()
plt.savefig("x.png")
plt.show()
# Iteration 12839. MSE: 0.011882 
# [0.45636826 3.53495095 0.27834219 0.97878671 