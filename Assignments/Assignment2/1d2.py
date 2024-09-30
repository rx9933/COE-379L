from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [8, 4]

# =====================================================================
# DO NOT CHANGE: Dataset generation part
order_true = 5
true_coefficient = np.array([0.5, -1, -0.5, -2, 5])

def true_fn(X):
    f = np.ones((X.shape))
    for i in range(order_true):
        f += true_coefficient[i] * X**i
    return f

# Holdout samples
n_holdout_samples = 8
X_holdout = np.sort(np.random.rand(n_holdout_samples))

# Training samples
n_samples = 7
X_train = np.sort(np.random.rand(n_samples))

# =====================================================================

delta = 0.05
phi_train = np.random.randn(n_samples)
phi_holdout = np.random.randn(n_holdout_samples)

y_train_data = true_fn(X_train) * (1 + delta * phi_train)
y_train_true = true_fn(X_train)

y_holdout_data = true_fn(X_holdout) * (1 + delta * phi_holdout)
y_holdout_true = true_fn(X_holdout)

degree = 10
mses = []
coefs = []
alphas = np.logspace(-5, 1, 1000)
# for alpha in np.linspace(10**-5, 10**1, 10**4):
for alpha in alphas:# 100 or 1000 max

    polynomial_features = PolynomialFeatures(degree = degree, include_bias= False)
    linear_regression = Lasso(alpha =alpha, max_iter = 1000000)
    pipeline = Pipeline([("pf", polynomial_features), ("lr", linear_regression)])
    pipeline.fit(X_train[:, np.newaxis], y_train_data)
    mse = np.linalg.norm(y_holdout_data - pipeline.predict(X_holdout[:, np.newaxis]))**2
    coef = pipeline.named_steps['lr'].coef_
    mses.append(mse)
    coefs.append(np.linalg.norm(coef))
    

plt.scatter(mses, coefs, s=7) 
plt.xscale('log')  # Logarithmic scale for alpha (lambda) values
plt.yscale('log')  # Logarithmic scale for alpha (lambda) values
plt.ylabel(r"Solution Norm ($||\theta||_2$)")
plt.xlabel(r"Residual Norm (MSE = $||y - f(x,\theta)||_2$)")
plt.title(r"L-curve: Residual Norm vs Regularization for $\lambda$")

ind = np.argmin(mses)
plt.scatter(mses[ind], coefs[ind], s = 10, c = "r")
plt.savefig("1d2log.png")


plt.clf()
for alpha in alphas:# 100 or 1000 max

    polynomial_features = PolynomialFeatures(degree = degree, include_bias= False)
    linear_regression = Lasso(alpha =alpha, max_iter = 1000000)
    pipeline = Pipeline([("pf", polynomial_features), ("lr", linear_regression)])
    pipeline.fit(X_train[:, np.newaxis], y_train_data)
    mse = np.linalg.norm(y_holdout_data - pipeline.predict(X_holdout[:, np.newaxis]))**2
    coef = pipeline.named_steps['lr'].coef_
    mses.append(mse)
    coefs.append(np.linalg.norm(coef))
    

plt.scatter(mses, coefs, s=7) 
# plt.xscale('log')  # Logarithmic scale for alpha (lambda) values
# plt.yscale('log')  # Logarithmic scale for alpha (lambda) values
plt.ylabel(r"Solution Norm ($||\theta||_2$)")
plt.xlabel(r"Residual Norm (MSE = $||y - f(x,\theta)||_2$)")
plt.title(r"L-curve: Residual Norm vs Regularization for $\lambda$")

ind = np.argmin(mses)
plt.scatter(mses[ind], coefs[ind], s = 10, c = "r")
plt.savefig("1d2lin.png")



plt.clf()
for alpha in alphas:# 100 or 1000 max

    polynomial_features = PolynomialFeatures(degree = degree, include_bias= False)
    linear_regression = Lasso(alpha =alpha, max_iter = 1000000)
    pipeline = Pipeline([("pf", polynomial_features), ("lr", linear_regression)])
    pipeline.fit(X_train[:, np.newaxis], y_train_data)
    mse = np.linalg.norm(y_holdout_data - pipeline.predict(X_holdout[:, np.newaxis]))**2
    coef = pipeline.named_steps['lr'].coef_
    mses.append(mse)
    coefs.append(np.linalg.norm(coef))
    

plt.scatter(mses, coefs, s=7) 
plt.xscale('log')  # Logarithmic scale for alpha (lambda) values
# plt.yscale('log')  # Logarithmic scale for alpha (lambda) values
plt.ylabel(r"Solution Norm ($||\theta||_2$)")
plt.xlabel(r"Residual Norm (MSE = $||y - f(x,\theta)||_2$)")
plt.title(r"L-curve: Residual Norm vs Regularization for $\lambda$")

ind = np.argmin(mses)
plt.scatter(mses[ind], coefs[ind], s = 10, c = "r")
plt.savefig("1d3lin.png")
res = mses
reg = coefs
from scipy.interpolate import splprep, splev
# Fit a spline to the L-curve data
tck, u = splprep([np.log(res), np.log(reg)], s=0)

# Evaluate the first and second derivatives of the spline
derivatives_1 = splev(u, tck, der=1)
derivatives_2 = splev(u, tck, der=2)

# Compute curvature using the formula for 2D parametric curves
curvature = (derivatives_1[0] * derivatives_2[1] - derivatives_1[1] * derivatives_2[0]) / \
            ((derivatives_1[0]**2 + derivatives_1[1]**2)**1.5)

# Find the index of the maximum curvature (the corner of the L-curve)
optimal_idx = np.argmax(np.abs(curvature))
optimal_lambda = alphas[optimal_idx]
plt.plot(res[optimal_idx], reg[optimal_idx], markersize = 200, c = "r")
plt.savefig("1d3lin.png")
print(optimal_lambda)