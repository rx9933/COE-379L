from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]

import pandas as pd
from sklearn import datasets
np.random.seed(0)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, LogisticRegression

## please keep the below codes for generating x, DO NOT CHANGE!

order_true = 5
true_coefficient = np.array([.5, -1, -.5, -2, 5])

def true_fn(X):
    f = np.ones((X.shape))
    for i in range(order_true):
        f += true_coefficient[i] * X**i
    return f

# ==========================================================================================
# holdout samples
n_holdout_samples = 8
X_holdout = np.sort(np.random.rand(n_holdout_samples)) 

# Training samples
n_samples = 7
X_train = np.sort(np.random.rand(n_samples)) 
# ==========================================================================================

# [CONTINUE YOUR WORK FROM HERE!]

# 1a
from matplotlib import pyplot as plt
delta = 0.05

phi_train = np.random.randn(n_samples)
phi_holdout = np.random.randn(n_holdout_samples)

y_train_data = true_fn(X_train) * (1+delta*phi_train)
y_train_true = true_fn(X_train)

y_holdout_data = true_fn(X_holdout) * (1+delta*phi_holdout)
y_holdout_true = true_fn(X_holdout)




def lasso(degree, alpha):
    polynomial_features = PolynomialFeatures(degree = degree, include_bias= False)
    linear_regression = Lasso(alpha = alpha)
    pipeline = Pipeline([("pf", polynomial_features), ("lr", linear_regression)])

    return pipeline.fit(X_train.reshape(X_train.shape[0], 1), y_train_data)


# Function to compute MSE and regularization norm for Lasso regression
def l_curve_values_lasso(alphas, degree):
    residual_norms = []
    reg_norms = []
    for alpha in alphas:
        model = lasso(degree, alpha)
        print(model.named_steps['lr'].coef_)
        lasso_y = model.predict(X_holdout.reshape(X_holdout.shape[0], 1))
        # print(lasso_y)
        mse = np.linalg.norm((lasso_y - y_holdout_true))**2  # Residual norm (MSE)
        coef_norm = np.linalg.norm(model.named_steps['lr'].coef_)  # Regularization norm (coef norm)
        residual_norms.append(mse)
        reg_norms.append(coef_norm)
    return residual_norms, reg_norms

# Plot L-curve for Lasso
def plot_l_curve_lasso(alphas, degree):
    residuals, norms = l_curve_values_lasso(alphas, degree)
    a = np.vstack((residuals, norms)).T
    print(a.shape)
    # print(residuals)
    # print(a[a[:, 0].argsort()])


    plt.plot(norms, residuals, label="Lasso L-curve")
    plt.ylabel(r"Solution Norm ($||\theta||_2$)")
    plt.xlabel(r"Residual Norm (MSE = $||y - f(x,\theta)||_2$)")
    plt.title(r"Lasso L-curve: Residual Norm vs Regularization for $\lambda$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig("1d.png")
    plt.show()

# Define range of alphas (lambdas)
alphas = np.linspace(10**-5, 10, 10**2)


# Plot L-curve for Lasso
plot_l_curve_lasso(alphas, 10)

