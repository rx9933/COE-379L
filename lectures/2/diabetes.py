import numpy as np
import pandas as pd
from sklearn import datasets
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)
# Use only the BMI feature
diabetes_X = diabetes_X.loc[:, ['bmi']]
# The BMI is zero-centered and normalized; we recenter it for ease of presentation
diabetes_X = diabetes_X * 30 + 25
# Collect 20 data points
diabetes_X_train = diabetes_X.iloc[-20:]
diabetes_y_train = diabetes_y.iloc[-20:]
# Display some of the data points
pd.concat([diabetes_X_train, diabetes_y_train], axis=1).head()