import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Filter out classes 2 and 3
mask = np.isin(y_train, [0, 1])  # Keep only classes 0 and 1
X_train_binary = X_train[mask]
y_train_binary = y_train[mask]

# Split the data into training and testing sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train_binary, y_train_binary, test_size=0.2, random_state=42
)

# Train Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_split, y_train_split)
y_pred_nb = nb_classifier.predict(X_test_split)

# Train Logistic Regression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train_split, y_train_split)
y_pred_logistic = logistic_classifier.predict(X_test_split)

# Evaluate and compare results
print("Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test_split, y_pred_nb))
print(classification_report(y_test_split, y_pred_nb, target_names=twenty_train.target_names[:2]))

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test_split, y_pred_logistic))
print(classification_report(y_test_split, y_pred_logistic, target_names=twenty_train.target_names[:2]))
