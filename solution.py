#!/usr/bin/env python3

import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

X = []
Y = []

NPOLY = 1    # Polynomial degree for feature augmentation

with open('data.csv', mode='r') as file:
    csvFile = csv.reader(file)

    line_counter = 0
    for line in csvFile:
        if line_counter == 0:  # Skip Header
            line_counter += 1
            continue
        Y.append(int(line[0]))
        X.append([int(value) for value in line[1:]])

X = np.array(X)
Y = np.array(Y)
# 3rd order features
poly = PolynomialFeatures(NPOLY)
poly.fit_transform(X)

X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.1, random_state=4)
fullreg = RandomForestClassifier()  # Chosen model
fullreg.fit(X, Y)

training_error = fullreg.score(X, Y)
print("Training on all features")
print("Training Error: ", fullreg.score(X, Y))
print("Validation Error: ", fullreg.score(X_val, Y_val))


# Feature selection
print("Feature selection:")
for n_features in range(5, 0, -1):
    estimator = RandomForestClassifier()
    selector = RFE(estimator, n_features_to_select=n_features)
    selector = selector.fit(X, Y)

    X = np.array([np.array(x) * selector.support_ for x in X])
    X_val = np.array([np.array(x) * selector.support_ for x in X_val])

    print(n_features, "features:")
    print("Selected features (x)", [i+1 for i in range(len(selector.support_)) if selector.support_[i]])

    poly = PolynomialFeatures(NPOLY)
    poly.fit_transform(X)

    reg = RandomForestClassifier()
    reg.fit(X, Y)
    print("Training Error: ", reg.score(X, Y))
    print("Validation Error: ", reg.score(X_val, Y_val))
    print("Training Error fraction: ", reg.score(X, Y)/training_error)
