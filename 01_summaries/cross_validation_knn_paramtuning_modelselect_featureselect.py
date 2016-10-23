#### CROSS-VALIDATION

# SOURCE: WEEK 2 DAY 3 NOTEBOOK

# - knn (classifier)
# - sklearn.metrics (metrics.accuracy for classifier)
# - sklearn.cross_validation (cross_val_score())
# - KNeighborsClassifier()
# - LogisticRegression()

# GOALS
# - MODEL SELECTION
# - PARAMETER TUNING
# - FEATURE SELECTION

## STANDARD IMPORTS

# Python 2 & 3 Compatibility
from __future__ import print_function, division

# Necessary imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import seaborn as sns
from seaborn import plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RidgeCV

%matplotlib inline


# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target


## MODEL EVALUATION PROCEDURES

# - train and test on entire dataset (WRONG)
# - train / test split
# - cross-validation

## 1) WRONG WAY - TRAIN AND TEST ON ENTIRE DATASET

# - rewards overly complex models
# - overfits training data

# KNN (k = 5)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred)) # 0.967


# KNN (k = 1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred)) # 1.0


## 2) TRAIN/TEST SPLIT

# print the shapes of X and y
print(X.shape) # (150, 4)
print(y.shape) # (150,)

# STEP 1: split X and y into training and testing sets
# - specify random_state in order to replicate
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size=0.4, random_state=42)

# print the shapes of the new X objects
print(X_train.shape) # (90, 4)
print(X_test.shape) # (60, 4)

# print the shapes of the new y objects
print(y_train.shape) # (90,)
print(y_test.shape) # (60,)

# Train on training set, and Test on testing set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred)) # 0.983333333333


## 3) CROSS-VALIDATION with cross_val_score

# k-fold cross-validation (k = 10 recommended)
# - use each fold as a test set
# - uses stratified sampling, by default
# - every observation is used for both training and testing

# - slower than train/test split (k times slower)
# - less details (no ROC curve, which is available with train/test split)

# 3 use cases for cross-validation
# - model selection
# - parameter tuning
# e.g. how to choose K (for KNN), depth, leaves for decision trees
# - feature engineering or feature selection

from sklearn.cross_validation import cross_val_score

## K-FOLD CROSS-VALIDATION FOR PARAMETER TUNING

# K = 5
# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores) # prints scores for each of the 10 folds
# [ 1.          0.93333333  1.          1.          0.86666667  0.93333333
#   0.93333333  1.          1.          1.        ]
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean()) # 0.967

# TEST A RANGE OF K VALUES (K = 1 TO 30)
# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores) # plot these scores

# plot scores vs K-values
import matplotlib.pyplot as plt
%matplotlib inline

# plot the value of K for KNN (x-axis) 
# versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
# - low K -> low bias, high variance
# - high K -> high bias, low variance
# - generally good to pick a higher K with a max score (K = 20 in this case)


## ALTERNATIVE TO K-FOLD CV FOR PARAMETER TUNING: GridSearchCV

# - less computationally expensive
# - replaces the for loop used (for k in k_range) when searching for optimal
# value of K for KNN

from sklearn.grid_search import GridSearchCV

# define the parameter values that should be searched
k_range = list(range(1, 31))
print(k_range)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
# 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)
# {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
# 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

# fit the grid with data
grid.fit(X, y)
# GridSearchCV(cv=10, error_score='raise',
#        estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, 
# metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=20, p=2,
#            weights='uniform'),
#        fit_params={}, iid=True, n_jobs=1,
#        param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
# 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]},
#        pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)

# view the complete results (list of named tuples)
grid.grid_scores_
# [mean: 0.96000, std: 0.05333, params: {'n_neighbors': 1},
#  mean: 0.95333, std: 0.05207, params: {'n_neighbors': 2},
#  ...
#  mean: 0.95333, std: 0.04269, params: {'n_neighbors': 29},
#  mean: 0.95333, std: 0.04269, params: {'n_neighbors': 30}]

# examine the first tuple
print(grid.grid_scores_[0].parameters)
print(grid.grid_scores_[0].cv_validation_scores)
print(grid.grid_scores_[0].mean_validation_score)
# {'n_neighbors': 1}
# [ 1.          0.93333333  1.          0.93333333  0.86666667  1.
#   0.86666667  1.          1.          1.        ]
# 0.96

# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores) # list of mean scores for each possible K

# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# plot shows K = 20 is largest K that reaches maximum accuracy

# examine the best model (note that this says K = 13 is best model)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
# 0.98
# {'n_neighbors': 13}
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=13, p=2,
#            weights='uniform')


## SEARCHING MULTIPLE PARAMETERS SIMULTANEOUSLY WITH GridSearchCV

# define the parameter values that should be searched
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']

# create a parameter grid: 
# map the parameter names to the values that should be searched
# k = 1 to 30
# 2 different weight options
# ==> 30 x 2 = 60 pairs of (k, weight) x 10 folds = 600 models
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)
# {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
# 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 
# 'weights': ['uniform', 'distance']}

# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)

# view the complete results
grid.grid_scores_
# [mean: 0.96000, std: 0.05333, params: {'n_neighbors': 1, 'weights': 'uniform'},
#  mean: 0.96000, std: 0.05333, params: {'n_neighbors': 1, 'weights': 'distance'},
#  mean: 0.95333, std: 0.05207, params: {'n_neighbors': 2, 'weights': 'uniform'},
#  mean: 0.96000, std: 0.05333, params: {'n_neighbors': 2, 'weights': 'distance'},
# ...
# mean: 0.95333, std: 0.04269, params: {'n_neighbors': 30, 'weights': 'uniform'},
#  mean: 0.96667, std: 0.03333, params: {'n_neighbors': 30, 'weights': 'distance'}]

# examine the best model
print(grid.best_score_)
print(grid.best_params_)
# 0.98
# {'n_neighbors': 13, 'weights': 'uniform'}


## REDUCING COMPUTATIONAL EXPENSE USING RandomizedSearchCV

# - randomly sample

from sklearn.grid_search import RandomizedSearchCV

# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)

# n_iter controls the number of searches (iterations)
# n_iter = 10 x cv = 10 = 100 fitted models (vs 600 models from GridSearchCV)
rand = RandomizedSearchCV(knn, param_dist, cv=10, 
	scoring='accuracy', n_iter=10, random_state=42)
rand.fit(X, y)
rand.grid_scores_
# [mean: 0.98000, std: 0.03055, params: {'n_neighbors': 20, 'weights': 'uniform'},
#  mean: 0.96667, std: 0.04472, params: {'n_neighbors': 26, 'weights': 'distance'},
#  mean: 0.97333, std: 0.03266, params: {'n_neighbors': 15, 'weights': 'uniform'},
#  mean: 0.96667, std: 0.04472, params: {'n_neighbors': 8, 'weights': 'uniform'},
#  mean: 0.96667, std: 0.03333, params: {'n_neighbors': 22, 'weights': 'uniform'},
#  mean: 0.96667, std: 0.04472, params: {'n_neighbors': 4, 'weights': 'distance'},
#  mean: 0.96667, std: 0.04472, params: {'n_neighbors': 11, 'weights': 'uniform'},
#  mean: 0.97333, std: 0.03266, params: {'n_neighbors': 29, 'weights': 'distance'},
#  mean: 0.96667, std: 0.04472, params: {'n_neighbors': 10, 'weights': 'uniform'},
#  mean: 0.97333, std: 0.03266, params: {'n_neighbors': 12, 'weights': 'uniform'}]

# examine the best model
print(rand.best_score_)
print(rand.best_params_)
# 0.98
# {'n_neighbors': 20, 'weights': 'uniform'}

# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)
# [0.97999999999999998, 0.97999999999999998, 0.97999999999999998, 
# 0.97299999999999998, 0.97999999999999998, 0.97999999999999998, 
# 0.97999999999999998, 0.97299999999999998, 0.97999999999999998, 
# 0.97999999999999998, 0.97999999999999998, 0.97299999999999998, 
# 0.97999999999999998, 0.97999999999999998, 0.97999999999999998, 
# 0.97299999999999998, 0.97999999999999998, 0.97999999999999998, 
# 0.97999999999999998, 0.97999999999999998]


## K-FOLD CROSS-VALIDATION FOR MODEL SELECTION

# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=20) # K = 20 selected from before
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()) # 0.98

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean()) # 0.95

# in this above comparison, KNN is better model than LogisticRegression


## K-FOLD CROSS-VALIDATION FOR FEATURE SELECTION

# Exercise : Use cross validation for feature selection 
# in your linear regression data
# 10-fold cross-validation, 
# Use r2 as the scoring criteria


## FURTHER IMPROVEMENTS TO CROSS-VALIDATION

# - create a hold out set BEFORE doing CV
# - repeat CV with different random splits (cv = ?)

