#### INTRO TO REGRESSION - REGULARISATION

# SOURCE: WEEK 2, DAY 2 NOTEBOOK

# RIDGE REGRESSION (L2 NORM)
# LASSO REGRESSION (L1 NORM)
# ELASTICNET REGRESSION (COMBINED L1 AND L2 NORM)

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

## reload a pickled file

# recall: pickled numpy arrays X and y
np.savez('data/poly_data.npz', X, y)

# reload
npz = np.load('data/poly_data.npz')
# Retrieve each array
X = npz['arr_0']
y = npz['arr_1']
X

## VISUALISATION OVERFITTING - THE NEED FOR REGULARISATION

from IPython.core.pylabtools import figsize
figsize(5,5)
plt.style.use('fivethirtyeight')

# We start by seeding the random number generator 
# so that everyone will have the same "random" results
np.random.seed(9)

# Function that returns the sin(2*pi*x)
def f(x):
    return np.sin(2 * np.pi * x)

# generate points used to plot
# This returns 100 evenly spaced numbers from 0 to 1
x_plot = np.linspace(0, 1, 100)

# generate points and keep a subset of them
n_samples = 100

# Generate the x values from the random uniform distribution between 0 and 1
X = np.random.uniform(0, 1, size=n_samples)[:, np.newaxis]

# Generate the y values by taking the sin 
# and adding a random Gaussian (normal) noise term
y = f(X) + np.random.normal(scale=0.3, size=n_samples)[:, np.newaxis]

# Split the data into a 20/80 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# Plot the training data against what we know to be the ground truth sin function
fig,ax = plt.subplots(1,1)
ax.plot(x_plot, f(x_plot), label='ground truth', color='green')
ax.scatter(X_train, y_train, label='data', s=100)
ax.set_ylim((-2, 2))
ax.set_xlim((0, 1))
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.legend()

# this shows a plot of the datapoints with overlay of cubic 'ground truth'


## PLOTTING THE ERROR AS A FUNCTION OF POLYNOMIAL DEGREE FOR DEGREES 1-9

from sklearn.metrics import mean_squared_error

# Step through degrees from 0 to 9 
# and store the training and test (generalization) error.
train_error = np.empty(10)
test_error = np.empty(10)
for degree in range(10):
    est = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    est.fit(X_train, y_train)
    train_error[degree] = mean_squared_error(y_train, est.predict(X_train))
    test_error[degree] = mean_squared_error(y_test, est.predict(X_test))

# Plot the training and test errors against degree
plt.plot(np.arange(10), train_error, color='green', label='train')
plt.plot(np.arange(10), test_error, color='red', label='test')
plt.ylim((0.0, 1e0))
plt.ylabel('log(mean squared error)')
plt.xlabel('degree')
plt.legend(loc='upper left')

# this plot shows:
# - the log(MSE) for training set continues to go down as degree increases
# - HOWEVER, the log(MSE) for test set is parabolic,
# it initially goes down but at some point reaches a minimum, then goes up


## USING RIDGE REGRESSION ESTIMATOR TO FIT POLYNOMIAL OF DEGREE 9

# - Ridge will penalise coefficients for less important features
# - 'dense' solutions (most coefficients are nonzero)
# - fit the maximum degree you want to try
# - try different values of alpha (regularisation parameter)
# - the smaller the alpha, the larger the coefficients

from sklearn.linear_model import Ridge

# Set up a figure and axes for 8 plots, 2 per row for 4 rows
fig, ax_rows = plt.subplots(4, 2, figsize=(15, 20))

# A helper function to plot the absolute value of the coefficients 
# on the right-hand column plot
def plot_coefficients(est, ax, label=None, yscale='log'):
    coef = est.steps[-1][1].coef_.ravel()
    if yscale == 'log':
        ax.semilogy(np.abs(coef), marker='o', label=label)
        ax.set_ylim((1e-1, 1e8))
    else:
        ax.plot(np.abs(coef), marker='o', label=label)
    ax.set_ylabel('abs(coefficient)')
    ax.set_xlabel('coefficients')
    ax.set_xlim((1, 9))

# Try out 4 different values of the RidgeRegression parameter alpha 
# and watch how the resulting models change

# With higher values of alpha, more complex (more wiggly) models 
# will be more punished and thus less likely
degree = 9 # max degree being considered
alphas = [0.0, 1e-8, 1e-5, 1e-1]
for alpha, ax_row in zip(alphas, ax_rows):
    ax_left, ax_right = ax_row
    est = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    est.fit(X_train, y_train)
    plot_approximation(est, ax_left, label='alpha=%r' % alpha)
    plot_coefficients(est, ax_right, 
    	label='Ridge(alpha=%r) coefficients' % alpha)

plt.tight_layout()


## USING LASSO REGRESSION ESTIMATOR TO FIT POLYNOMIAL OF DEGREE 9

# - LASSO will drive most coefficients to zero ('sparse' solution)
# - penalise coefficients for less important features
# - fit the maximum degree you want to try
# - try different values of alpha (regularisation parameter)
# - the smaller the alpha, the larger the coefficients

from sklearn.linear_model import Lasso

# Create only 2 plot rows, only trying 2 alphas
fig, ax_rows = plt.subplots(2, 2, figsize=(15, 10))

# Plot the results next to the coefficient values for each of hte 2 alphas
degree = 9
alphas = [1e-3, 1e-2]
for alpha, ax_row in zip(alphas, ax_rows):
    ax_left, ax_right = ax_row
    est = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha))
    est.fit(X_train, y_train)
    plot_approximation(est, ax_left, label='alpha=%r' % alpha)
    plot_coefficients(est, ax_right, 
    	label='Lasso(alpha=%r) coefficients' % alpha, yscale=None)

plt.tight_layout()


## USING ELASTICNET ESTIMATOR TO FIND THE BEST OF BOTH WORLDS (L1 AND L2 NORM)

from sklearn.linear_model import ElasticNet

# Create only 2 plot rows, only trying 2 alphas
fig, ax_rows = plt.subplots(2, 2, figsize=(15, 10))

# Plot the results next to the coefficient values for each of hte 2 alphas
degree = 9
alphas = [1e-3, 1e-2]
for alpha, ax_row in zip(alphas, ax_rows):
    ax_left, ax_right = ax_row
    est = make_pipeline(PolynomialFeatures(degree), ElasticNet(alpha=alpha))
    est.fit(X_train, y_train)
    plot_approximation(est, ax_left, label='alpha=%r' % alpha)
    plot_coefficients(est, ax_right, 
    	label='ElasticNet(alpha=%r) coefficients' % alpha, yscale=None)

plt.tight_layout()


