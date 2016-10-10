#### INTRO TO REGRESSION - OLS AND POLYNOMIAL REGRESSION

# SOURCE: WEEK 2, DAY 2 NOTEBOOK

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

## INITIAL EDA

# Load the data in
df = pd.read_table('http://www.ats.ucla.edu/stat/examples/chp/p054.txt')
# Take a look at the datatypes
df.info()

# Quick peak
df.head()
df.shape # dimension

# clean up column names (as needed)

df.columns # check for white space
df.columns = df.columns.map(str.strip)
df.columns

# check correlations
df.corr()


## VISUALISATION (EXPLORATORY)

# Plot all of the variable-to-variable relations as scatterplots
sns.pairplot(df, size = 1.2, aspect=1.5)


## OLS REGRESSION WITH STATSMODELS ##

# 2 ways
statsmodels.api as sm # preferred (needs patsy)
statsmodels.formula.api as smf 

# 1) sm.OLS()
# statsmodels.api as sm
# patsy

# 1-a) Create your feature matrix (X) and target vector (y)
y, X = patsy.dmatrices('Y ~ X1 + X2 + X3 + X4 + X5 + X6', data=df, 
return_type="dataframe")

# 1-b) Create your model
model = sm.OLS(y, X)

# 1-c) Fit your model to your training set
fit = model.fit()

# 1-d) Print summary statistics of the model's performance
fit.summary() # note, this is training set results


# 2) smf.ols()
# statsmodels.formula.api

# 2-a) Define the model
# no need to use patsy to create feature matrix/target vector
lm1 = smf.ols('Y ~ X1 + X2 + X3 + X4 + X5 + X6', data=df)

# 2-b) Fit the model
fit1 = lm1.fit()

# 2-c) Print summary statistics of the model's performance
fit1.summary()


## PLOTTING OLS RESIDUALS (statsmodels)

# Use statsmodels to plot the residuals
# fit2 is a fit object
# model = sm.OLS(y, X) # instantiate OLS object
# fit = model.fit() # call fit() method on OLS object
# fit.summary() # call summary() method on fit object

fit2.resid.plot(style='o', figsize=(12,8))


## REGRESSION WITH sklearn ## 

# a) Create an empty model (instantiate)
lr = LinearRegression()

# b) Choose the predictor variables
# (here all but the first which is the response variable)
X = df.iloc[:, 1:] # Y ~ X1 + X2 + X3 + X4 + X5 + X6
X = df[['X1', 'X3', 'X6']] # selection by col_name

# c) Choose the response variable(s)
y = df.iloc[:, 0] # y is first col

# d) Fit the model to the full dataset (not advisable)
lr.fit(X, y)

# e) Print out the R^2 for the model against the full dataset
lr.score(X,y)

## sklearn METHODS (called on an instance, e.g. LinearRegression() object)

.fit() # fits a model to a training set
.score() # score (e.g. R^2)
.predict() # predicts y given a feature vector

##  sklearn ATTRIBUTES (called on LinearRegression() object)

# intercept
lr.intercept_

# coefficient estimates (array)
lr.coef_

## PICKLING RESULTS FOR LATER RETRIEVAL

# pickling a dataframe
df.to_pickle('data/survey_data.pkl')

# pickling a statsmodels object (e.g. a fit() object) with .save()
fit.save('data/survey_sm_model.pkl')

# pickling a sklearn object (e.g. a LinearRegression() object, `lr`)
from sklearn.externals import joblib
joblib.dump(lr, 'data/survey_sk_model.pkl')

# pickling numpy arrays X and y
np.savez('data/poly_data.npz', X, y)


## POLYNOMIAL REGRESSION WITH sklearn ##

# example (randomly generated data)

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

# single plot with fig, ax syntax

# Plot the training data against 
# what we know to be the ground truth sin function
fig,ax = plt.subplots(1,1)
ax.plot(x_plot, f(x_plot), label='ground truth', color='green')
ax.scatter(X, y, label='data', s=100)
ax.set_ylim((-2, 2))
ax.set_xlim((0, 1))
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.legend()


## GENERATING AND PLOTTING A POLYNOMIAL

# PolynomialFeatures(degree)
# make_pipeline() 

# import PolynomialFeatures and make_pipeline for Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Plot the results of a pipeline against ground truth and actual data
def plot_approximation(est, ax, label=None):
    """Plot the approximation of ``est`` on axis ``ax``. """
    ax.plot(x_plot, f(x_plot), label='ground truth', color='green')
    ax.scatter(X, y, s=100)
    ax.plot(x_plot, est.predict(x_plot[:, np.newaxis]), color='red', label=label)
    ax.set_ylim((-2, 2))
    ax.set_xlim((0, 1))
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.legend(loc='upper right',frameon=True)


## EXAMPLE: 3RD DEGREE POLYNOMIAL

# Set up the plot
fig,ax = plt.subplots(1,1)

# Set the degree of our polynomial
degree = 3

# Generate the model type with `make_pipeline`
# This tells it the first step is to generate 3rd degree polynomial features 
# in the input features 
# and then run a linear regression on the resulting features
est = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit our model to the training data
est.fit(X, y)

# Plot the results - see the function defined above
plot_approximation(est, ax, label='degree=%d' % degree)


## EXAMPLE: 2ND DEGREE POLYNOMIAL

# Set up the plot
fig,ax = plt.subplots(1,1)

# Set the degree of our polynomial
degree = 2

# Generate the model type with make_pipeline
# This tells it the first step is to generate 2nd degree polynomial features
# in the input features
# and then runa linear regression on the resulting features
est = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit our model to the training data
est.fit(X, y)

# Plot the results
plot_approximation(est, ax, label='degree=%d' % degree)


## EXAMPLE: MULTIPLE PLOTS OF DIFFERENT DEGREE POLYNOMIALS

# Step through degrees from 0 to 9 
# and store the training and test (generalization) error.

# This sets up 5 rows of 2 plots each (KEEP)
fig, ax_rows = plt.subplots(5, 2, figsize=(15, 20))

for degree in range(10):
    est = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    est.fit(X, y)
    # This sets the appropriate axis for each degree (KEEP)
    ax_row_left, ax_row_right = ax_rows[degree/2]
    if degree%2 == 0:
        ax = ax_row_left
    else:
        ax = ax_row_right
    plot_approximation(est, ax, label='degree=%d' % degree)


# Pickle the numpy arrays X and y
np.savez('data/poly_data.npz', X, y)


## REGULARISED AND CROSS-VALIDATED REGRESSION ##

## sklearn METHODS (called on an instance, e.g. LinearRegression() object)

.fit() # fits a model to a training set
.score() # score (e.g. R^2)
.predict() # predicts y given a feature vector

##  sklearn ATTRIBUTES (called on LinearRegression() object)

# intercept
lr.intercept_

# coefficient estimates (array)
lr.coef_

## CROSS-VALIDATION WITH sklearn's `train_test_split`

# NOTE: each time you train train_test_split, you get a different split

# a) Create an empty model (i.e. instantiate LinearRegression() class)
lr = LinearRegression()

# b) construct y, X with patsy.dmatrices()
y, X = patsy.dmatrices('Y ~ X1 + X2 + X3 + X4 + X5 + X6', data=df, 
return_type="dataframe")

# c) split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# d) Fit the model against the training data
lr.fit(X_train, y_train)

# e) Evaluate the model against the testing data
lr.score(X_test, y_test)




