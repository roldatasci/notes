#### INTRO FEATURE SCALING

# SOURCE: WK 3 DAY 1

##  FEATURE SCALING ##

# Standard Scaler - z score (mean = 0, sd = 1)

# MinMax Scaler = (X - min(X)) / (max(X) - min(X)), range -> [0,1] or [-1,1]

# Normalisation = X / ||X|| (i.e. divide by norm) <- not really covered


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

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot as plt


## READ IN DATA ##

df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
    )

df.columns=['Class label', 'Alcohol', 'Malic acid']

df.head()


## SCALERS ##

## 1) STANDARD SCALER
std_scale = preprocessing.StandardScaler().fit(df[['Alcohol', 'Malic acid']])
df_std = std_scale.transform(df[['Alcohol', 'Malic acid']])

# print mean and sd for standard scaler (to check mean = 0, sd = 1)
# - note, this is just a check (not really needed)
print('Mean after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_std[:,0].mean(), df_std[:,1].mean()))
print('\nStandard deviation after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'
      .format(df_std[:,0].std(), df_std[:,1].std()))

# Mean after standardization:
# Alcohol=-0.00, Malic acid=-0.00

# Standard deviation after standardization:
# Alcohol=1.00, Malic acid=1.00


## 2) MIN-MAX SCALER
minmax_scale = preprocessing.MinMaxScaler().fit(df[['Alcohol', 'Malic acid']])
df_minmax = minmax_scale.transform(df[['Alcohol', 'Malic acid']])


## PLOT DATA POINTS TO SEE DIFFERENCES IN SCALING ##
# - IN THIS EXAMPLE, Y ~ X (malic ~ alcohol)

def plot():
    plt.figure(figsize=(8,6))

    plt.scatter(df['Alcohol'], df['Malic acid'],
            color='green', label='input scale', alpha=0.5)

    plt.scatter(df_std[:,0], df_std[:,1], color='red',
            label='Standardized [$ N  (\mu=0, \; \sigma=1) $]', alpha=0.3)
    
    plt.scatter(df_minmax[:,0], df_minmax[:,1],
        color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)

    plt.title('Alcohol and Malic Acid content of the wine dataset')
    plt.xlabel('Alcohol')
    plt.ylabel('Malic Acid')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()

plot()
plt.show()


## SPLIT DATA

df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
    header=None,
    )
X_wine = df.values[:,1:]
y_wine = df.values[:,0]

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine,
    test_size=0.30, random_state=42)

## SCALE THE TRAIN AND TEST FEATURE MATRICES ##

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

 
## USING PCA FOR DIMENSIONALITY REDUCTION ## 

# perform PCA on standardised and non-standardised data

# on non-standardized data
pca = PCA(n_components=2).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# om standardized data
pca_std = PCA(n_components=2).fit(X_train_std)
X_train_std = pca_std.transform(X_train_std)
X_test_std = pca_std.transform(X_test_std)

# Lets visualize the first two principle components

# side-by-side plot
# left (non-standardised)
# right (standardised)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))


# non-standardised (X_train)
for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train[y_train==l, 0], X_train[y_train==l, 1],
        color=c,
        label='class %s' %l,
        alpha=0.5,
        marker=m
        )

# standardised (X_train_std)
for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std[y_train==l, 0], X_train_std[y_train==l, 1],
        color=c,
        label='class %s' %l,
        alpha=0.5,
        marker=m
        )

ax1.set_title('Transformed NON-standardized training dataset after PCA')    
ax2.set_title('Transformed standardized training dataset after PCA')    

for ax in (ax1, ax2):

    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.legend(loc='upper right')
    ax.grid()
plt.tight_layout()

plt.show() 

# result of this plot shows:
# - (left) PCA on non-standardised
# - (right) PCA on standardised data

# the above is just a demonstration of the importance of scaling

## FEATURE RANKING / FEATURE SELECTION ## 

## 1) SELECTION BASED ON PAIR-WISE RELATIONSHIPS

# - SelectKBest
# removes all but the K highest scoring features

# - SelectPercentile 
# removes all but a user-specified highest scoring % of features

# - Pearson's Correlation
# only appropriate for linear relationships between response and predictor

# - Mutual Information
# better in judging NONLINEAR relationship between response and predictor

## Mutual Information VS Correlation

# the code below compares values for 3 different relationships
# - a 'linear' relationship that, while linear, is very wide (high variance)
# - a zig-zag (oscillating) relationship (possibly harmonic or polynomial)
# - a plot of 2 variables in which there is no real relationship

#  'f-test'
# - the f-test here is the ratio of the coefficients from `f_regression`
# to the max/largest coefficient
# - it effectively measures linear relationship (correlation)
f_test, _ = f_regression(X, y)
print(f_test) # [ 187.42118421   52.52357392    0.47268298]
f_test /= np.max(f_test) 
print(f_test) # [ 1.          0.28024353  0.00252204]

# 'mutual information' [0,1]
# - the 'mi' here is the ratio of the coefficients from `mutual_info_regression`
# to the max/largest coefficient
# - it effectively measures NONLINEAR relationship
mi = mutual_info_regression(X, y)
print(mi) # [ 0.31431334  0.86235026  0.        ]
mi /= np.max(mi)
print(mi) # [ 0.36448455  1.          0.        ]

# the code below just plots simulated data showing the difference
# between f_test and mi results
np.random.seed(0)
X = np.random.rand(1000, 3)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test) 

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
              fontsize=16)
plt.show()


## 2) SELECTION BASED ON SHRINKAGE METHODS

# LASSO
# - sparse solution
# - shrinks coefficients for weak features to zero
# - built-in feature selection

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
  
boston = load_boston()
scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])
Y = boston["target"]
names = boston["feature_names"]
  
lasso = Lasso(alpha=.3)
lasso.fit(X, Y)
# Lasso(alpha=0.3, copy_X=True, fit_intercept=True, max_iter=1000,
#    normalize=False, positive=False, precompute=False, random_state=None,
#    selection='cyclic', tol=0.0001, warm_start=False)
lasso.coef_
# array([-0.23616802,  0.08100299, -0.        ,  0.54017417, -0.70027816,
#         2.99189989, -0.        , -1.08067403,  0.        , -0.        ,
#        -1.75682067,  0.63108483, -3.70696598])

# - use the coefficients above to judge the features
# - zero coefficients imply weak features

# NOTE: Ridge is more stable than LASSO
# does not force coeffs to zero, but just makes them really small
# probably still OK to use LASSO if you really want parsimony


## 3) SELECTION BASED ON TREE-BASED METHODS

# - every node in the decision trees is a condition on a SINGLE feature
# - splits the dataset into 2 so that similar response values end up in same set
# - `impurity` is measure used for optimal condition

# measures for regression vs classification: 
# - regression (trees): variance
# - classification: Gini impurity OR information gain/entropy

from sklearn.datasets import load_boston # dataset
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load boston housing dataset as an example
boston = load_boston() # instance of the dataset
X = boston["data"]
Y = boston["target"] # numeric target
names = boston["feature_names"]

# Selection using Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X, Y)
rf.feature_importances_

sorted(zip(rf.feature_importances_, names),reverse=True)
# [(0.41451334496962311, 'RM'),
#  (0.40845179219736705, 'LSTAT'),
#  (0.064251564086993701, 'DIS'),
#  (0.022225134249836537, 'CRIM'),
#  (0.018063028132252457, 'TAX'),
#  (0.015845912107650167, 'PTRATIO'),
#  (0.014753017164997778, 'NOX'),
#  (0.012515227462335656, 'AGE'),
#  (0.012276900567378008, 'B'),
#  (0.0077420271445252332, 'RAD'),
#  (0.0074500246784034007, 'INDUS'),
#  (0.00096141611643791225, 'CHAS'),
#  (0.00095061112219894483, 'ZN')]
