## BIKESHARING DATASET

# data source: http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

# - OLS with statsmodels
# - visualisation with seaborn (fig, ax syntax)
# - LinearRegression with sklearn
# - indicator variables
# - automatic cross-validation (RidgeCV vs. train_test_split)


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

# read in data:
path_to_data = 'data/hour.csv'
df = pd.read_csv(path_to_data)

# check basic info
df.head()
df.info()
df.shape

# subset df
# Create a list of columns to keep
cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
'weathersit', 'temp', 'hum', 'windspeed', 'cnt']
# Select out only those columns
df = df[cols]
# Use head to review the remaining data, you should now have 12 columns
df.head()

# check correlations (only for numeric)
df.corr()

# sort correlations against target variable
df.corr()['cnt'].sort_values(ascending=False)


## VISUALISATION (EXPLORATORY)

# visualise pairwise correlations with pairplot
# NOTE: can subset desired columns
sns.pairplot(df[['mnth', 'season', 'hr', 'temp', 'hum', 'cnt']])


## MODELLING WITH statsmodels.formula.api

# Let's jump right in and try a model with statsmodels 
# using all variables above .10 correlation
lsm = smf.ols('cnt ~ temp + hr + yr + season + mnth + weathersit + hum', 
	data = df)
fit1 = lsm.fit()
print(fit1.summary())


## VISUALISATION FOR EXPLORING DISTRIBUTIONS (seaborn)

# Let's visualize some of the different variables against 'cnt'

# Temp
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.set_title('Counts by Temperature')
# Create a seaborn boxplot of count by temperature ordered by temperature
sns.boxplot(x=df['temp'].sort_values(), y=df['cnt'])
ax.set_xlabel('Temp')
ax.set_ylabel('Count')

# Humidity
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.set_title('Counts by Humidity')
# Create a seaborn boxplot of counts by humidity ordered by increasing humidity
sns.boxplot(x=df['hum'].sort_values(), y=df['cnt'])
ax.set_xlabel('Humidity')
ax.set_ylabel('Count')

# Year <-- not so useful because there are only 2 years
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.set_title('Counts by Year')
# Create a seaborn boxplot of counts by year ordered by increasing year
sns.boxplot(x=df['yr'].sort_values(), y=df['cnt'])
ax.set_xlabel('Year')
ax.set_ylabel('Count')

# Month <-- nonlinear relationship
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.set_title('Counts by Month')
# Create a seaborn boxplot of counts by month ordered by increasing month
sns.boxplot(x=df['mnth'].sort_values(), y=df['cnt'])
ax.set_xlabel('Month')
ax.set_ylabel('Count')

# Season <-- nonlinear relationship
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.set_title('Counts by Season')
# Create a seaborn boxplot of counts by season ordered by increasing season
sns.boxplot(x=df['season'].sort_values(), y=df['cnt'])
ax.set_xlabel('Season')
ax.set_ylabel('Count')


## INCORPORATING NON-LINEAR RELATIONSHIPS IN statsmodels

# Define your model with the appropriate R formula and ols()
lsm = smf.ols('cnt ~ temp + hr + mnth + mnth^2 + mnth^3 + mnth^4 
	+ yr + season + season^2 + season^3 + season^4 + hum', df)

# Call fit on your model
fit2 = lsm.fit()

# Call summary to print how you did
fit2.summary()


## FURTHER VISUALISATION

# Hour <-- nonlinear relationship
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.set_title('Counts by Hour')
# Create a seaborn boxplot of counts by hour ordered by increasing hour
sns.boxplot(x=df['hr'].sort_values(), y=df['cnt'])
ax.set_xlabel('Hour')
ax.set_ylabel('Count')


## UPDATE MODEL WITH NONLINEAR RELATIONSHIP FOR HOUR

# Define your model with the appropriate R formula and ols()
lsm = smf.ols('cnt ~ temp + hr + hr^2 + hr^3 + hr^4 
	+ mnth + mnth^2 + mnth^3 + mnth^4 + yr 
	+ season + season^2 + season^3 + season^4 + hum', df)

# Call fit on your model
fit2 = lsm.fit()

# Call summary to print how you did
fit2.summary()


## MINIMISE MODEL COMPLEXITY TO REDUCE OVERFITTING

# Use ols to create a model the same as the last minus mnth^3
lsm = smf.ols('cnt ~ temp + hr + hr^2 + hr^3 + hr^4 
	+ mnth + mnth^2 + mnth^4 + yr 
	+ season + season^2 + season^3 + season^4 + hum', df)

# Use fit to fit the model
fit2 = lsm.fit()

# Use summary to see how you did
fit2.summary()


# Use ols to create the same model minus season (linear)
lsm = smf.ols('cnt ~ temp + hr + hr^2 + hr^3 + hr^4 
	+ mnth + mnth^2 + mnth^4 + yr 
	+ season^2 + season^3 + season^4 + hum', df)

# Call fit to fit your model
fit2 = lsm.fit()

# Call summary to print results
fit2.summary()


####


## FIT A MODEL USING sklearn.linear_model.LinearRegression

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# instantiate a LinearRegression() object
lr = LinearRegression()

# Separate out our predictor variables from our reponse variables
X = df.iloc[:, 0:10] # predictors
y = df.iloc[:, 11] # target

# Let's generate a 70/30 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


## INDICATOR VARIABLES

# in this example, split hours into bins with indicator variables to reflect:
# - low-traffic night hours (`hr_low`)
# - high-traffic peak commute hours (`hr_high`)
# - middle-traffic daytime hours (`hr_med`)

# Create the 3 new columns initialized to 0
# You can do this the same way you might add a new entry to a Dict
df['hr_low'] = 0
df['hr_med'] = 0
df['hr_high'] = 0

# Use a map function on each new column 
# to make it a 1 if it should be, 0 otherwise
df['hr_low'] = df.hr.map(lambda x: 1 if (x < 7 or x > 20) else 0)
df['hr_med'] = df.hr.map(lambda x: 1 if x in 
	[7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20] else 0)
df['hr_high'] = df.hr.map(lambda x: 1 if x in [8, 17, 18] else 0)
df.head(24)

# with new variables, check correlations (in order) against response
df.corr()['cnt'].sort_values(ascending=False)

# This is promising, both the hr_high and hr_low have stronger correlations 
# than any from before
# Let's try building another model with this new information
X = df.loc[:, ['hr_low', 'hr_med', 'hr_high', 'temp', 'hum', 
'weathersit', 'windspeed', 'season', 'workingday']]
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lr.fit(X_train, y_train)
lr.score(X_test, y_test)

## AUTOMATIC CROSS-VALIDATION (AND FITTING) WITH sklearn

from sklearn.linear_model import RidgeCV

# create a ridge object with parameter `cv` set to 10
# - this does 10-fold cross-validation
# - RidgeCV has the k-fold CV built in
# but it is also a Ridge Regressor
rcv = RidgeCV(cv = 10)

## FITTING AND CROSS-VALIDATING WITH RidgeCV
# What about incorporating polynomials?  Ridge regression
# Is there an easier way to do train/test/validation 
# and Ridge Regression altogether?  Of course there is!

rcv.fit(X, y) # this is the full data set (not just train)
# this is okay because of built-in cross-validation
rcv.score(X_test, y_test) # 0.568

# VISUALISE DIFFERENCE FOR WORKDAY VS NONWORKDAY

# With just a little consideration of the data 
# we've improved our baseline model about 50%
# Let's look just a little further at hours
# For when workday = 1 vs 0
# Set up the plots
fig, axes = plt.subplots(2,1, figsize=(20,10))
ax = axes[0]
ax.set_title('Counts by Hour on Workdays')
# Generate your workingday boxplot here
sns.boxplot(x=df[df['workingday']==1]['hr'].sort_values(), y=df['cnt'], ax=ax)
ax.set_xlabel('Hour')
ax.set_ylabel('Count')
ax = axes[1]
ax.set_title('Counts by Hour on Non-Workdays')
# Generate your non-workingday boxplot here
sns.boxplot(x=df[df['workingday']==0]['hr'].sort_values(), y=df['cnt'], ax=ax)
ax.set_xlabel('Hour')
ax.set_ylabel('Count')

# These are clearly 2 very different distributions!  
# Maybe want to consider these 2 situations separately 
# and build a model for each (workday vs nonworkday)

# Let's just take a quick look for workingday = 1
X = df.loc[:, ['hr_low', 'hr_med', 'hr_high', 'temp', 'hum', 
'weathersit', 'windspeed', 'season', 'workingday']]
X = X[X['workingday']==1]
y = df[df['workingday']==1]['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test) # 0.65

