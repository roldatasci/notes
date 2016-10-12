#### ADDITIONAL REGRESSION METHODS

# - RandomForestRegressor() {sklearn.ensemble}
# - GradientBoostingRegressor() {sklearn.ensemble}
# - SGDRegressor() {sklearn.linear_model}

# SOURCE: WEEK 3, DAY 2 `Cal_Housing` NOTEBOOK

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

# specific imports below
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
## NOTE: 
# `cross_validation` module deprecated to `model_selection` module


## READ IN / SUBSET DATA

datafile = "cal_housing_data.csv" # week 3 day 2 folder
df=pd.read_csv(datafile)

# subset the data
df = df[df['medianHouseValue']<500000]
X=df.loc[:,'longitude':'medianIncome']
y=df['medianHouseValue']
X
plt.hist(df.medianHouseValue,range=[0,600000],bins=200);

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size=0.3, random_state=42)

## LINEAR REGRESSION (FOR COMPARISON AS BASE CASE TO IMPROVE UPON)

model_lr = LinearRegression(fit_intercept=True)
model_lr.fit(X_train, y_train)
pred_vals_lr = model_lr.predict(X_test)

# RMSE (for linear regression model)
RMSE_lr = np.sqrt(np.sum((pred_vals_lr - y_test)**2)/len(y_test))
RMSE_lr # 62010.025574271393

model_lr.coef_ # print out coefficients

# plot actual values vs predicted values (using holdout 'test' set)
pred_vals_lr = model_lr.predict(X_test)
y_test # from the splitting above
temp = np.linspace(0,500000,500) # to specify range of plot
plt.scatter(y_test,pred_vals_lr, alpha = .1)
plt.plot(temp,temp,'k--')


## LOG-LINEAR REGRESSION (LOG TRANSFORM ON RESPONSE)

model_lr_log = LinearRegression(fit_intercept=True)
model_lr_log.fit(X_train, np.log(y_train))
pred_vals_lr_log = np.exp(model_lr_log.predict(X_test))
RMSE_lr_log = np.sqrt(np.sum((pred_vals_lr_log - y_test)**2)/len(y_test))
RMSE_lr_log # 66399.115361212753 # slightly better than level

# plot actual vs predicted for log-linear model
temp = np.linspace(0,500000,500)
plt.scatter(y_test,pred_vals_lr_log, alpha = .03)
plt.plot(temp,temp,'k--')


## STOCHASTIC GRADIENT DESCENT (SGD) REGRESSOR ##

model_sgd = SGDRegressor(loss='squared_loss',n_iter=100,random_state=42)
model_sgd.fit(X_train, y_train)
pred_vals_sgd = model_lr.predict(X_test)
RMSE_sgd = np.sqrt(np.sum((pred_vals_sgd - y_test)**2)/len(y_test))
RMSE_sgd # 62010.025574271393


## RANDOM FOREST REGRESSION ##

# Try with increasing number of trees
rfmodel1 = RandomForestRegressor(n_estimators = 10, 
	max_features = 3, min_samples_leaf = 5, n_jobs=4)
rfmodel1.fit(X_train,y_train)
pred_vals_rf1 = rfmodel1.predict(X_test)

RMSE_rf1 = np.sqrt(np.sum((pred_vals_rf1 - y_test)**2)/len(y_test))
RMSE_rf1 # 49312.414164692222

# Try with different max features
rfmodel2 = RandomForestRegressor(n_estimators = 1000, max_features = 6,
                                min_samples_leaf = 5, n_jobs=4)
rfmodel2.fit(X_train,y_train)
pred_vals_rf2 = rfmodel2.predict(X_test)

RMSE_rf2 = np.sqrt(np.sum((pred_vals_rf2 - y_test)**2)/len(y_test))
RMSE_rf2 # 45423.006354064717


## GRADIENT BOOSTING REGRESSION

# Try with increasing number of iterations
gbmodel1 = GradientBoostingRegressor(n_estimators = 100, 
                                     learning_rate = .1,
                                    max_depth = 4)
gbmodel1.fit(X_train,y_train)
pred_vals_gb1 = gbmodel1.predict(X_test)

RMSE_gb1 = np.sqrt(np.sum((pred_vals_gb1 - y_test)**2)/len(y_test))
RMSE_gb1 # 46620.37992499712

# Try with different depths
gbmodel2 = GradientBoostingRegressor(n_estimators = 1000, 
                                     learning_rate = .1,
                                    max_depth = 7)
gbmodel2.fit(X_train,y_train)
pred_vals_gb2 = gbmodel2.predict(X_test)

RMSE_gb2 = np.sqrt(np.sum((pred_vals_gb2 - y_test)**2)/len(y_test))
RMSE_gb2 # 42767.85447899584


## PLOTS OF PREDICTED VS ACTUAL RESPONSE
# - LINEAR REGRESSION MODEL
# - RANDOM FOREST REGRESSION MODEL ()
# - GRADIENT BOOSTING REGRESSION MODEL ()

plt.figure(figsize=(10, 10))

temp = np.linspace(0,500000,500)

# linear regression model (RMSE: 62010.025574271393)
plt.subplot(3, 1, 1)
plt.scatter(y_test,pred_vals_lr, alpha = .1)
plt.plot(temp,temp,'k--')

# random forest regression model 2 (RMSE: 45423.006354064717)
plt.subplot(3, 1, 2)
plt.scatter(y_test,pred_vals_rf2, alpha = .1)
plt.plot(temp,temp)

# gradient boosting model 2 (RMSE: 42767.85447899584) <-- lowest
plt.subplot(3, 1, 3)
plt.scatter(y_test,pred_vals_gb2, alpha = .1)
plt.plot(temp,temp)
