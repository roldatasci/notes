#### CLASSIFICATION - LOGISTIC REGRESSION

# NOTE: refer to cross_validation_knn.py for K Nearest Neighbors

# SOURCE: WEEK 4 DAY 2 `US_Deaths...`

## STANDARD IMPORTS

# Python 2 & 3 Compatibility
from __future__ import print_function, division

# Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import seaborn as sns

%matplotlib inline

# specific imports below
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, 
roc_auc_score, roc_curve

## NOTE: 
# `cross_validation` module deprecated to `model_selection` module


## READ IN AND CLEAN DATA

df = pd.read_csv("DeathRecords.csv")
df = df[df.AgeType==1]
df = df[df.Age<120]
df.MannerOfDeath.value_counts()
# Code,Description
# 1,Accident
# 2,Suicide
# 3,Homicide
# 4,"Pending investigation"
# 5,"Could not determine"
# 6,Self-Inflicted
# 7,Natural
# 0,"Not specified"

y = 1-(df["MannerOfDeath"]==7).astype(int)
#non_natural_death is 1 if the death was not from natural causes


# Make 1-0 versions of these variables
df['Gender'] = (df['Sex']=='F').astype(int)
df['Autopsy'] = (df['Autopsy']=='Y').astype(int)


## SPLIT DATA INTO TRAIN, TEST, HOLDOUT

# - TRAIN for training
# - TEST for testing
# - HOLDOUT for post-test testing

# 70% to TRAIN and TEST, 30% to HOLDOUT
X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(df, y, 
	test_size=0.3, random_state=42)

# FROM THE 70% TRAIN & TEST, FURTHER SPLIT TO: 
# - 70% TRAIN (effectively 0.7 * 0.7 ~ 0.49 of all data), 
# - 30% TEST (effectively 0.7 * 0.3 ~ 0.21 of all data)
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, 
	test_size=0.3, random_state=42)


## VISUAL EXPLORATION OF CLASS BALANCE FOR `AGE`

# - check histograms to see if distributions are similar between classes

plt.figure(figsize=(10, 10))
# non-natural death
plt.subplot(2, 1, 1)
hist1 = plt.hist(X_train.loc[y==1,'Age'],bins=120,range=[0,120]);
# natural death
plt.subplot(2, 1, 2)
hist0 = plt.hist(X_train.loc[y==0,'Age'],bins=120,range=[0,120]);

# - alternatively, check the counts for each class


## FITTING A LOGISTIC REGRESSION MODEL

# let's also explore 'Autopsy" and 'Gender' (add to df)
X_train_1 = X_train.loc[:,['Gender','Age','Autopsy']]
X_test_1 = X_test.loc[:,['Gender','Age','Autopsy']]

# fit a logistic model
model = LogisticRegression()
fit = model.fit(X_train_1, y_train)

# compute predicted values
y_pred_vals = fit.predict(X_test_1)

# compute predicteod probabilities
# - the probabilities are in the second column
y_pred_prob = model.predict_proba(X_test_1)


# plot of the predicted probabilities (second column = col at index 1)
plt.hist(y_pred_prob[:,1]) 


## CHECKING ACCURACY with `accuracy_score` and `confusion_matrix`

# CHECKING CONFUSION MATRIX

# - rows are true state (+, -), columns are predicted state (+, -)
# - top left = true positive / hit / recall / sensitivity / correct identification
# - top right = false negative / miss / type 2 error / incorrect rejection
# - bottom left = false positive/alarm / type 1 error / incorrect identification
# - bottom right = true negative / specificity / correct rejection

confusion_matrix(y_test, y_pred_vals)

# CHECKING ACCURACY SCORE

accuracy_score(y_test, y_pred_vals)


## PLOTTING PRECISION-RECALL CURVE

# function similar to roc_curve function in sklearn.metrics
def pr_curve(truthvec, scorevec, digit_prec=2):
    threshvec = np.unique(np.round(scorevec,digit_prec))
    numthresh = len(threshvec)
    tpvec = np.zeros(numthresh)
    fpvec = np.zeros(numthresh)
    fnvec = np.zeros(numthresh)

    for i in range(numthresh):
        thresh = threshvec[i]
        tpvec[i] = sum(truthvec[scorevec>=thresh])
        fpvec[i] = sum(1-truthvec[scorevec>=thresh])
        fnvec[i] = sum(truthvec[scorevec<thresh])
    recallvec = tpvec/(tpvec + fnvec)
    precisionvec = tpvec/(tpvec + fpvec)
    plt.plot(precisionvec,recallvec)
    plt.axis([0, 1, 0, 1])
    return (recallvec, precisionvec, threshvec)

# plot the curve
pr_curve(y_test,y_pred_prob[:,1]);

# retrieve recall, precision, and threshold arrays
recallvec, precisionvec, threshvec = pr_curve(y_test,y_pred_prob[:,1]);

# find the threshold at which recall and precision are balanced
prcurvedf = pd.DataFrame(list(zip(recallvec, precisionvec, threshvec)))
prcurvedf[prcurvedf.iloc[:,0] == prcurvedf.iloc[:,1]]


## PLOTTING ROC CURVE and FINDING AREA UNDER CURVE

# `roc_curve` function in sklearn.metrics
# - plot true positive rate (y-axis) vs false positive rate (x-axis)
# - ideal is a right angle line along the y-axis and top border
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob[:,1])
plt.plot(fpr, tpr)

# ROC area under curve (ideal is as close to 1 as possible)
roc_auc_score(y_test,y_pred_prob[:,1])



