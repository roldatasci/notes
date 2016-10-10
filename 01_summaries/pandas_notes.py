# PANDAS NOTES

#### SESSION MANAGEMENT

# standard imports - within a .py script

from __future__ import print_function, division
import pandas as pd
import numpy as np # pandas is based on numpy
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns # to use seaborn default

# additional imports and settings - within Jupyter notebook

from Ipython.display import Image
%matplotlib inline # enable inline plots within notebook

# check versions

print("Pandas version:",pd.__version__)
print("Numpy version:",np.__version__)

# display settings within pandas

pd.set_option('display.max_columns', None) # displays all columns
pd.set_option('display.max_rows', 25) # display max 25 rows
pd.set_option('display.precision', 3) # display results up to 3 decimals

# getting docs (can also shift-tab OR shift-tab-tab

?pd.Series # gives docstring for pandas Series object
pd.DataFrame?

# getting help
help(df.drop) # here, drop is a pandas method

#### CREATING DATAFRAMES AND SERIES

# creating DataFrames by passing a list of dictionaries

data_lst = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6, 'd': 7}]
df = pd.DataFrame(data_lst)

# make a dataframe from a dictionary

# the example below builds the following df:
# col A: all values are: 1.0
# col B: all entries are: 2013-01-02
# col C: all entries are 1.0
# col D: all entries are 3 (integer)
# col E: entries alternate: test, train, test, train (categorical)
# col F: all entries are 'foo'
df2 = pd.DataFrame({ 'A' : 1., \
                   'B' : pd.Timestamp('20130102'), \
                   'C' : pd.Series(1,index=list(range(4)),dtype='float32'), \
                   'D' : np.array([3] * 4,dtype='int32'), \
                   'E' : pd.Categorical(["test","train","test","train"]), \
                   'F' : 'foo' })


# creating DataFrames by passing lists of lists
# note: not recommended because length of data_cols
# must match length of each sublist in data_vals

data_vals = [[1,2,3], [4,5,6]]
data_cols = ['a', 'b', 'c']
df = pd.DataFrame(data = data_vals, columns = data_cols)

# creating a DataFrame from a numpy array (matrix)
# e.g. 6x4 matrix with elements from a normal dist
# column/variable names are A, B, C, D

df1 = pd.DataFrame(np.random.randn(6,4), columns = list('ABCD'))

# reading external data into DataFrames (csv format)

df = pd.read_csv('my_data.csv')
df = pd.read_csv('my_data.csv', header = None)
df = pd.read_csv('my_data.csv', header = None, names = ['col1', 'col2', ..])
df = pd.read_csv('my_data.csv', delimiter = ';') # if ; instead of , is used

#### INITIAL DATAFRAME EXPLORATION

# initial look at data - attributes

df.shape # dimensions or rows, columns
df.columns # list of all column names
df.type # data type of each pandas col (Series)

# initial look at data - DataFrame methods (numerical)

df.info() # data type for each column, and number of null values
df.describe() # summary stats (count, mean, std, min, 25%, 50%, 75%, max)
df.head(n) # first n rows (default n = 5, but can pass any number as param)
df.tail(n) # last n rows (default n = 5, but can pass any number as param)

df.corr() # pairwise correlation
df.cov() # pairwise covariance

# univariate exploration (individual col/variable)

# inital look at data - column (Series) .value_counts()
# note that type(df.colname.value_counts()) is a Series object

df.colname.value_counts() # returns unique values and their counts
df.colname.value_counts().plot(kind = 'barh') # plots value_counts
 
# determine unique values for each variable/columns - col/Series .unique()
# .unique() returns a list of unique values 

df.colname.unique()

# removing leading spaces
# - turn variable into string, then strip leading whitespace

df.colname = df.colname.str.strip()
df['colname'] = df.colname.str.strip()

# initial look at data - column-specific descriptives
# can pass a list of desire columns to examine
# descriptives for a subset of columns

df[['age', 'capital_gain', 'capital_loss', 'hours_per_week']].describe()
df.colname.mean()
df.colname.median()

# examine a random sample of the data
# sample size n, starting from seed k

df.sample(n, ramdom_state = k)

# editing column names - removing spaces to facilitate dot notation

df2 = df.copy() # make a new copy to preserve original df
cols = df2.columns.tolist() # create a list of column names
cols = [col.replace(' ', '_') for col in cols] # replace spaces with _
df2.columns = cols # assign new column names to replace old ones
df2.columns # to verify

# creating and dropping columns (2 methods)
# - renaming using bracket notation {}
# - feature engineering with .eval() method

# 1) renaming columns with bracket notation
# df.rename(columns = {'old_col_name': 'new_col_name'}, inplace = False)
# inplace is false by default

df.rename(columns = {'col 1': 'col_1', 'col 2': 'col_2'}, inplace = True)

# 2) feature engineering with .eval() method

df.eval('new_col = col1 - col2')
df.columns # verify new_col was created

# dropping a column (axis = 1 for columns, unlike numpy)
# NOTE: remember to drop in place or assign df with dropped column(s) to new df

df.drop('new_col', axis = 1, inplace = True)
df.columns # verify new_col was dropped


#### NULL VALUES 
# (see pandas docs for further parameters)

df[df.colname.isnull()] # finds null values in variable 'colname'
df.colname.fillna(-1, inplace = True) # fill nulls in colname with -1s
df.colname.fillna(df.colname.median(), inplace = True) # use median for nulls
df.dropna(how = 'any') # default, drops any observation with null value
df.dropna(inplace = True) # drops all null values
df.isnull() # returns True for rows with null
df.notnull() # returns True for rows with no null values

# example: restricting observation to non-null vales for variable 'age'

titanic = titanic[titanic.age.notnull()]


#### SLICING AND INDEXING

# one-way slicing - grab an entire column

df['colname']
df.colname # dot notation only works if colnames have no spaces

# one-way slicing - grab a selection of columns by passing a list

df[['col1', 'col2']]

# one-way slicing - grab a range of rows (must have BOTH starting and ending index)

df[:3] # first 3 rows (indices 0 to 2)
df[1:2] # row at index 1

# two-way slicing - grab a range of rows and columns (3 methods)

df.loc[] # label location-based
df.iloc[] # integer location-based
df.ix[] # MOST FLEXIBLE, primarily label location, but falls back to integer

# two-way slicing with .loc[] method - uses only labels for column references
# NOTE: .loc[] row ranges INCLUDE the ending index

df.loc[0, 'colname'] # row index and col name are considered 'labels'
df.loc[0:10, 'colname'] # numeric row ranges and colnames
df.loc[10:15, ['col1', 'col2']] # can pass list of columns

# two-way slicing with .iloc[] - uses numeric indices for both rows and columns
# .iloc[] remembers indices even if rows are shuffled
# NOTE: .iloc[] rows ranges EXCLUDE the ending index

df.iloc[0, 0] # col reference must be an integer, not a label
df.iloc[0:10, 0] # col reference must be an integer, not a label
df.iloc[10:15, [0, 4]] # col reference must be an integer, not a label
df.iloc[-2:] # last two rows
df.iloc[::2, 2:5].head() # every other row, cols 3 to 5

# two-way slicing with .ix[] - MOST FLEXIBLE
# NOTE: .ix[] row ranges also INCLUDE the ending index

# also works with .loc[]
df.ix[0, 'colname']
df.ix[0:10, 'colname']
df.ix[10:15, ['col1', 'col2']]
# also works with .iloc[]
df.ix[0, 0]
df.ix[0:10, 0]
df.ix[10:15, [0, 4]]

# query-based slicing (2 ways)
# - using a mask/condition
# - using .query() method on a DataFrame object <- PREFERRED

# query-based slicing using a single mask/condition
# all variables, subset of rows

df['colname'] <= 0.8 # mask - creates a vector of booleans
df[df['colname'] <= 0.8] # plug in the mask as an index
df[df.colname <= 0.8] # dot notation

# query-based slicing using multiple conditions in separate () joined by '&'
# df.ix[(condition_1) & (condition_2), [list_of_cols]]

(df['colname'] >= 0.02) & (df['colname'] <= 0.06) # conditions
df.ix[(df['colname'] >= 0.02) & (df['colname'] <= 0.06)] # plug in conditions
df.ix[(df.occupation == ' Tech-support') & (df.gender == 'Male'), 
['age', 'education', 'occupation', 'gender', 'income']].head()

# query-based slicing using .query('condition1 and condition2 and ...')
# .query('condition1 and condition2')[[list_of_cols]] returns a dataframe subset

df.query('col1 >= 0.2 and col1 <= 0.6 and col2 >= 3.0 and col2 <= 5.0')
df.query('age >=30 and gender=="Male" and age<=40').head()
df.query('occupation == " Tech-support" and gender == "Male"')
[['age', 'education', 'occupation', 'gender', 'income']].head()


#### GROUPBY, SORT_VALUES 

# SQL-style query methods
# - .groupby('colname') <- returns a groupby object on w/c to call methods
# - .sort_values('colname')

## SQL-style .groupby() - grouping by one column/variable

df['col1'].unique() # check if 'col1' has only a few unique values/levels
groupby_obj = df.groupby('col1') # instantiate the groupby object

# for the following, the rows represent the groups of 'col1'
groupby_obj.mean() # shows the group means for all other remaining columns
groupby_obj.max() # shows the group max values for all other remaining columns
groupby_obj.count() # shows the group counts for all other remaining columns
groupby_obj.count()['col2'] # shows the 'col1' group counts for variable 'col2'

# SQL-style .groupby() - grouping by multiple columns
# group by 'col1', then 'col2', and so on...
# calling any method (e.g. .count(), .mean()) converts Groupby object into a df

df.groupby(['col1', 'col2']) # groupby object
df.groupby(['col1', 'col2']).count() # grp and subgrp counts for all other cols
df.groupby(['col1', 'col2']).count()['col3'] # grp/subgrp cts for 'col3'

# groupby education and age, then look at mean hours per week and capital gain
df.groupby(['education','age',])[['hours_per_week','capital_gain']].mean()

# groupby education and age, then look at mean hours per week
df.groupby(['education','age']).hours_per_week.mean()

# reset index (this turns the aggregration into a DataFrame)
# this flattens the data
# creates a new df with just education, age, hours_per_week, capital_gain
df.groupby(['education','age',])[['hours_per_week','capital_gain']].mean()
.reset_index()

## .groupby() multiple columns AND multiple functions
# df.groupby([list_of_vars_to_groupby]).var.agg([list_of_fns_to_compute])
df.groupby(['income', 'native_country']).age.agg(['count', 'mean'])
# grouped in order of which column is listed first
# group by income, then by native country, then
# agg(regate) computes the count and mean of age
# returns count and mean (separate cols) of age
# for each income group and native country

# combine groupby with boolean
df[df.native_country == ' United-States'].groupby(['education']).hours_per_week
.mean()

# SQL-style .sort_values() - sorting by one columns

df.sort_values('col1') # sorts by 'col1'
df.sort_values('col1', ascending = False) # ascending = True, by default

# SQL-style .sort_values() - sorting successively by multiple columns

df.sort_values(['col1', 'col2'], ascending = False)

# groupby income and country
# and then sort by their mean age within each data block
df_grouped = df.groupby(['income','native_country'])
.mean().sort_values('age', ascending = True)

# We want to group people by their income and country
# Then sort them by their income (ascending), 
# and then sort by average age within that group (descending) 
(df
 .groupby(['income','native_country'])
 .mean()
 .reset_index()
 .sort_values(['income','age'], ascending=[True,False])
)
# Note: In above example, we sort by the SAME column which we grouped by earlier 
# (eg. we first groupby 'income' and then sort by 'income')
# In this case, we must use .reset_index() to re-index the groupby objects, 
# because the 'income' column no longer exists
# after the groupby and hence cannot be sorted directly


#### TIME SERIES DATA

# reading in a time series with a datetime object as index

# example: read in weather data, set the Date/Time as the index
# first colname = 'Date/Time' is being used as the index
# Take a look (univariate time series plot of var 'Temp(C)')

weather = pd.read_csv('day04_data/weather.csv', index_col='Date/Time')
weather['Temp (C)'].plot(figsize=(15, 6))


## MAP, APPLY, APPLYMAP

# using map() to convert values of a series
# Series.map(function)

# example: convert celsius to fahrenheit

# Function that converts Celsius to Fahrenheit
def celsius_to_fahrenheit(temp):
    return (9.0*temp/5.0) + 32
# Use it to make the conversion and add a new column for it
# map() is a method on a Series (df col)
# Series.map(function)
weather['Temp (F)'] = weather['Temp (C)'].map(celsius_to_fahrenheit) # new col
weather.head()

# using lambda function to convert celsius to fahrenheit

weather['Temp (F)'] = weather['Temp (C)'].map(lambda x: 9.0*x/5.0 + 32)
weather.head()

# using apply() to perform operations on select columns within a df
# df.apply(function)

# example: select only temperature columns and find their range
# apply works on each Series as a whole within a dataframe
# `lambda input: input.max() - inputmin.max()` <-- this gets the range
# this gives you one (scalar) output per column
weather_temps = weather[['Temp (C)', 'Temp (F)']] # new df with just 2 cols
weather_temps.apply(lambda x: x.max() - x.min())

# using applymap() to perform operations on every elem of every series of a df
# df.applymap(function)

# example: # apply function to format numerics
format = lambda x: '%.2f' % x # 2 decimal float
weather_temps.applymap(format)
# does to every single element in the whole df


## STRING OPERATORS

# replacing string values - Series.str.replace(oldval, newval)

# example:
# replaces anything that begins with 'Fog...' with ' Fog' (extra space)
# - df.col is just a single column of Series
# - .str turns elem in Series into strings
# - .replace(oldvalue, newvalue) is a essentially 
spacey_fog = weather.Weather.str.replace('^Fog', ' Fog') 
spacey_fog.unique() # check if value changed

# strip() to remove leading/trailing whitespace

spacey_fog.str.strip().unique() # removed whitespace in 'Fog'

# check if something appears within a Series/col - Series.contains('pattern')

# example: subset observations that have 'Snow' as value for 'Weather' variable
is_snowing = weather.Weather.str.contains('Snow') # new boolean Series or mask
weather[is_snowing] # df[mask] to subset df
# weather[is_snowing != True] # quick way for not snowing

# using a DateTimeIndex to subset a df

# example:
# date range with frequency specifier every 3 days, 
# starting january 1st, for 6 cycles
# normally, pd.date_range('start', 'end', ...)
dates = pd.date_range('20130101', periods=6, freq='3D') 
# freq is frequency, here '3D' is every 3 days
dates # this is a DatetimeIndex which needs to be applied as an index to a df
# Create a random dataframe with the 'dates' DatetimeIndex
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df # note that end range (6) is included

## upsampling or downsampling with resample() and DateTimeIndex

# - upsampling is increasing the frequency (e.g. going from daily to hourly)
# - dowsampling is decreasing the frequency (e.g. going from hourly to daily)
# downsampling reduces the observations

# example: downsample from original hourly frequency to monthly frequency
# returns the proportion of hours in a month that is snowing
weather.index # check index or row labels  (len = 8784, starts 2012-01-01)
# turn index into a DateTimeIndex on original frequency (H - hourly)
weather.index = pd.date_range('20120101', periods = 8784, freq = 'H')
weather.index # confirm index is now a DateTimeIndex
# construct a boolean Series or mask
is_snowing = weather.Weather.str.contains('Snow') # True, False for each row
is_snowing.index # confirm index of mask is a DateTimeIndex
# convert boolean Series/mask elements (True,False) into (0, 1)
is_snowing.astype(float)
# compute the proportion of hours in a month when it is snowing
# - average of 0s and 1s will give you the percentage of 1s (True)
is_snowing.astype(float).resample('M', how=np.mean) # deprecated syntax
is_snowing.astype(float).resample('M').mean() # new syntax

# example: compute proportion of every day that is snowing
# same as above, except change 'M' to 'D'
is_snowing.astype(float).resample('D', how=np.mean) # deprecated syntax
is_snowing.astype(float).resample('D').mean() # new syntax

# other ways to convert an index to a datetime index.
weather.index = weather.index.to_datetime()

# another way to do it is to add a date parser in reading the csv
weather = pd.read_csv('data/weather.csv', 
	index_col='Date/Time',parse_dates='Date/Time')


## DIFFERENCING and SHIFTING

# differencing a Series (calculate difference between rows) - Series.diff()

# example: create a new Series that is the difference of an existing Series
weather['Temp Diff'] = weather['Temp (F)'].diff()
weather.head() # check result

# shifting forward or backward by some number of rows (periods)

# examples (useful for creating lagged variables):
# 'Forward' is a new col that takes Temp(F) and shift the first 2 values down
weather['Forward'] = weather['Temp (F)'].shift(periods=2)
# 'Backward' is a new col that takes Temp(F) and shift the first value UP
weather['Backward'] = weather['Temp (F)'].shift(periods=-1)
weather


#### COMBINING DATA SETS WITH PANDAS

# - pd.merge() <-- PREFERRED
# - pd.concat()
# - df.join() # called on a df object/instance

# converting a categorical variable into dummy variables - get_dummies()

# example: column = 'quality'
wine_df.quality.unique() # array([5,6,7,4,8,3])
quality_dummies = pd.get_dummies(wine_df.quality, prefix = 'quality')
quality_dummies.head() # shows a df with newly created columns:
# quality_3, quality_4, quality_5, quality_6, quality_7, quality_8

# converting a var into categorical with .astype()
df.colname.astype('category')
df.info() # check

# joining newly created dummy variables to the df using df.join()
# NOTE: this method is called on a DataFrame object

joined_df = wine_df.join(quality_dummies)
joined_df.head() # verify join

# joining newly created dummy variables to the df using pd.concat()
# NOTE: this is a pandas module function

joined_df2 = pd.concat([quality_dummies, wine_df], axis = 1)
joined_df2.head() # verify concatenation

## combining dfs vertically with pd.concat()
weather_concat = pd.concat([weather.iloc[0:100,], weather.iloc[200:300,]])
weather_concat.info()

## add a single row to a df with .append()
weather_append = weather_concat.append(weather.iloc[305,])
weather_append.info() # count increased by 1

## combining two different datasets using pd.merge()

red_wines_df = pd.read_csv('data/winequality-red.csv', delimiter = ';')
white_wines_df = pd.read_csv('data/winequality-white.csv', delimiter = ';')
red_wines_df.columns # check names
white_wines_df.columns # check names

# create new dfs showing the quality groups and group means for 'fixed acidity'
rwq_df = red_wines_df.groupby('quality').mean()['fixed acidity'].reset_index()
rwq_df.head()
wwq_df = white_wines_df.groupby('quality').mean()['fixed acidity'].reset_index()
wwq_df.head()

# using pd.merge() to join 2 dfs on the same column/variable
pd.merge(rwq_df, wwq_df, on = ['quality'], suffixes = [' red', ' white'])
# output is a new df with columns:
# ['quality', 'fixed acidity red', 'fixed acidity white']

# alternative syntax for .merge()

# example: merge the df 'weather' with the boolean Series 'is_snowing'
# 'weather' is the left (original) df (eveything before Weather_y)
# and pd.DataFrame(is_snowing) is the newly created right df
# (created from is_snowing, which is a boolean Series of True,False)
weather_snowing = weather.merge(pd.DataFrame(is_snowing), 
	left_index=True, right_index=True) # matches up the rows by DateTimeIndex
weather_snowing.head() # this is a new df

# using pd.cut() to turn a column with continuous data into categoricals

pd.cut(red_wines_df['fixed acidity'], bins = np.arange(4, 17)).head()

# create a new series to use as categorical variable with levels as bins
# join to the df (as last column)
fixed_acidity_bins = np.arange(4,17)
fixed_acidity_series = pd.cut(red_wines_df['fixed acidity'],
bins = fixed_acidity_bins, labels = fixed_acidity_bins[:-1])
fixed_acidity_series.name = 'fa_bin' # rename new series
red_wines_df = pd.concat([red_wines_df, fixed_acidity_series], axis = 1)

# pivot_table() function in pandas module

# e.g.
# mean 'residual sugar' for each 'quality' category and fixed acidity/'fa_bin'
# this outputs a crosstabulation where:
# - 'fa_bin' bins are the columns
# - 'quality' levels are the rows/index
# - mean of 'residual sugar' values for each cross-group
# (e.g. bin 4 quality 3 is a cross-group)

pd.pivot_table(red_wines_df, values = 'residual sugar', index = 'quality',
columns = 'fa_bin') # aggfunc = np.mean, by default

pd.pivot_table(red_wines_df, values = 'residual sugar', index = 'quality',
columns = 'fa_bin', aggfunc = np.max) # returns the max 'residual sugar' value
# for each cross-group (.e.g max value for quality 4, fa_bin 4)
