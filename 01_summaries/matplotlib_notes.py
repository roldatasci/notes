#### VISUALISATION (MATPLOTLIB)

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


## LINE PLOTS - solid, connected line

x = range(10)
y = range(10)
plt.plot(x,y); # plot a line (semicolon suppresses object type returned)
plt.plot(x, np.power(x,2)); # plot a quadratic

## plotting math functions with numpy

x = range(20)
y = np.sin(x)
plt.plot(x,y);

## smoother 2D line plots using linspace

x = np.linspace(0, 20, 100) # between 0 and 20, generate 100 data points
y = np.sin(x)
plt.plot(x, y) # should be smoother


## SCATTER PLOTS - datapoints, not connected

# (specify two variables)
df.plot(kind = 'scatter', x = 'col1', y = 'col2') # plots 'col2' ~ 'col1'
df.plot(x, y) # for 2 Series or 2 arrays, by default shows scatterplot

plt.scatter(x, y) # plots the actual data points as points (not as a line)

# using jittering for smoother scatter plot

x = range(20)
y = np.arange(20) + (np.random.randn(20) * 2) # adds jitter/noise
plt.scatter(x, y, c = 'r') # c = 'r' makes the datapoints red

# add a trend line to a scatter plot

plt.scatter(x, y)
plt.plot(x, x, 'r--'); # adds a 45deg the trend line (r = red, -- = broken line)
plt.plot(x, y, 'r--') # connects the original data points instead


## HISTOGRAM (of a specific column/variable)

df['col1'].plot(kind = 'hist')
df.col1.plot(kind = 'hist')

plt.hist(x) # for an array
plt.hist(x, bins=20); # set no. of bins


## VERTICAL BAR CHARTS

years = np.arange(2010, 2015) # [2010 2011 2012 2013 2014]
values = [2, 5, 9, 5, 7]

plt.bar(years,
        values,
        color='blue',
        edgecolor='none',
        width=0.5,
        align='center',
        label='y1')
plt.xticks(years, [str(year) for year in years]);

# xticks is tickmarks on x-axis
# xticks(original_values, converted_ticks)
# converted year range as string (see list comp above)
# still need to change xlim and ylim to show more of the plot

# other parameters
# - color
# - edgecolor
# - width
# - align (of bars)
# - label (for legend)


## HORIZONTAL BAR CHARTS

# simple example
years = np.arange(2010, 2015)
values = [2, 5, 9, 5, 7]
plt.figure() # to generate a separate figure (not on top of previous one)
plt.barh(np.arange(len(years)), values)
plt.yticks(np.arange(len(years)),
           [yr for yr in years]);

# yticks are tickmarks for y-axis
# yticks(original_values, converted_ticks)
# here, year was not converted to a string

# customised example
years = np.arange(2010, 2015)
values = [2, 5, 9, 5, 7]
num_years = len(years) # 5
plt.barh(range(num_years), # years on y-axis
         values, # values on x-axis 
         color='blue',
         edgecolor='none',
         height=0.6, # thickness of bars
         align='center') # center on y-axis values
# can also add xlim here, so it's not cut off
plt.yticks(range(num_years),
           [yr for yr in years]);

# yticks are tickmarks for y-axis
# yticks(original_values, converted_ticks)
# here, year was not converted to a string

# other parameters
# - color
# - edgecolor
# - height
# - align


# BOXPLOT

df.plot(kind = 'box') # all variables
df[['col1', 'col2', 'col3']].plot(kind = 'box') # selected variables/columns


## SUBPLOTS

# Simple subplots with plt.subplot(nrows, ncols, plot_number)

dist1 = np.random.normal(42, 7, 1000)
dist2 = np.random.normal(59, 3, 1000)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1) # 1 row, 2 cols, first plot
plt.hist(dist1)
plt.title('dist1') # title of first figure

plt.subplot(1, 2, 2) # 1, 2 col, second plot
plt.scatter(dist2, dist1)
plt.xlabel('dist2')
plt.ylabel('dist1')
plt.title('Scatter Plot'); # title of second figure

# Customised subplots using ax - 2 plots side by side

dist1 = np.random.normal(42, 7, 1000)
dist2 = np.random.normal(59, 3, 1000)

# this uses axis ('ax')
fig, ax = plt.subplots(1, 2, figsize=(10, 4)) # 1 row, 2 cols

# - fig points to whole figure
# - ax is handler for list of axes
# - here, ax[0] is first plot
# - ax[1] is second plot on right

# first plot (left side)
ax[0].hist(dist1) # instead of plt.hist(dist1)
ax[0].set_title('dist1') # title of 1st figure on left

# second plot (right side)
ax[1].scatter(dist2, dist1)
ax[1].set_xlabel('dist2')
ax[1].set_ylabel('dist1')
ax[1].set_title('Scatter Plot'); # title of 2nd figure on right

# Customised subplots using ax - 2 plots stacked vertically

# You can stack them vertically
dist1 = np.random.normal(42, 7, 1000)
dist2 = np.random.normal(59, 3, 1000)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))  # 2 Rows, 1 Col

ax[0].hist(dist1)
ax[0].set_title('dist1') # title of 1st figure on top

ax[1].scatter(dist2, dist1)
ax[1].set_xlabel('dist2')
ax[1].set_ylabel('dist1')
ax[1].set_title('Scatter Plot'); # title of 2nd figure on bottom


## INTERACTIVE MODE (call this before plotting)

%matplotlib notebook
# - instead of %matplotlib inline
# - this generates an interactive window to allow you to interact with figure

dist1 = np.random.normal(42, 7, 1000)
dist2 = np.random.normal(59, 3, 1000)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].hist(dist1)
ax[0].set_title('dist1')

ax[1].scatter(dist2, dist1)
ax[1].set_xlabel('dist2')
ax[1].set_ylabel('dist1')
ax[1].set_title('Scatter Plot');
