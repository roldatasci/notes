#### VISUALISATION (SEABORN)

# standard imports - within a .py script

from __future__ import print_function, division
import pandas as pd
import numpy as np # pandas is based on numpy
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot') # not required
import seaborn as sns # to use seaborn default

# additional imports and settings - within Jupyter notebook

from Ipython.display import Image
%matplotlib inline # enable inline plots within notebook

# loading a dataset in seaborn

titanic = sns.load_dataset("titanic")
titanic.info() # colnames and types

# restricting observation to non-null vales for variable 'age'

titanic = titanic[titanic.age.notnull()]


## HISTOGRAM

plt.hist(titanic.age)


## KERNEL DENSITY PLOT (PDF OF A FUNCTION)

# - KDE is estimation of distribution given sample
# - problem is the domain includes negative numbers
# - KDE is continuous, but data is discrete, so that's why it includes
# - values outside of the support

sns.kdeplot(titanic.age)


## CUMULATIVE DENSITY FUNCTION (CDF) PLOT

sns.kdeplot(titanic.age, cumulative=True)


## OVERLAY HISTOGRAM AND KDE WITH DISTPLOT

sns.distplot(titanic.age)


## BOX PLOT (UNIVARIATE)

sns.boxplot(titanic.age)


## BOX PLOT (MULTIVARIATE)

# - normally, you would have to split your df in matplotlib
# - seaborn can do multi box plots on same scale

## HORIZONTAL BOXPLOTS

# example: continuous var on x-axis ('age'), categorical on y-axis ('sex')
# results in horizontal boxplots (stacked on top of each other)

sns.boxplot(titanic.age, titanic.sex)


## VERTICAL BOXPLOTS
# example: categorical var on x-axis ('sex'), continuous on y-axis ('age')
# results in vertical boxplots (next to each other)

sns.boxplot(titanic.sex, titanic.age)


## VIOLIN PLOT (combines a KDE with a boxplot inside)

sns.violinplot(titanic.age) # whole dist
sns.violinplot(titanic.age, titanic.sex) # by sex, sex on y-axis


# FACET GRID (distributions by crossing two categorical variables)

# example: Distribution of Age and Survived
# Now look at survived data
# - this doesn't require splitting your df by groups
# - this example shows intersection of male and survived

g = sns.FacetGrid(titanic, row='sex', col='survived', sharex=True, sharey=True)
g.map(plt.hist, "age")

# survived males are top right figure
# top left figure just shows that there's a lot more men in general

# example: plot how many survived, by gender and class

grid_plot = sns.FacetGrid(titanic, row='sex', col='pclass')
grid_plot.map(sns.regplot, 'survived', 'age',color='.3', fit_reg=False, x_jitter=.1)

# right sides of each plot is who survived
# must add 'jitter' 
# top right shows that class 3, many did not survived

# PANDAS SCATTER MATRIX (pd.scatter())

pd.scatter(df)
pd.scatter_matrix(iris, figsize=(12,8));



## SEABORN PAIRPLOT (better than pd scatter matrix)

sns.pairplot(iris)
sns.pairplot(iris, hue="species") # specify colours based on 'species' variable


# JOINT PLOT

# univariate distributions, pearson correlaton and p-value
# - distributions shown on the axes
# - scatterplot with correlation and p-value shown on the inside

sns.jointplot("petal_width", "petal_length", kind="regplot", data=iris)

