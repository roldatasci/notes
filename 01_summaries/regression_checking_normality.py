#### CHECKING NORMALITY ASSUMPTION

# import
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

%matplotlib inline

## EXAMPLE WITH NORMALLY DISTRIBUTED DATA ## 

# Getting some normal distribution random samples
normal_samples = np.random.normal(0, 1, size=1000)

## VISUAL CHECK - HISTOGRAM ##

# Checking the Normality of our randomly drawn samples 
# -- pretty good OK to go!
plt.hist(normal_samples);

## VISUAL CHECK - NORMAL QQ PLOTS ## 

# Using the Scipy stats API to generate a normal quantile-quantile plot
# Check that indeed the normal_samples fall on a straight line
stats.probplot(normal_samples, dist="norm", plot=plt);
plt.show()

# Using statsmodel qqplot function to plot the same thing, again pretty straight line
# Normality satisfied!
normal_samples = np.random.normal(0, 1, size=1000)
sm.qqplot(normal_samples, line='s')
plt.show()


## EXAMPLE WITH UNIFORM DISTRIBUTED DATA ##

# What happens if we draw our samples from a [-1, 1] uniform distribution?
# Generate uniform samples and fact check with histogram
# -- Good to go!
uniform_samples = np.random.uniform(-1, 1, 1000)

## VISUAL CHECK - HISTOGRAM ##

plt.hist(uniform_samples,facecolor = 'r', alpha = 0.99);

## VISUAL CHECK - QQ PLOT ## 

# What would it look like on the qqplot? 
# How would it deviate from a straight line?
stats.probplot(uniform_samples, dist="norm", plot=plt);

# The uniform distribution has truncated tails compared to normal
# So we are not going to get extreme values in this case
# Hence, in the qqplot, the largest values are not as large (not as extreme)
# as we would expect if they were from normal dist.
# Smallest numbers are not as far out in the left tail either, giving rise to
# the specific curved-shape
