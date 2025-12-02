import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (summarize, ModelSpec as MS)
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots  

Carseats = load_data('Carseats')

# part (a)
y = Carseats['Sales']
predictors = MS(['Price', 'Urban', 'US'])
X = predictors.fit_transform(Carseats)
regression = sm.OLS(y,X)
results = regression.fit()
print(results.summary())
# part (b)
# The Price predictor is quantitative and its coefficient is -0.0545, so an increase 
# in one unit of price results in less 0.0545 sales. The Urban and US predictors 
# are qualitative and so, if a car is a Urban, then it results in less -0.0219 sales (HIGH p-value). 
# However, if a car is US then it results in more 1.2 sales.

# part (c)
# In equation form it would be Y = 13.0435 - 0.0545*Price -0.0219*Urban[Yes] + 1.2006*US[Yes], where 
# Urban[Yes] = 1 if it is Urban; 0 otherwise; and US[Yes] = 1 if it is made in the US; 0 otherwise

# part (d)
# We can reject the null hypothesis for the predictors Price and US[Yes], however not for Urban since it has a p-value of 0.936 

# part (e)
predictors2 = MS(['Price', 'US'])
X = predictors2.fit_transform(Carseats)
regression2 = sm.OLS(y,X)
results2 = regression2.fit()
 
# part (f)
# Both models do not perform very well, since they have an R^2 of 0.239, which could indicate that the relationship between 
# the predictors and the target is not linear

# part (g)
#print(results2.conf_int(alpha=0.05))

# part (h)
infl = results2.get_influence()
ax = subplots(figsize = (8,8))[1]
ax.scatter(np.arange(X.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)
# A lot of points can be considered to be high leverage point since their leverage is above (p+1)/n = 0.007
ax = subplots(figsize = (8,8))[1]
ax.scatter(results2.fittedvalues, results2.resid)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.show()
# There are also a lot of points with considerable residuals, meaning the existence of outliers