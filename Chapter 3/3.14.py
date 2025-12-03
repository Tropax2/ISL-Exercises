import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF 
from ISLP.models import (ModelSpec as MS, summarize) 

# part (a)
rng = np.random.default_rng(10)
x1 = rng.uniform(0, 1, size=100)
x2 = 0.5 * x1 + rng.normal(size=100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)
# The form of the linear model is Y = \beta_0 + \beta_1 * x1 + \beta_2 * x2 
# The coefficients are \beta_0 = 2, \beta_1 = 2 and \beta_2 = 0.3

# data treatment for regression on part (c)
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
y = data['y']
predictors = MS(['x1', 'x2'])
X = predictors.fit_transform(data)

# part (b)
corr = data.corr(method='pearson')
# the correlation between x1 and x2 is 0.77723
pd.plotting.scatter_matrix(data)
 
# part (c) 
regression = sm.OLS(y, X)
results = regression.fit()
# \beta_0 = 1.9579, \beta_1 = 1.6154, \beta_2 = 0.9428
# \beta_0 is very close to the true value, while \beta_1 is a little far off and \beta_2 is even more.
# There is evidence to reject \beta_1 = 0, since the p-value is 0.003 and is some evidence to not reject \beta_2 = 0
# since the t-statistic is just of 1.134 with associated p-value of 0.259.

# part (d)
predictors = MS(['x1'])
X = predictors.fit_transform(data)
regression = sm.OLS(y, X)
results = regression.fit()
# By using just x1 as predictor we get an estimate much closer to the real value of \beta_1
# We have evidence to reject H_0: \beta_1 = 0

# part (e)
predictors = MS(['x2'])
X = predictors.fit_transform(data)
regression = sm.OLS(y, X)
results = regression.fit()
# By using just x1 as predictor we get an estimate a lot worse when compared to the real value of \beta_1
# We have evidence to reject H_0: \beta_2 = 0

# part (f)
'''
The results seem to contradict each other, since in part (c) we should not reject H_0:\beta_2 = 0, and in part (f)
we should. However, in part (c) the coefficient associated with x2 is a lot closer to the real value, than it is on part (e)
This could be explained either by the existence of high leverage points with x2 coordinate or just that the predictors 
are colinear and hence by rejecting one or the other we get better t-statistics and p-values, but in the case of x2, since 
the relation is not linear we obtain worse results.
Other possible explanations come from the fact that x1 and x2 are highly correlated. So when we use both predictors, 
x2 offers no new information that is not in x1. However, when modeling just x2 it has a significance because it also 
carries information about x1. This seems to be the case due to the results below.
'''
predictors = MS(['x1', 'x2'])
X = predictors.fit_transform(data)
vals = [VIF(X, i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif': vals}, index=X.columns[1:])
# the VIF is 2.478223 so it is not very high. Collinearity seems not to be the problem.
# Lets see the leverage.
infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)
# We also don't have a lot of points with high leverage.

