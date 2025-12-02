import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF 
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize) 

Auto = load_data('Auto')
# part (a)
pd.plotting.scatter_matrix(Auto, alpha=0.5, figsize=(9,9), diagonal='hist')
#plt.show()
# From the scatterplot matrix one may conclude that the relations between 
# displacement/weight, displacement/horsepower, horsepower/weight and horsepower/displacement
# might be linear 

# part (b)
corr_matrix = Auto.corr(method='pearson', numeric_only=True)

# part (c)
predictors = Auto.columns.drop('mpg')
predictors = MS(predictors)
X = predictors.fit_transform(Auto)
y = Auto['mpg']
regression = sm.OLS(y, X)
results = regression.fit()
# parts (i), (ii) and (iii) 
# The predictors acceleration, horsepower and cylinders are not statistically significant 
# The coefficient for the year predictor, 0.7508, suggests that if all other variables are fixed
# and the year is increased by 1, then an increase of 0.75 in mpg, which suggest (when compared to the others)
# that the year has a great influence in mpg

# part (d)
ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
#plt.show()
# The plot shows a U-shape pattern which means that multiple linear regression is not the ideal model
# Are there any high-leverage points?
infl = results.get_influence()
ax = subplots(figsize = (8,8))[1]
ax.scatter(np.arange(X.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)
#plt.show()
# We verify that there are a lot of points, since the sample size is small, that have high-leverage.
# Lets verify if there are collinear predictors
vals = [VIF(X, i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif': vals}, index=X.columns[1:])
# The VIFs of cylinders, horsepower and weight are very close to 10, with horsepower over 20
# The indicates high collinearity between the predictors, which affects the final model.

# part (e)
# Since, by the correlation matrix, a lot of the relations between variables is highly linear, some examples are cylinders/displacement, cylinder/horsepower, cylinder/weight
# displacement/horsepower, displacement/weight, etc. We can include them as interaction terms in the predictors
X = MS(['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', ('cylinders', 'displacement'), ('cylinders', 'horsepower'), ('cylinders', 'weight'), ('displacement', 'horsepower'), ('displacement', 'weight')]).fit_transform(Auto)
regression2 = sm.OLS(y, X)
results2 = regression.fit()
 
# part (f)

predictors = Auto.columns.drop('mpg')
new_predictors = Auto.apply(np.log)
new_predictors = MS(new_predictors)
X = new_predictors.fit_transform(Auto)
regression3 = sm.OLS(y, X)
results3 = regression3.fit()
print(summarize(results3))
# By applying a logarithmic transformation to the predictors, the results are even worse!