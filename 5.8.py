import numpy as np
import pandas as pd 
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS, sklearn_sm, summarize
from sklearn.model_selection import cross_validate 
import matplotlib.pyplot as plt
# Part (a)
rng = np.random.default_rng(seed=1)
x = rng.normal(size=100)
y = x - 2 * x**2 + rng.normal(size=100)
'''
In this data set, n = 100 and p = 1. Given x in a normal distribution,
y is of the form x-2x^2+ epsilon, where epsilon follows a normal.
'''

# Part (b)
s = plt.scatter(x, y)
#plt.show()
'''
We observe a downward-opening parabola due to the fact that y is a quadratic polynomial
of x and the term with degree 2 has a negative coefficient.
'''

# Part (c)
df = pd.DataFrame(data={'x': x, 'y':y})
X, Y = df.drop(columns=['y']), df['y']

linear_model = sklearn_sm(sm.OLS, MS(['x']))
cv_linear = cross_validate(linear_model, X, Y, cv=X.shape[0])
cv_linear_err = np.mean(cv_linear['test_score'])
#print(cv_linear_err)
quadratic_model = sklearn_sm(sm.OLS, MS([('x', 'x'), 'x']))
cv_quadtric = cross_validate(quadratic_model, X, Y, cv=X.shape[0])
cv_quadtric_err = np.mean(cv_quadtric['test_score'])
#print(cv_quadtric_err)
cubic_model = sklearn_sm(sm.OLS, MS([('x', 'x', 'x'), ('x', 'x'), 'x']))
cv_cubic = cross_validate(cubic_model, X, Y, cv=X.shape[0])
cv_cubic_err = np.mean(cv_cubic['test_score'])
#print(cv_cubic_err)
quartic_model = sklearn_sm(sm.OLS, MS([('x', 'x', 'x', 'x') ,('x', 'x', 'x'), ('x', 'x'), 'x']))
cv_quartic = cross_validate(quartic_model, X, Y, cv=X.shape[0])
cv_quartic_err = np.mean(cv_quartic['test_score'])
#print(cv_quartic_err)

# Parts (d) and (e)
'''
We can verify that the model that has lower LOOCV estimated MSE is the quadratic one with 1.12. This should be expected
as the relationship between x and y is quadratic with a little error as the independent term.
'''

# Part (f)

# i) Linear Model
predictor = MS(['x']).fit_transform(df)
response = Y 
linear = sm.OLS(Y, predictor)
results_linear = linear.fit()
#print(summarize(results_linear))

# ii) Quadratic Model 
predictors = MS([('x','x'), 'x']).fit_transform(df)
quadratric = sm.OLS(Y, predictors)
results_quadratic = quadratric.fit()
#print(summarize(results_quadratic))

# iii) Cubic Model 
predictors = MS([('x', 'x','x'), ('x','x'), 'x']).fit_transform(df)
cubic = sm.OLS(Y, predictors)
results_cubic = cubic.fit()
#print(summarize(results_cubic))

# iv) Quartic Model 
predictors = MS([('x', 'x','x', 'x'), ('x', 'x','x'), ('x','x'), 'x']).fit_transform(df)
quartic = sm.OLS(Y, predictors)
results_quartic = quartic.fit()
#print(summarize(results_quartic))

'''
We verify that the cubic variable, as well as the intercept, are not statistically significant for 
the cubic model. 
We verify that the cubic, quartic and intercept are not statistically significant for the 
quartic model.
However, when we fit either linear or quadratic all variables are significant. This goes in line
with the LOOCV estimates for MSE as the quadratic fit is the best one for this data given the 
relationship between x and y.

'''