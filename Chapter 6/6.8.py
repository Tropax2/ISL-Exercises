import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm 
from ISLP.models import ModelSpec as MS, Stepwise, sklearn_selected, poly, summarize
from functools import partial
from sklearn.model_selection import cross_validate  
from sklearn import linear_model
# Part (a)
np.random.seed(42)
X = np.random.normal(size=(100))
epsilon = np.random.normal(size=(100))

# Part (b)
Y = 2 + 3 * X - 4 * X ** 2 + 5 * X ** 3 + epsilon 

# Here \beta_0 = 2, \beta_1 = 3, \beta_2 = -4 and \beta_3 = 5 
data = pd.DataFrame(data={'X1': X, 'Y': Y})
# We add the polynomial terms to the data frame!!!
for i in range(2, 11):
    data[f'X{i}'] = X**i
 
# Part (c)
def nCp(sigma2, estimator, X, Y):
    n, p = X.shape 
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    return - (RSS + 2 * p * sigma2) / n 

design = MS([f'X{i}' for i in range(1, 11)]).fit(data)
predictors = design.transform(data)
response = np.array(data['Y'])
results = sm.OLS(response, predictors).fit()
sigma2 = results.scale
#print(summarize(results))

neg_CP = partial(nCp, sigma2)
strategy = Stepwise.first_peak(design, direction='forward', max_terms=len(design.terms)) 

data_cp = sklearn_selected(sm.OLS, strategy, scoring=neg_CP)
data_cp.fit(data, Y)
#print(data_cp.selected_state_)

'''
The results obtained are 1.8694 for intercept, 2.4471 for the X1 coefficient, -4.8594 for the X2 coef and 6.9152 for the X3 coeff.
The forward best selection returns the correct coefficients stating to only use X1, X2 and X3.
'''

# Part (d)
strategy = Stepwise.first_peak(design, direction='backward', max_terms=len(design.terms)) 

data_cp = sklearn_selected(sm.OLS, strategy, scoring=neg_CP)
data_cp.fit(data, Y)
#print(data_cp.selected_state_)
'''
Backward subset selection outputs no result.
'''

# Part (e)

model = make_pipeline(
    StandardScaler(),
    linear_model.LassoCV(cv=5)
)
model.fit(predictors, response)
alpha = model.named_steps['lassocv'].alpha_
coef = model.named_steps['lassocv'].coef_
coef_table = pd.Series(coef)
#print(coef_table)
'''
The lasso makes the intercept equal to 0 as well as the variables corresponding to X5, X6, X7, X8 and X9.
'''

# Part (f)
Y = 2 - 4 * X ** 7 + epsilon # \beta_7 is chosen to be -4 
data = pd.DataFrame(data={'X1': X, 'Y': Y})
for i in range(2, 11):
    data[f'X{i}'] = X**i

design = MS([f'X{i}' for i in range(1, 11)]).fit(data)
predictors = design.transform(data)
response = np.array(data['Y'])
results = sm.OLS(response, predictors).fit()
sigma2 = results.scale

neg_CP = partial(nCp, sigma2)
strategy = Stepwise.first_peak(design, direction='forward', max_terms=len(design.terms)) 

data_cp = sklearn_selected(sm.OLS, strategy, scoring=neg_CP)
data_cp.fit(data, Y)
#print(data_cp.selected_state_)
'''
Forward selection choosed X2, X7 and X9.
'''
X_poly = data[[f'X{i}' for i in range(1, 11)]].values
print(X_poly)
model = make_pipeline(
    StandardScaler(),
    linear_model.LassoCV(cv=5)
)
model.fit(X_poly, response)
alpha = model.named_steps['lassocv'].alpha_
coef = model.named_steps['lassocv'].coef_
coef_table = pd.Series(coef)
print(coef_table)
'''
In this case the Lasso performed poorly, like it did on the previous case because the 
predictors are highly correlated.
'''
