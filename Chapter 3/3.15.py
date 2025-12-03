import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF  
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots 

Boston = load_data('Boston')
y = Boston['crim']

# part (a)
zn = MS(['zn'])
zn = zn.fit_transform(Boston)
regression_zn = sm.OLS(y, zn)
results_zn = regression_zn.fit()
### 
indus = MS(['indus'])
indus = indus.fit_transform(Boston)
regression_indus = sm.OLS(y, indus)
results_indus = regression_indus.fit()
### 
chas = MS(['chas'])
chas = chas.fit_transform(Boston)
regression_chas = sm.OLS(y, chas)
results_chas = regression_chas.fit()
### 
nox = MS(['nox'])
nox = nox.fit_transform(Boston)
regression_nox = sm.OLS(y, nox)
results_nox = regression_nox.fit()
### 
rm = MS(['rm'])
rm = rm.fit_transform(Boston)
regression_rm = sm.OLS(y, rm)
results_rm = regression_rm.fit()
### 
age = MS(['age'])
age = age.fit_transform(Boston)
regression_age = sm.OLS(y, age)
results_age = regression_age.fit()
### 
dis = MS(['dis'])
dis = dis.fit_transform(Boston)
regression_dis = sm.OLS(y, dis)
results_dis = regression_dis.fit()
### 
rad = MS(['rad'])
rad = rad.fit_transform(Boston)
regression_rad = sm.OLS(y, rad)
results_rad = regression_rad.fit()
### 
tax = MS(['tax'])
tax = tax.fit_transform(Boston)
regression_tax = sm.OLS(y, tax)
results_tax = regression_tax.fit()
### 
ptratio = MS(['ptratio'])
ptratio = ptratio.fit_transform(Boston)
regression_ptratio = sm.OLS(y, ptratio)
results_ptratio = regression_ptratio.fit()
### 
lstat = MS(['lstat'])
lstat = lstat.fit_transform(Boston)
regression_lstat = sm.OLS(y, lstat)
results_lstat = regression_lstat.fit()
### 
medv = MS(['medv'])
medv = medv.fit_transform(Boston)
regression_medv = sm.OLS(y, medv)
results_medv = regression_medv.fit()

# The models for which there is a statistically significant relationship between the predictor and the response are 
# zn, indus, nox, rm, age, dis, rad, tax, ptratio, lstat and medv. In other words, the only one where there is not a 
# statistically significant relationship is between chas and crim.

# part (b)
predictors = MS(['zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat', 'medv'])
predictors = predictors.fit_transform(Boston)
regression = sm.OLS(y, predictors)
results = regression.fit()
# We verify that the predictors indus, chas, rm, age, tax, ptratio and lstat have very low t-statistic, with absolute value 
# less than 2, and consequently, big p-values. This means that these predictors are not statistcally significant. We should not 
# reject H_0 for all this predictors. 

# part (c)
# The results can be partly explained by the existence of some collinearity between the variables, where the variables specified 
# in (b) offer no additional information and so they become irrelevant. When we do just simple linear regression, then information 
# comes from just one variable, as well as the information from other where collinearity exists, and so they become relevant for
# the study.

# part (d)

zn = MS(['zn', ('zn', 'zn'), ('zn','zn','zn')])
zn = zn.fit_transform(Boston)
regression_zn = sm.OLS(y, zn)
results_zn = regression_zn.fit()
print(summarize(results_zn))
### 
indus = MS(['indus', ('indus','indus'), ('indus','indus','indus')])
indus = indus.fit_transform(Boston)
regression_indus = sm.OLS(y, indus)
results_indus = regression_indus.fit()
print(summarize(results_indus))
### 
chas = MS(['chas',('chas','chas'), ('chas','chas','chas')])
chas = chas.fit_transform(Boston)
regression_chas = sm.OLS(y, chas)
results_chas = regression_chas.fit()
print(summarize(results_chas))
### 
nox = MS(['nox', ('nox','nox'), ('nox','nox','nox')])
nox = nox.fit_transform(Boston)
regression_nox = sm.OLS(y, nox)
results_nox = regression_nox.fit()
print(summarize(results_nox))
### 
rm = MS(['rm', ('rm', 'rm'), ('rm','rm','rm')])
rm = rm.fit_transform(Boston)
regression_rm = sm.OLS(y, rm)
results_rm = regression_rm.fit()
print(summarize(results_rm))
### 
age = MS(['age', ('age','age'), ('age','age','age')])
age = age.fit_transform(Boston)
regression_age = sm.OLS(y, age)
results_age = regression_age.fit()
print(summarize(results_age))
### 
dis = MS(['dis', ('dis', 'dis'), ('dis', 'dis', 'dis')])
dis = dis.fit_transform(Boston)
regression_dis = sm.OLS(y, dis)
results_dis = regression_dis.fit()
print(summarize(results_dis))
### 
rad = MS(['rad', ('rad','rad'), ('rad','rad','rad')])
rad = rad.fit_transform(Boston)
regression_rad = sm.OLS(y, rad)
results_rad = regression_rad.fit()
print(summarize(results_rad))
### 
tax = MS(['tax', ('tax', 'tax'), ('tax', 'tax', 'tax')])
tax = tax.fit_transform(Boston)
regression_tax = sm.OLS(y, tax)
results_tax = regression_tax.fit()
print(summarize(results_tax))
### 
ptratio = MS(['ptratio', ('ptratio', 'ptratio'), ('ptratio', 'ptratio', 'ptratio')])
ptratio = ptratio.fit_transform(Boston)
regression_ptratio = sm.OLS(y, ptratio)
results_ptratio = regression_ptratio.fit()
print(summarize(results_ptratio))
### 
lstat = MS(['lstat', ('lstat', 'lstat'), ('lstat','lstat','lstat')])
lstat = lstat.fit_transform(Boston)
regression_lstat = sm.OLS(y, lstat)
results_lstat = regression_lstat.fit()
print(summarize(results_lstat))
### 
medv = MS(['medv', ('medv', 'medv'), ('medv', 'medv', 'medv')])
medv = medv.fit_transform(Boston)
regression_medv = sm.OLS(y, medv)
results_medv = regression_medv.fit()
print(summarize(results_medv))

# There is a non-linear, cubic, association between indus and crim, nox and crim, dis and crim, ptratio and crim, and medv and crim