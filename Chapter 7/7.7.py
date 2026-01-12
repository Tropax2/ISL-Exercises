import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data  
from ISLP.models import ModelSpec as MS, summarize, poly, bs, ns
import matplotlib.pyplot as plt 
from matplotlib.pyplot import subplots 

Wage = load_data('Wage')
Wage.dropna()
print(Wage)

spec = MS(['maritl', poly('age', degree=3)]) 
X = spec.fit_transform(Wage)
Y = Wage['wage']
model = sm.OLS(Y, X).fit()
#print(summarize(model))

'''
We verify that marital status is not that relevant to the wage response variable as well as the fourth degree polynomial.
Third degree is statisticall more significant.
'''

bspline = MS(['maritl', bs('age', df=4)])
Xbs = bspline.fit_transform(Wage)
model_spline = sm.OLS(Y, Xbs).fit()
print(summarize(model_spline))

nspline = MS(['maritl', ns('age', df=4)])
Xns = nspline.fit_transform(Wage)
model_nspline = sm.OLS(Y, Xns).fit()
print(summarize(model_nspline))

'''
The results obtained either from bsplies or natural splines are very similar. The 4th basis function is not statistically
significant.
'''

################## 


spec2 = MS(['jobclass', poly('age', degree=3)]) 
X = spec2.fit_transform(Wage)
Y = Wage['wage']
model = sm.OLS(Y, X).fit()
#print(summarize(model))

'''
We verify that jobclass has a very significant relation with the wage response variable.
'''

bspline = MS(['jobclass', bs('age', df=3)])
Xbs = bspline.fit_transform(Wage)
model_spline = sm.OLS(Y, Xbs).fit()
#print(summarize(model_spline))

nspline = MS(['jobclass', ns('age', df=3)])
Xns = nspline.fit_transform(Wage)
model_nspline = sm.OLS(Y, Xns).fit()
#print(summarize(model_nspline))

