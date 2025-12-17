import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize 
from sklearn.base import clone 
from functools import partial

Default = load_data('Default')
predictors = ['balance', 'income']
X = MS(predictors).fit_transform(Default)
Y = Default.default == 'Yes'
model = sm.GLM(Y, X, family=sm.families.Binomial())
result = model.fit()

# Part (a)
#print(summarize(result))
'''
The standard error of balance is too small to be represented, so it appears as 0.0 and the standard error 
of income is 0.000005.
'''
# Part (b)

def boot_SE(func, df, n=None, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or df.shape[0]
    for _ in range(B):
        idx = rng.choice(df.index, n, replace=True)
        value = func(df, idx)
        first_ += value 
        second_ += value**2
    return np.sqrt(second_ / B - (first_ /B)**2)

def boot_fn(model_matrix, response, df, idx):
    df_ = df.loc[idx]
    Y_ = (df_[response] == 'Yes').astype(int)
    X_ = clone(model_matrix).fit_transform(df_)
    return sm.GLM(Y_, X_, family=sm.families.Binomial()).fit().params

preds_func = partial(boot_fn, MS(['balance', 'income']), 'default')
pred_se = boot_SE(preds_func, Default, B=1000, seed=10)
print(pred_se)
