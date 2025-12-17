import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data 
from ISLP.models import ModelSpec as MS, summarize 

Weekly = load_data('Weekly')

# Part (a)
predictors = ['Lag1', 'Lag2']
X = MS(predictors).fit_transform(Weekly)
Y = Weekly.Direction == "Up"
model = sm.GLM(Y, X, family=sm.families.Binomial())
result = model.fit()
#print(summarize(result))

# Part (b)
Weekly_except_first_obs = Weekly.iloc[1:Weekly.shape[0]]
Weekly_first_obs =  Weekly.iloc[0:1]
spec = MS(predictors)
X_train = spec.fit_transform(Weekly_except_first_obs)
Y_train = (Weekly_except_first_obs.Direction == "Up")
X_test = spec.transform(Weekly_first_obs)
model2 = sm.GLM(Y_train, X_train, family=sm.families.Binomial())
result2 = model2.fit()
#print(summarize(result2))

# Part (c)
prob = result2.predict(X_test)
label = np.array(['Down'])
label = prob > 0.5
# The observation was not correctly classified.

# Part (d)
n = 0
for i in range(Weekly.shape[0]):
    Weekly_except_one_obs = Weekly.drop(index=i)
    spec = MS(predictors)
    X_train = spec.fit_transform(Weekly_except_one_obs)
    Y_train = (Weekly_except_one_obs.Direction == "Up")
    Weekly_one_obs =  Weekly.iloc[i:i+1]
    X_test = spec.transform(Weekly_first_obs)
    model_i = sm.GLM(Y_train, X_train, family=sm.families.Binomial())
    results_i = model_i.fit()
    prob_i = results_i.predict(X_test)
    label_i = prob_i.iloc[0] > 0.5 
    if label_i == True and Weekly.loc[i, 'Direction'] == 'Up':
       n += 1
   
# Part (e)
print(f'The success rate is {n / Weekly.shape[0]}')
# The success rate of this method is of 55.5%


