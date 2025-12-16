import numpy as np
import pandas as pd 
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression as LR
from ISLP import confusion_table

Default = load_data('Default')
predictors = ['balance', 'income']

# Part (a)
X = MS(predictors).fit_transform(Default) 
Y = Default.default == 'Yes'
model = sm.GLM(Y, X, family=sm.families.Binomial())
results = model.fit()
#print(summarize(results))

# Part (b)
# Divide the data set into 80% training and 20% testing 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
# Fit the model into the training data
model2 = sm.GLM(Y_train, X_train, family=sm.families.Binomial())
results2 = model2.fit()
#print(summarize(results2))
# Predictions 
probs = results2.predict(X_test)
labels = np.array(['No'] * X_test.shape[0])
labels = probs > 0.5
#print(confusion_table(labels, Y_test))
#print(np.mean(labels != Y_test))
'''
We verify that the logistic model predicted correctly that an individual would not default 1921 times and incorrectly 51 times.
It predicted correctly that an individual would default 18 times and incorrectly 10 times.
Out of the 2000 observations, the error is of 3.05%.
''' 

# Part (c)
'''
By increasing the test_size to 0.3 we obtain that the error rate is now of 2.67%. By increasing it to 0.4, the 
error actualy increases to 2.73%.
'''

# Part (d)
new_predictors = ['student', 'balance', 'income']
X = MS(new_predictors).fit_transform(Default) 
# Divide the data set into 80% training and 20% testing 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42)
# Fit the model into the training data
model2 = sm.GLM(Y_train, X_train, family=sm.families.Binomial())
results2 = model2.fit()
# Predictions 
probs = results2.predict(X_test)
labels = np.array(['No'] * X_test.shape[0])
labels = probs > 0.5
print(confusion_table(labels, Y_test))
print(np.mean(labels != Y_test))
'''
We now incorrectly predict that one more individual will not default and correctly predict that one more 
individual will default. The error rate is still at 3.05%. Hence, the student predictor is not relevant 
for this particular model.
'''