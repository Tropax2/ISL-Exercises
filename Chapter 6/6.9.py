import numpy as np
import pandas as pd
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression 
from sklearn.pipeline import make_pipeline 
import statsmodels.api as sm

College = load_data('College')
College.dropna()
College['Private'] = College['Private'].astype("category")
X = College.drop(columns='Apps')
predictors = MS(X.columns).fit_transform(College)
response = College['Apps']
# Part (a)
X_train, X_test, Y_train, Y_test = train_test_split(predictors, response, test_size=0.2, shuffle=True, random_state=42)

# Part (b)
model = sm.OLS(Y_train, X_train)
results = model.fit()
predicted = results.predict(X_test)
tolerance = 250 
# Since we are using linear regression to estimate apps, we let a tolerance of 250, which seems reasonable having in mind the types of data
within_tolerance = np.abs(Y_test - predicted) < tolerance 
#print(np.mean(within_tolerance)) 

# Part (c)
numeric_cols = X.drop(columns='Private').astype(float)
enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
private_encoded = enc.fit_transform(X[['Private']])
X_encoded = np.hstack([numeric_cols.values, private_encoded])
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, response, test_size=0.2, shuffle=True, random_state=42)

model = make_pipeline(StandardScaler(), RidgeCV(fit_intercept=True))
model.fit(X_train, Y_train)
alpha = model.named_steps['ridgecv'].alpha_
coef = model.named_steps['ridgecv'].coef_
predicted = model.predict(X_test)
within_tolerance = np.abs(Y_test - predicted) < tolerance 
#print(np.mean(within_tolerance))
# Ridge regression performs slightly better than normal linear regression

# Part (d)
model = make_pipeline(StandardScaler(), LassoCV(fit_intercept=True, random_state=42))
model.fit(X_train, Y_train)
alpha = model.named_steps['lassocv'].alpha_
coef = model.named_steps['lassocv'].coef_
coef = pd.Series(coef)
predicted = model.predict(X_test)
within_tolerance = np.abs(Y_test - predicted) < tolerance 
print(coef)
#print(np.mean(within_tolerance))
# lasso regression performs slightly better than ridge
