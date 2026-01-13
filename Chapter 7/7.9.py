import numpy as np
import pandas as pd
import statsmodels.api as sm 
from ISLP.models import ModelSpec as MS, poly, summarize, sklearn_sm, bs 
from sklearn.model_selection import cross_validate
from ISLP import load_data 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import subplots 

Boston = load_data('Boston')
Boston = Boston.dropna()
Y = Boston['nox']

# Part (a)
spec = MS([poly('dis', degree=3)])
X = spec.fit_transform(Boston)
model = sm.OLS(Y, X).fit()
#print(summarize(model))

values_to_predict = np.linspace(Boston['dis'].min(), Boston['dis'].max(), 100) 
X_pred = pd.DataFrame(data={'dis': values_to_predict})

X_transformed = spec.transform(X_pred)
prediction = model.predict(X_transformed)
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(X_pred, prediction, color='blue')
ax.scatter(Boston['dis'], Y)
ax.set_ylabel("Predicted nox")
ax.set_xlabel("Values of dis")
ax.set_title("Prediction")
#plt.show()

# Part (b)
list_of_RSS = []
for d in range(1, 11):
    spec = MS([poly('dis', degree=d)])
    X_train = spec.fit_transform(Boston)
    model = sm.OLS(Y, X_train).fit()
    Y_hat = model.predict(X_train)

    RSS = np.sum((Y - Y_hat)**2)
    list_of_RSS.append(RSS)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(X_pred, prediction, color='blue')
    ax.scatter(Boston['dis'], Y)
    ax.set_ylabel("Predicted nox")
    ax.set_xlabel("Values of dis")
    ax.set_title("Prediction")
    #plt.show()

    
print(np.argmin(list_of_RSS) + 1)
'''
We verify that the model with lowest RSS is a polynomial with degree 10
'''

# Part (c)
score_results = []
for d in range(1, 11):
    spec = MS([poly('dis', degree=d)])
    X = Boston.drop(columns='nox')
    poly_model = sklearn_sm(sm.OLS, MS([poly('dis', degree=d)]))
    cv_results = cross_validate(poly_model, X, Y, cv=5)
    cv_err = np.mean(cv_results['test_score'])
    score_results.append(cv_err)

print(np.argmax(score_results) + 1)
'''
We verify that the model with highest test score is a polynomial with degree 10
'''

# Part (d)




