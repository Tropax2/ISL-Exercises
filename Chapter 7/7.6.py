import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data  
from ISLP.models import ModelSpec as MS, summarize, poly 
from sklearn.model_selection import cross_validate 
from ISLP.models import sklearn_sm 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import subplots 

# Part (a)
Wage = load_data('Wage')
X = Wage.drop(columns=['wage'])
Y = Wage['wage']
for d in range(1,6):
    model = sklearn_sm(sm.OLS, MS([poly('age', degree=d)]))
    cv = cross_validate(model, X, Y) # 5-fold cross-validation
  #  print(f'For d = {d}: {np.mean(cv['test_score'])}')
'''
We verify, without surprise, that for test MSE is smaller for d= 3 and d = 4
'''
# Plot for d = 4 
d = 4 
spec = MS([poly('age', degree=4)])
X = spec.fit_transform(Wage)
model = sm.OLS(Y, X).fit()

values_to_predict = np.linspace(Wage['age'].min(), Wage['age'].max(), 100)
X_pred = pd.DataFrame(data={'age': values_to_predict})
x_pred_transformed = spec.transform(X_pred)
predictions = model.predict(x_pred_transformed)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(Wage['age'], Y, alpha=0.3, label='Observed data')
ax.plot(X_pred, predictions, color='red', label='Polynomial of degree 4')
ax.set_xlabel('Age')
ax.set_ylabel('Wage')
ax.set_title('Polynomial regression of degree 4')
ax.legend()
#plt.show()

# Part (b)
X = Wage.drop(columns=['wage'])
Y = Wage['wage']

max_cuts = 10
cv_errors = []

for k in range(1, max_cuts+1):
    bin_labels = [f"bin{i}" for i in range(k)]
    Wage[f'age_bin_{k}'] = pd.cut(Wage['age'], bins=k, labels=bin_labels)
    X_step = pd.get_dummies(Wage[f'age_bin_{k}'], drop_first=True)

    spec = MS(list(X_step.columns))
    model = sklearn_sm(sm.OLS, spec)

    cv = cross_validate(model, X_step, Y) 
    cv_errors.append(np.mean(cv['test_score']))  

print(np.argmin(cv_errors) + 1)