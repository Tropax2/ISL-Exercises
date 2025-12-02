import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.pyplot import subplots
from ISLP.models import (ModelSpec as MS, summarize)

# part (a)
np.random.seed(1)
X = np.random.standard_normal(100)

# part (b)
eps = np.random.normal(0, 0.25, 100)
 
# part (c)
Y = -1 + 0.5 * X + eps 
# The vector Y has length 100 
# The values o \beta_0 and \beta_1 in this linear model are -1 and 0.5, respectively 

# part (d)
ax = subplots(figsize=(8,8))[1]
ax.scatter(X, Y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#plt.show()
# The relationship between X and Y seems positively linear.

# part (e)
predictors = pd.DataFrame({'X': X})
X_train = MS(['X']).fit_transform(predictors)
regression = sm.OLS(Y, X_train)
results = regression.fit()
# Both the intercept and the X coordinate are very close to the original values; this might occur due to part (d), where,
# just by looking at the plot, the relationship between X and Y seemed linear.

# part (f)
Y_pred = results.predict(X_train)
fig, ax = plt.subplots(figsize=(8, 8))
# original 
ax.scatter(X, Y, color='blue', label='Original')
# predicted 
ax.scatter(X, Y_pred, color='red', label='Predicted')
ax.legend()
plt.show()

# part (g)
X_train_2 = MS(['X', ('X','X')]).fit_transform(predictors)
regression2 = sm.OLS(Y, X_train_2)
results2 = regression2.fit()
# There is no evidence that a quadratic term improves the model, in fact it might
# make it worse, since it has a very low t-statistic, and consequently, a very high p-value

# part (j)
print(results.conf_int(alpha=0.05))