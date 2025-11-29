import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF 
from statsmodels.stats.anova import anova_lm 
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, poly, summarize)

Auto = load_data('Auto')

# part (a)

predictor = MS(['horsepower'])
X = predictor.fit_transform(Auto)
y = Auto['mpg']
model = sm.OLS(y, X)
results = model.fit()

# The T-statistic is -24.489 with a p-value of 0, which suggests a strong relationship. 
# Since |T-statistic| > 2, we conclude that the relationship is very strong.
# Since the coefficient associated with mpg is negative, the relationship is negative. 

value_to_predict = pd.DataFrame({"horsepower":[98]})
newX = predictor.transform(value_to_predict)
prediction = results.get_prediction(newX)

# The predicted value is 24.47 mpg
# The 95% confidence interval is [23. 97, 24.96]
# The prediction interval is [14.81, 34.12]

# part (b)
ax = subplots(figsize = (8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
plt.show()

# The existence of a U-shape pattern indicates a strong non-linearity between the data
# In fact, it matches the plot presented on Section 3.3.3 

# part (c)
infl = results.get_influence()
ax = subplots(figsize = (8,8))[1]
ax.scatter(np.arange(X.shape[0]),infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)

# The number with leverage greater than (p+1)/ n = 0.05 is very high, that might explain the poor fit
# Simply the relationship between the two variables is non-linear