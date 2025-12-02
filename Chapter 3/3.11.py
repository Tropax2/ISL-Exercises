import numpy as np
import pandas as pd
import statsmodels.api as sm 
from ISLP.models import (summarize, ModelSpec as MS)

# part (a)
rng = np.random.default_rng(1)
X = rng.normal(size=100)
y = 2 * X + rng.normal(size=100)

regression = sm.OLS(y,X)
results = regression.fit()
# The coefficient is 1.9762, with standard error of 0.117, t-statistic of 16.898 and a p-value of 0.0.

# part (b)

regression2 = sm.OLS(X, y)
results2 = regression2.fit()
print(summarize(results))
# Every result is the same as the previous case.

# part (c)
# The values are exactly the same