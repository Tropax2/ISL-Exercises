import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# part (a)
df = pd.read_csv(r"C:\Users\Antonio\Desktop\ISL\datasets\Boston.csv")

# part (b)
print(df)

# part (c)
df.plot.scatter(x='lstat', y='crim')
plt.xlabel("low-stat population")
plt.ylabel("Crime rate")
plt.show()

df.plot.scatter(x='chas', y='nox')
plt.xlabel('is on Charles River')
plt.ylabel('Nitric oxides concentration')
plt.show()

df.plot.scatter(x='tax', y='crim')
plt.xlabel('Tax')
plt.ylabel('Crime rate')
plt.show()

#part (d)
# There is a visible relationship between crime-rate and Full-value property-tax rate per $10,000. The suburbs where tax rate reaches 666 also have some of the highest crime rate.

# part (e)
print(df.sort_values('crim', ascending=False))
print(df.sort_values('tax', ascending=False))
print(df.sort_values('ptratio', ascending=False))

# part (f)
print(df.value_counts('chas'))
# 35 suburbs bound the Charles River.

# part (g)
ptratio_median = df['ptratio'].median()
print(ptratio_median)
# The median of pupil-teacher ratio among the towns in this data set is 19.05

# part (i)
print(df.value_counts(df['rm'] > 7))
# 64 suburbs average more than 7 rooms per dwelling
print(df.value_counts(df['rm'] > 8))
# 13 suburbs average more than 8 rooms per dwelling
 