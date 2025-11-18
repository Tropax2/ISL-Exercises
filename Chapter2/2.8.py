import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from pandas.plotting import scatter_matrix

# part (a)
college = pd.read_csv(r"C:\Users\Antonio\Desktop\ISL\datasets\College.csv")

# part (b)
college3 = college.rename({"Unnamed: 0": "College"}, axis=1)
college = college3

# part(c)
print(college.describe())

# part(d)
data = ['Top10perc', 'Apps', 'Enroll']
pd.plotting.scatter_matrix(
    college[data],
    alpha=0.5,
    figsize=(10, 10)
    )
plt.show()

# part (e)
data_outstate_private = college.loc[college['Private'] == 'Yes', 'Outstate']
data_outstate_noprivate = college.loc[college['Private'] == 'No', 'Outstate']

data_plot = [data_outstate_private, data_outstate_noprivate]

plt.boxplot(data_plot)
plt.title("Outstate Expenses by Private or Non-Private School")
plt.ylabel("Expenses")
plt.xlabel("School Type")
plt.xticks(ticks=[1, 2], labels=['Private', 'Non-Private'])
plt.show()

# part (f)
college['Elite'] = pd.cut(college['Top10perc'], bins=[0, 50, 100], labels=['No', 'Yes'])
print(college["Elite"].value_counts())

data_outstate_elite = college.loc[college['Elite'] == 'Yes', 'Outstate']
data_outstate_noelite = college.loc[college['Elite'] == 'No', 'Outstate']
data_plot = [data_outstate_elite , data_outstate_noelite]

plt.boxplot(data_plot)
plt.title("Outstate Expenses by Elite or Non-Elite School")
plt.ylabel("Expenses")
plt.xlabel("School Type")
plt.xticks(ticks=[1, 2], labels=['Elite', 'Non-Elite'])
plt.show()

# part(g)
college['Expenses'] = pd.cut(college['Expend'], bins=[0, 7000, 18000, 50000, 150000], labels=['Low', 'Medium', 'High', 'Very High'])
data_expenses_private = college.loc[college['Private'] == 'Yes', 'Expenses']
data_expenses_noprivate = college.loc[college['Private'] == 'No', 'Expenses']
plt.hist(data_expenses_noprivate)
plt.title("Distribution of Expenses in Non-Private Universities")
plt.ylabel("Number of Universities")
plt.xlabel("Type of Expenses")
plt.show()
 
