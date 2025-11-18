import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

auto = pd.read_csv(r"C:\Users\Antonio\Desktop\ISL\datasets\Auto.csv", na_values=['?'])
auto = auto.dropna()

# part (a)
"mpg, cylinders, displacement, horsepower, weight, acceleration and year are all quantitative predictors, while"
"the origin and the name are qualitative"

auto['mpg'] = auto['mpg'].astype(float) 
auto['cylinders'] = auto['cylinders'].astype(float) 
auto['displacement'] = auto['displacement'].astype(float) 
auto['horsepower'] = auto['horsepower'].astype(float) 
auto['weight'] = auto['weight'].astype(float) 
auto['acceleration'] = auto['acceleration'].astype(float) 

# part (b)
range_mpg = auto['mpg'].max() - auto['mpg'].min()
range_cylinders =  auto['cylinders'].max() - auto['cylinders'].min()
range_displacement = auto['displacement'].max() - auto['displacement'].min()
range_horsepower =  auto['horsepower'].max() - auto['horsepower'].min()
range_weight =  auto['weight'].max() - auto['weight'].min()
range_acceleration =  auto['acceleration'].max() - auto['acceleration'].min()

# part (c)
mean_mpg = auto['mpg'].mean()
mean_cylinders = auto['cylinders'].mean()
mean_displacement = auto['displacement'].mean()
mean_horsepower = auto['horsepower'].mean()
mean_weight = auto['weight'].mean()
mean_acceleration = auto['acceleration'].mean()
# Same for std 

# part (d) 
for i in range(10, 86):
    auto.drop(auto.index[i], inplace=True)

mean_mpg = auto['mpg'].mean()
mean_cylinders = auto['cylinders'].mean()
mean_displacement = auto['displacement'].mean()
mean_horsepower = auto['horsepower'].mean()
mean_weight = auto['weight'].mean()
mean_acceleration = auto['acceleration'].mean()

# part (e)
auto = pd.read_csv(r"C:\Users\Antonio\Desktop\ISL\datasets\Auto.csv", na_values=['?'])
auto = auto.dropna()

# relationship between year and the average of mpg 
mpg_by_year = auto.groupby('year')['mpg'].mean()

x_values = mpg_by_year.index.astype(int)
y_values = mpg_by_year.values
plt.bar(x_values, y_values, align='center')
plt.title("Evolution of MPG through the 70s")
plt.xlabel("Years")
plt.ylabel("MPG")
plt.show()