import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS, summarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Auto = load_data('Auto')
# Part (a)
mpg01 = lambda x: 1 if x > Auto['mpg'].median() else 0 
Auto['mpg01'] = Auto['mpg'].apply(mpg01)

# part (b)
# Lets try to find the correlation matrix between all the variables 
#print(Auto.corr())
# It seems that there exists considerable correlation between mpg01 and mpg, cylinders, weight and displacement 
pd.plotting.scatter_matrix(Auto, figsize=(10,10), diagonal='hist')
#plt.show()

# Part (c)
'''
We split the data into a training set and test set, where the training data consists of 80% of all data, with 40% 
assigned to each value of mpg01; and the remaining 20 % will be the testing set. The values of the chosen rows below 
 match this criterion.
 '''
Auto = Auto.sort_values(by=['mpg'], ascending=True)

predictors = ['mpg', 'cylinders', 'weight', 'displacement']
X = Auto[predictors]
Y = Auto['mpg01']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y , test_size=0.2, stratify=Y, random_state=42, shuffle=True
)

# Part (d)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
lda = LDA(store_covariance=True)
lda.fit(X_train, Y_train)
pred = lda.predict(X_test)
#print(confusion_table(pred, Y_test))
#print(np.mean(pred == Y_test))
'''
By using the approach of using 80% of the data as training and scalling the predictors, we obtain that the LDA 
predicted 0 correctly 37 times and 3 times incorrectly. LDA predicted correctly 1 every time. The success rate is 
of 96%!
'''

# Part (e)
qda = QDA(store_covariance=True)
qda.fit(X_train, Y_train)
pred = qda.predict(X_test)
#print(confusion_table(pred, Y_test))
#print(np.mean(pred == Y_test))
'''
QDA predicted 0 correctly 38 times and 2 times incorrectly. LDA predicted correctly 1 every time. The success rate is 
of 97%!
'''

# Part (f)
Auto = load_data('Auto')
mpg01 = lambda x: 1 if x > Auto['mpg'].median() else 0 
Auto['mpg01'] = Auto['mpg'].apply(mpg01)
Y = Auto.mpg01 == 1
X = MS(predictors).fit_transform(Auto)
model = sm.GLM(Y,X, sm.families.Binomial())
results = model.fit()
#print(summarize(results))
'''
The p-values associated with each predictors are very high, hence the logistic regression is not a good model 
to make predictions on this data set. This might be an indicator that the parametric model obtained by the method 
is not adequate.
'''
probs = results.predict()
labels = np.array([0] * 392) 
labels[probs > 0.5] = 1
#print(confusion_table(labels, Auto['mpg01']))
'''
Logistic model gets the 100% of the predictions correct if we use a threshold of 0.5 as the probability.
This means that the predictors are not relevant for the model to make predictions. This may be due to the fact 
that the decision boundary is a straight vertical line and hence the maximum of the likelihood function do not exist, tending 
to infinity.
'''

# Part (g)
clf = GaussianNB()
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
#print(confusion_table(pred, Y_test))
#print(np.mean(pred == Y_test))
'''
Naive Bayes falls a little behind QDA and LDA, of having a correct prediction rate of 92%, guessing correctly 0 34 times, 
and failing 6 times, and predicting 1 correctly all 39 times. The inferior rate may be atributed to the fact the predictors
are not sufficiently independ, as NAive Bayes assumes, to beat the other methods.
'''

# Part (h)
neigh = KNeighborsClassifier(n_neighbors=200)
neigh.fit(X_train, Y_train)
pred = neigh.predict(X_test)
print(confusion_table(pred, Y_test))
print(np.mean(pred == Y_test))
'''
KNN classifier with K = 1 predicted 0 correctly every time, but failed to predict 1 correctly 2 times. Its success rate is of 97%
For K=3 the results are the same. For K = 10 it guesses 0 incorrectly 1 time, which makes the success rate to drop.
'''