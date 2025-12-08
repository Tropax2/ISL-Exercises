import pandas as pd
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots 
from ISLP import load_data 
from ISLP.models import ModelSpec as MS, summarize 
from ISLP import confusion_table
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier

Weekly = load_data('Weekly')

# Part (a)
# First lets  obtain the means of Lag1-Lag5 and of Today
Lag1 = Weekly['Lag1'].mean()
Lag2 = Weekly['Lag2'].mean()
Lag3 = Weekly['Lag3'].mean()
Lag4 = Weekly['Lag4'].mean()
Lag5 = Weekly['Lag5'].mean()
# In general we notice a decreasing pattern
#print(Lag1, Lag2, Lag3, Lag4, Lag5)
# The correlation between the numerical variables is
numerical = Weekly.drop(columns=['Year', 'Direction'])
corr = numerical.corr()
# The variables don't seem very correlated, however, there is a bigger relation between Today's price and Lag1.
#print(corr)
# Now, lets examine the mean of the today's value when the price increase and decrease
up = Weekly[Weekly['Direction'] == 'Up']
down =  Weekly[Weekly['Direction'] == 'Down']
#print(np.mean(up.Today), np.mean(down.Today))
# On average, when the value of the stock goes down it goes down by 1.74% and when goes up, goes on average, 1.66%

# Part (b)
predictors = Weekly.drop(columns=['Year', 'Today', 'Direction'])
y = Weekly.Direction == 'Up'
X = MS(predictors).fit_transform(Weekly)
model = sm.GLM(y, X, family=sm.families.Binomial())
results = model.fit()
#print(summarize(results))
# Despite considerably high p-values, only two predictors seem statistically significant, namely, Lag1 and Lag2.

# Part (c)
# Lets assume, for ease, that we consider, in the prediction model, that the price will go up if the posterior probability 
# is >= 0.5; and down otherwise.
probs = results.predict()
labels = np.array(['Down'] * 1089)
labels[probs > 0.5] = 'Up'
#print(confusion_table(labels, Weekly['Direction'])) 
# We correctly predicted that the price would go down 54 times, but incorrectly 48 times. However, and here the results 
# are a lot worse, we predicted that the price would go up 430 times, when in reality it went down; we predicted correctly that the price
# would go up 557 times. 
#print(np.mean(labels == Weekly['Direction']), np.mean(labels != Weekly['Direction']))
# In general we have a correct prediction rate of 56%, and an incorrect prediction rate of 44%

# Part (d)
prior_2009, after_2009 = Weekly[Weekly['Year'] <= 2008], Weekly[Weekly['Year'] > 2008]
X_train = MS(['Lag2']).fit_transform(prior_2009)
X_test = MS(['Lag2']).fit_transform(after_2009)
Y_train = prior_2009.Direction == 'Up'
Y_test = after_2009['Direction']
model2 = sm.GLM(Y_train, X_train, family=sm.families.Binomial())
results2 = model2.fit()
#print(summarize(results2))
# The coefficient obtained for Lag2 is 0.0581 with a considerably low p-value, of 'just', 0.043
probs2 = results2.predict(X_test)
labels2 = np.array(['Down'] * 104)
labels2[probs2 > 0.5] = 'Up'
#print(confusion_table(labels2, Y_test))
#print(np.mean(labels2 == Y_test), np.mean(labels2 != Y_test))
# We predicted correctly 9 times that the price would go down, and incorrectly 34 times; 
# We predicted correctly 56 times that the price would go up, and incorrectly 5 times.
# The correct prediction rate is much higher, 62.5%, when compared to the previous part.

# Part (e)
lda = LDA(store_covariance=True)
# need to convert Y_train back to its true values instead of bool
Y_train = prior_2009['Direction']
X_train, X_test = [M.drop(columns=['intercept']) for M in [X_train, X_test]]
lda.fit(X_train, Y_train)
lda_pred = lda.predict(X_test)
#print(confusion_table(lda_pred, Y_test))
#print(np.mean(lda_pred == Y_test))
# Using LDA we predicted correctly 9 times that the price would go down, and 34 times incorrectly;
# We predicted correctly 56 times that the price would go up, and 5 times incorrectly.
# We get exactly the same results as in part (d), with a correct prediction rate of 62.5%

# Part (f)
qda = QDA(store_covariance=True)
# need to convert Y_train back to its true values instead of bool
qda.fit(X_train, Y_train)
qda_pred = qda.predict(X_test)
#print(confusion_table(qda_pred, Y_test))
#print(np.mean(qda_pred == Y_test))
# Using QDA we predicted correctly 0 times that the price would go down; and 43 incorrectly;
# We predicted correctly 61 times that the price would go down; and 0 incorrectly.
# These are some interesting results, but the correct prediction rate is lower when compared to LDA and 
# logistic, it is 58%

# Part (g)
# start by setting K=1 and fit the model
kneighbors = KNeighborsClassifier(n_neighbors=1)
kneighbors.fit(X_train, Y_train)
kneighbors_predict = kneighbors.predict(X_test)
#print(confusion_table(kneighbors_predict, Y_test))
#print(np.mean(kneighbors_predict == Y_test))
# The K-Neirest Neighbourhood method with K=1 predicted correctly 22 times that the price would go down, and incorrectly 31 times 
# It predicted correctly that the price would go up 30 times, and incorrectly 21 times
# The correct prediction rate is 50% 

# Part (h)
NB = GaussianNB()
NB.fit(X_train, Y_train)
NB_predict = NB.predict(X_test)
print(confusion_table(NB_predict, Y_test))
print(np.mean(NB_predict == Y_test))
# The Naive Bayes approach predicted correctly 0 times that the price would go down, and 43 times incorrectly
# It predicted correctly 61 times that the price would go up and 0 times incorrectly
# Its results are equal to QDA. This indicates that the class covariant matrices are diagonal and the diagonal entries are equal.
