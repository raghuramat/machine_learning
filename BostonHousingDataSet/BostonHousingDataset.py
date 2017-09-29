import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model as lm
from sklearn import model_selection as ms
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy, r2_score
from sklearn import preprocessing
from sklearn import utils


# This exercise is to determine a correlation between various data points available for Boston and the median value
# of a property

# The code piece below helps load the data set to a data frame.
# Once that's done, we merge the data-sets into one single data frame to contain both target and data points
# Add target to the original data-frame so that the data-frame can be analysed

boston = load_boston()
df = pd.DataFrame(boston.data)
y = boston.target
df.columns = boston["feature_names"]
df['MEDIAN_PRICE'] = pd.Series(list(y))
print(df.head())
print(df.describe())
print(df.info())

# Columns in the data set
# 'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDIAN_PRICE'

# The below piece of code helps visualize correlations between target and various data points
# plt.interactive(True)
# plt.violinplot(y, showmedians=True)
# sns.pairplot(df, x_vars=['CRIM','ZN','INDUS','CHAS'], y_vars=['MEDIAN_PRICE'])
# sns.pairplot(df, x_vars=['NOX','RM','AGE','DIS'], y_vars=['MEDIAN_PRICE'])
# sns.pairplot(df, x_vars=['RAD','TAX','PTRATIO','B','LSTAT'], y_vars=['MEDIAN_PRICE'])
# sns.pairplot(df, x_vars=['RM'], y_vars=['MEDIAN_PRICE'])
# sns.pairplot(df, x_vars=['LSTAT'], y_vars=['MEDIAN_PRICE'])
# sns.pairplot(df, x_vars=['PTRATIO'], y_vars=['MEDIAN_PRICE'])
# plt.show(block=True)


# The above plots show some sense of correlations between LSTAT and MEDIAN_PRICE for sure. On closer inspection,
# RM also shows a correlation with the target
# Observations :
# Linear relationship between RM and MEDIAN_PRICE, i.e., MEDIAN_PRICE increases with RM increasing
# Non linear Inverse relationship between LSTAT and MEDIAN_PRICE; mp decreases with RM increase
# Scattered relationship between MEDIAN_PRICE AND PTRATIO

# Figure out the correlation between different data points.
# print(df.corr())

# correlation in absolute terms with a greater than 65% correlation check confirms RM and LSTAT as variables to look at
# print(df.corr().applymap(lambda x: '' if abs(x) < 0.60 else abs(int(100 * x))))

# First method is to arrive as a fit through brute force.
# The correlation between LSTAT and target appears to be non-linear but one between RM and target appears linear.
# Try different combinations to figure out what works.

# print(np.corrcoef(df['LSTAT'], df['MEDIAN_PRICE']))  # 0.73
# print(np.corrcoef(1/(4 + df['LSTAT']), df['MEDIAN_PRICE']))  # 0.821
# print(np.corrcoef(np.power(df['RM'], 3), np.power(df['MEDIAN_PRICE'])))  # 0.75
# print(np.corrcoef(np.power(df['RM'], 4), np.power(df['MEDIAN_PRICE'], 1)))

# ax = sns.regplot(1/(4 + df['LSTAT']), df['MEDIAN_PRICE'])
# ax.set(xlabel='1 / (4 + LSTAT)')
# ax.set(ylabel='MEDIAN_PRICE')
# plt.show(block=True)

# the above is what could be the best accuracy levels achieved. Only relation with lstat
# we will use sklearn below to find out from different regression models
#


# def rmse(predicted, targets):
#     return np.sqrt(np.mean((targets-predicted)**2))
#
#
dfX = df[['RM','LSTAT','PTRATIO']]
dfY = df['MEDIAN_PRICE']
dfLStat = df[['LSTAT']]
# print(dfX.shape)

# Once that fit is obtained, we do a split on data for train and test
X_train, X_test, y_train, y_test = ms.train_test_split(dfLStat, dfY, random_state=13)
# # Below is a trial through linear regression. Limiting input params to RM, LSTAT AND PTRATIO
# skreg = lm.LinearRegression()
# skreg.fit(X_train,y_train)
# y_pred = skreg.predict(X_test)
#
# print("coeff/intercept:", skreg.coef_, skreg.intercept_)
# print("score on train:", skreg.score(X_train, y_train))
# print("score on test:", skreg.score(X_test, y_test))
# print("rmse: ", rmse(y_pred, y_test))

# coeffs = skreg.coef_[:]
# intercept = skreg.intercept_

# trying k-fold validation on linear regression with varying k
# k=5
# kscores = ms.cross_val_score(skreg, X_train, y_train, cv=5)
# print(kscores)
# print("mean score: {0:.3f} std dev: {1:.3f}".format(kscores.mean(), kscores.std()))

# trying regularization with alpha = 0.5

# ridge = lm.Ridge(alpha=0.5)  # alpha is the new lambda
# ridge.fit(X_train, y_train)
# y_pred = ridge.predict(X_test)
# print(rmse(y_pred, y_test))
# print(ridge.coef_, ridge.intercept_)
