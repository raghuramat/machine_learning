import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston = load_boston()
df = pd.DataFrame(boston.data)
y = boston.target
df.columns = boston["feature_names"]

# df.head()
# print(boston["DESCR"])
# df.info()
# Add target to the original dataframe so that the dataframe can be analysed

df['target'] = pd.Series(list(y))

#
# 'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','target'

plt.interactive(True)
# plt.violinplot(y, showmedians=True)
# sns.pairplot(df, x_vars=['CRIM','ZN','INDUS','CHAS'], y_vars=['target'])
# sns.pairplot(df, x_vars=['NOX','RM','AGE','DIS'], y_vars=['target'])
# sns.pairplot(df, x_vars=['RAD','TAX','PTRATIO','B','LSTAT'], y_vars=['target'])
# sns.pairplot(df, x_vars=['LSTAT'], y_vars=['target'])
plt.show(block=True)

# print(df.corr())

# Figure out the correlation in absolute terms. First method is to arrive as a fit
# through brute force. Once that fit is obtained, we do a split on data for train and test

df.corr().applymap(lambda x: '' if abs(x) < 0.65 else abs(int(100 * x)))
