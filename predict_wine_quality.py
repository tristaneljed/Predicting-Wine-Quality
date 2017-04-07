from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import statsmodels.stats.api as sm

%pylab inline

# Read data
wine = pd.read_csv('wine_data.csv', sep='\t')
wine.head()

# We see 11 signs describing the chemical composition of the wines. Here is the distribution of expert evaluation of wines in the sample
plt.figure(figsize(8,6))
stat = wine.groupby('quality')['quality'].agg(lambda x : float(len(x))/wine.shape[0])
stat.plot(kind='bar', fontsize=14, width=0.9, color="green")
plt.xticks(rotation=0)
plt.ylabel('Proportion', fontsize=14)
plt.xlabel('Quality', fontsize=14)

# Type and Quality of Wine
y = wine[['Type', 'quality']]
y.head()

wine.groupby('Type')['Type'].count()

# We separate 25% of the set for the quality prediction control
X_train, X_test, y_train, y_test = train_test_split(wine.ix[:, wine.columns != 'quality'], wine['quality'], test_size=0.25)											
X_train['Type'] = X_train['Type'].apply(lambda x : -1 if x == 'red' else 1)
X_test['Type'] = X_test['Type'].apply(lambda x : -1 if x == 'red' else 1)

# If we do not have any more information about the wines, our best guess on the assessment – - the average, available in the training set
np.mean(y_train)
# If we predict this value assessment of all the winse oin the training set, we get the average square error
sqrt(mean_squared_error([np.mean(y_train)]*len(y_train), y_train))
# and on the test
sqrt(mean_squared_error([np.mean(y_train)]*len(y_test), y_test))

regressor = LinearRegression()
regressor.fit(X_train['Type'].reshape(-1,1), y_train)

y_train_predictions = regressor.predict(X_train['Type'].reshape(-1,1))
y_test_predictions = regressor.predict(X_test['Type'].reshape(-1,1))

sqrt(mean_squared_error(y_train_predictions, y_train))
sqrt(mean_squared_error(y_test_predictions, y_test))

pyplot.figure(figsize(10,10))
pyplot.scatter(y_test, y_test_predictions, color="blue", alpha=0.1)
pyplot.xlim(2,10)
pyplot.ylim(2,10)
plot(range(11), color='black')
grid()
plt.xlabel('Quality')
plt.ylabel('Estimated quality')

def fun(v):
    return v + np.random.uniform(low=-0.5, 
                                 high=0.5, 
                                 size=len(v))

pyplot.figure(figsize(15, 35))
for i in range (1, 12):
    pyplot.subplot(6, 2, i)
    pyplot.scatter(fun(wine['quality']), wine.ix[:, i], 
                   color=wine["Type"], 
                   edgecolors="green")
    pyplot.xlabel('Quality')
    pyplot.ylabel(str(wine.columns[i]))

# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)

sqrt(mean_squared_error(lm.predict(X_train), y_train))
sqrt(mean_squared_error(lm.predict(X_test), y_test))

plt.figure(figsize(16,8))
plt.subplot(121)
pyplot.scatter(y_train, lm.predict(X_train), color="red", alpha=0.1)
pyplot.xlim(2,10)
pyplot.ylim(2,10)
plot(range(11), color='black')
grid()
pyplot.title('Train set', fontsize=16)
pyplot.xlabel('Quality')
pyplot.ylabel('Estimated quality')

plt.subplot(122)
pyplot.scatter(y_test, lm.predict(X_test), color="red", alpha=0.1)
pyplot.xlim(2,10)
pyplot.ylim(2,10)
plot(range(11), color='black')
grid()
pyplot.title('Test set', fontsize=16)
pyplot.xlabel('Quality')
pyplot.ylabel('Estimated quality')

# We calculate the coefficient of determination - the proportion by the explained model of the disperse response.
lm.score(X_test, y_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(X_train, y_train)

sqrt(mean_squared_error(rf.predict(X_train), y_train))
sqrt(mean_squared_error(rf.predict(X_test), y_test))

plt.figure(figsize(16,7))
plt.subplot(121)
pyplot.scatter(y_train, rf.predict(X_train), color="red", alpha=0.1)
pyplot.xlim(2,10)
pyplot.ylim(2,10)
plot(range(11), color='black')
grid()
pyplot.title('Train set', fontsize=16)
pyplot.xlabel('Quality')
pyplot.ylabel('Estimated quality')

plt.subplot(122)
pyplot.scatter(y_test, rf.predict(X_test), color="red", alpha=0.1)
pyplot.xlim(2,10)
pyplot.ylim(2,10)
plot(range(11), color='black')
grid()
pyplot.title('Test set', fontsize=16)
pyplot.xlabel('Quality')
pyplot.ylabel('Estimated quality')

# The coefficient of determination for the random forest
rf.score(X_test, y_test)

# We compare the errors of the linear regression and random forest on a test sample
plt.figure(figsize(8,6))
plt.hist(abs(y_test - lm.predict(X_test)) - abs(y_test - rf.predict(X_test)), bins=16, normed=True)
plt.xlabel('Difference of absolute errors')

# The differences between the average absolute errors are significant
tmeans = sm.CompareMeans(sm.DescrStatsW(abs(y_test - lm.predict(X_test))), 
                         sm.DescrStatsW(abs(y_test - rf.predict(X_test))))

tmeans.ttest_ind(alternative='two-sided', usevar='pooled', value=0)[1]

# 95% confidence interval for the average difference of absolute errors
tmeans.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='pooled')

importances = pd.DataFrame(zip(X_train.columns, rf.feature_importances_))
importances.columns = ['feature name', 'importance']
importances.sort(ascending=False)
# The alcohol content has the greatest influence on the expert evaluation of wine quality.