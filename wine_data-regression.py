
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import re
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('winemag-data_first150k.csv', encoding='latin-1')
data.head(3)

#Check original size of the dataset
len(data)

data2 = pd.read_csv('winemag-data-130k-v2.csv', encoding='latin-1')
data2.head(3)

#Combining two datasets
bigdata = data.append(data2, ignore_index=True, sort=True)

#Check size
#len(bigdata)

#drop duplicates
new = bigdata.drop_duplicates(['description'], keep=False)

#delete 'id' column
del new['id']

# Get rid of price outliers.
q1 = new['price'].quantile(0.25)      
q2 = new['price'].quantile(0.5)            
q3 = new['price'].quantile(0.75)
q4 = new['price'].quantile(1.0)
iqr = q3 - q1
lower_bound  = q1 - (1.5  * iqr)
upper_bound = q3 + (1.5 * iqr)
new = new.loc[(new['price'] > lower_bound) & (new['price'] < upper_bound)]

#Drop na
new_data = new.fillna(new.mean())
# The pandas function "get dummies" generates G dummy variables for a predictor with G levels; 
# which are represented as 0/1 - absence/presence.

predictors = pd.concat([new_data.points,  
            pd.get_dummies(new_data.variety),
           pd.get_dummies(new_data.province), pd.get_dummies(new_data.country)], axis = 1)   


predictors.isnull().sum()


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
predictors_imputed = imp.fit_transform(predictors)

#First Lasso regression run - all wine types (variety) included
X_train, X_test, y_train, y_test = train_test_split(predictors, 
                                        np.log(new_data["price"]), 
                                               test_size=0.30, random_state=42)

model_lasso_cv = LassoCV(cv=10, precompute = False, normalize=True, 
                         n_jobs = -1).fit(X_train, y_train)

sns.set(rc={'figure.figsize':(11.7, 30)})


lasso_coef = pd.DataFrame(np.round_(model_lasso_cv.coef_, decimals=3), 
predictors.columns, columns = ["penalized_regression_coefficients"])
# remove the non-zero coefficients
lasso_coef = lasso_coef[lasso_coef['penalized_regression_coefficients'] != 0]
# sort the values from high to low
lasso_coef = lasso_coef.sort_values(by = 'penalized_regression_coefficients', 
ascending = False)
# plot the sorted dataframe
ax = sns.barplot(x = 'penalized_regression_coefficients', y= lasso_coef.index , 
data=lasso_coef, ci=50)
ax.set(xlabel='Penalized Regression Coefficients')


train_RMSE = np.sqrt(mean_squared_error(y_train, model_lasso_cv.predict(X_train)))
test_RMSE = np.sqrt(mean_squared_error(y_test, model_lasso_cv.predict(X_test)))
print('training data RMSE')
print(train_RMSE)
print('test data RMSE')
print(test_RMSE)

# plot price (log transformed and original scale) for 'Nebbiolo' dummy 
# set up the canvas- 1 row with 2 columns for the plots
figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(9, 9), sharex=True)
# set up the plot for the log-transformed price variable
ax = sns.stripplot(x=pd.get_dummies(new.variety)['Nebbiolo'], 
y=np.log(new.price), ax=axes[0], jitter = True)
ax.set(ylabel='Log (PriceRetail)')
# set up the plot for the original scale price variable
sns.stripplot(x=pd.get_dummies(new.variety)['Nebbiolo'], 
y=new.price, ax=axes[1], jitter = True)

#relationship between the dummy variable representing Nebbiolo wines 
#(0 for non-Nebbiolo, 1 for Nebbiolo wines) and wine price, 
#the log and original scale

## predicted versus actual on log and non-log scale
figure, axes = plt.subplots(nrows=2, ncols=1,figsize=(9, 9))
# both test data and predictions on log scale
ax = sns.regplot(x = y_test, y = model_lasso_cv.predict(X_test), ax=axes[0], 
 scatter_kws={"color": "green"}, line_kws={"color": "red"})
ax.set(xlabel='Actual Log (price): Test Set', 
ylabel = 'Predicted Log (price): Test Set')
# both test data and predictions on actual (anti-logged) scale
ax = sns.regplot(x = np.exp(y_test), y = np.exp(model_lasso_cv.predict(X_test)), ax=axes[1], scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax = ax.set(xlabel='Actual price: Test Set', 
ylabel = 'Predicted price: Test Set')

#Lasso Regression second run
#narrow down list of wine variety

#new['variety'].nunique()

# we have 644 types of wine, lets set a treshold to narrow down
prob = new['variety'].value_counts(normalize=True)
threshold = 0.002
fig, ax = plt.subplots(figsize = (10,4))
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=90)
plt.show()

top_wines = ('Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Red Blend', 'Sauvignon Blanc', 
             'Bordeaux-style Red Blend', 'Syrah', 'Riesling', 'Merlot', 'Zinfandel', 'Rosé',
             'Malbec', 'White Blend', 'Sangiovese', 'Tempranillo', 'Sparkling Blend', 'Portuguese Red',
             'Rhône-style Red Blend', 'Shiraz', 'Viognier','Pinot Gris', 'Cabernet Franc', 
             'Corvina, Rondinella, Molinara', 'Pinot Grigio', 'Gewürztraminer', 'Nebbiolo', 'Grüner Veltliner',
             'Petite Sirah', 'Champagne Blend', 'Portuguese White', 'Sangiovese Grosso', 'Bordeaux-style White Blend',
             'Port', 'Tempranillo Blend', 'Chenin Blanc', 'Grenache', 'Moscato', 'Pinot Blanc', 'Barbera', 'Aglianico',
            "Nero d'Avola", 'Garnacha', 'Verdejo', 'Sauvignon', 'Glera', 'Meritage', 'Primitivo', 'Petit Verdot')


new = new[new['variety'].isin(top_wines)]
new['variety'].nunique()

predictors2 = pd.concat([new.points,  
            pd.get_dummies(new.variety),
           pd.get_dummies(new.country)] , axis = 1)   

predictors2.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(predictors2, 
                                        np.log(new["price"]), 
                                               test_size=0.30, random_state=42)

model_lasso_cv = LassoCV(cv=10, precompute = False, normalize=True, 
                         n_jobs = -1).fit(X_train, y_train)

sns.set(rc={'figure.figsize':(11.7, 30)})


lasso_coef = pd.DataFrame(np.round_(model_lasso_cv.coef_, decimals=3), 
predictors2.columns, columns = ["penalized_regression_coefficients"])
# remove the non-zero coefficients
lasso_coef = lasso_coef[lasso_coef['penalized_regression_coefficients'] != 0]
# sort the values from high to low
lasso_coef = lasso_coef.sort_values(by = 'penalized_regression_coefficients', 
ascending = False)
# plot the sorted dataframe
ax = sns.barplot(x = 'penalized_regression_coefficients', y= lasso_coef.index , 
data=lasso_coef)
ax.set(xlabel='Penalized Regression Coefficients')

train_RMSE = np.sqrt(mean_squared_error(y_train, model_lasso_cv.predict(X_train)))
test_RMSE = np.sqrt(mean_squared_error(y_test, model_lasso_cv.predict(X_test)))
print('training data RMSE')
print(train_RMSE)
print('test data RMSE')
print(test_RMSE)

# plot price (log transformed and original scale) for 'Petite Syrah' dummy 
import matplotlib.pyplot as plt
# set up the canvas- 1 row with 2 columns for the plots
figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(9, 9), sharex=True)
# set up the plot for the log-transformed price variable
ax = sns.stripplot(x=pd.get_dummies(new.variety)['Petite Sirah'], 
y=np.log(new.price), ax=axes[0], jitter = True)
ax.set(ylabel='Log (PriceRetail)')
# set up the plot for the original scale price variable
sns.stripplot(x=pd.get_dummies(new.variety)['Petite Sirah'], 
y=new.price, ax=axes[1], jitter = True)

## predicted versus actual on log and non-log scale
figure, axes = plt.subplots(nrows=2, ncols=1,figsize=(9, 9))
# both test data and predictions on log scale
ax = sns.regplot(x = y_test, y = model_lasso_cv.predict(X_test), ax=axes[0],  
                 scatter_kws={"color": "green"}, line_kws={"color": "red"})
ax.set(xlabel='Actual Log (price): Test Set', 
ylabel = 'Predicted Log (price): Test Set')


# both test data and predictions on actual (anti-logged) scale
ax = sns.regplot(x = np.exp(y_test), y = np.exp(model_lasso_cv.predict(X_test)), ax=axes[1],
                 scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax = ax.set(xlabel='Actual price: Test Set', 
ylabel = 'Predicted price: Test Set')