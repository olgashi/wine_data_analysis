import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import re
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


data1 = pd.read_csv('winemag-data_first150k.csv', encoding='latin-1')
data2 = pd.read_csv('winemag-data-130k-v2.csv', encoding='latin-1')

#combine dataset
bigdata = data1.append(data2, ignore_index=True, sort=True)

new = bigdata.drop_duplicates(['description'], keep=False)

# remove outliers

q1 = bigdata['price'].quantile(0.25)      
q2 = bigdata['price'].quantile(0.5)            
q3 = bigdata['price'].quantile(0.75)
q4 = bigdata['price'].quantile(1.0)
iqr = q3 - q1
lower_bound  = q1 - (1.5  * iqr)
upper_bound = q3 + (1.5 * iqr)
bigdata = new.loc[(bigdata['price'] > lower_bound) & (bigdata['price'] < upper_bound)]

#exploratory analysis:

prob = bigdata['province'].value_counts(normalize=True)
threshold = 0.005
fig = plt.subplots(figsize = (10,5))
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=80)
plt.show()

prob = bigdata['country'].value_counts(normalize=True)
threshold = 0.002
mask = prob > threshold
fig = plt.subplots(figsize = (10,5))
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=25)
plt.show()

prob = bigdata['designation'].value_counts(normalize=True)
threshold = 0.002
fig = plt.subplots(figsize = (10,5))
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=25)
plt.show()

prob = bigdata['variety'].value_counts(normalize=True)
threshold = 0.003
fig, ax = plt.subplots(figsize = (10,4))
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=80)
plt.show()


sns.jointplot(x='price', y='points', data=bigdata, facecolors='none', edgecolors='darkblue', alpha=0.1)
plt.show()

# We take the logarithm of the price and points columns
log_score = np.log(bigdata[['price', 'points']])
log_score = log_score.dropna().reset_index(drop=True)
log_score.columns = ['log_price', 'log_score']

# We visualize our DataFrame
log_score.head()

# Scatter plot
sns.jointplot(x='log_price', y='log_score', data=log_score, facecolors='none', edgecolors='darkblue', alpha=0.05)
plt.show()

# Correlation coefficient: 
corr = np.corrcoef(bigdata['points'], bigdata['price'])
corr2 = np.corrcoef(log_score['log_score'], log_score['log_price'])
print('Data correlation coefficient: %.4f \nLog-data correlation coefficient: %.4f' % (corr[0][1], corr2[0][1]))

# Price quartiles
quart1 = bigdata[bigdata.price < bigdata.price.quantile(.25)].reset_index(drop=True)
quart1 = quart1.dropna().reset_index(drop=True)

quart2 = bigdata[(bigdata.price < bigdata.price.quantile(.50)) & (bigdata.price >= bigdata.price.quantile(.25))].reset_index(drop=True)
quart2 = quart2.dropna().reset_index(drop=True)

quart3 = bigdata[(bigdata.price < bigdata.price.quantile(.75)) & (bigdata.price >= bigdata.price.quantile(.50))].reset_index(drop=True)
quart3 = quart3.dropna().reset_index(drop=True)

quart4 = bigdata[bigdata.price >= bigdata.price.quantile(.75)].reset_index(drop=True)
quart4 = quart4.dropna().reset_index(drop=True)

plt.figure(figsize=(20,10))

plt.subplot(2, 2, 1)

plt.title('Quartile 1', fontsize=20)
sns.distplot( quart1['points'], color='green', kde=False)
plt.axvline(np.mean(bigdata.points), 0,1, linestyle='--', color='black', label='Mean score')
plt.axvline(np.mean(quart1.points), 0,1, linestyle='--', color='green', label='Q1 mean score')
plt.legend(fontsize=15)

plt.subplot(2, 2, 2)
plt.title('Quartile 2', fontsize=20)
sns.distplot( quart2['points'], color='gold', kde=False)
plt.axvline(np.mean(bigdata.points), 0,1, linestyle='--', color='black', label='Mean score')
plt.axvline(np.mean(quart2.points), 0,1, linestyle='--', color='gold', label='Q2 mean score')
plt.legend(fontsize=15)

plt.subplot(2, 2, 3)
plt.title('Quartile 3', fontsize=20)
sns.distplot( quart3['points'], color='red', kde=False)
plt.axvline(np.mean(bigdata.points), 0,1, linestyle='--', color='black', label='Mean score')
plt.axvline(np.mean(quart3.points), 0,1, linestyle='--', color='red', label='Q3 mean score')
plt.legend(fontsize=15)

plt.subplot(2, 2, 4)
plt.title('Quartile 4', fontsize=20)
sns.distplot( quart4['points'], color='skyblue', kde=False)
plt.axvline(np.mean(bigdata.points), 0,1, linestyle='--', color='black', label='Mean score')
plt.axvline(np.mean(quart4.points), 0,1, linestyle='--', color='skyblue', label='Q4 mean score')
plt.legend(fontsize=15)

plt.show()