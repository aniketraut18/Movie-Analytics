#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import scipy.stats as stats


# In[2]:


tmbdMovieData = pd.read_csv(r'C:\Users\anike\Aniket\DA Projects\ref\tmdb.movies.csv', na_filter=True, na_values='[]', encoding = 'utf-8', index_col = 0)


# In[3]:


#to drop null values
tmbdMovieData.dropna()

#converting datatype of release_date column to datetime 
tmbdMovieData['release_date'] = pd.to_datetime(tmbdMovieData['release_date'])

#removing white space
tmbdMovieData['original_title'] = tmbdMovieData['original_title'].str.strip()

#to drop duplicate values
tmbdMovieData = tmbdMovieData.drop_duplicates()

tmbdMovieData.head(5)


# In[4]:


tmbdMovieData.info()


# In[5]:


#to check null values
tmbdMovieData.isnull().sum()


# In[6]:


#to check if there are duplicates
tmbdMovieData.duplicated().value_counts()


# In[7]:


tmbdMovieData.shape


# In[8]:


#read movies data file and make sure values are in the same format, and drop null values
tnMovieData = pd.read_csv(r'C:\Users\anike\Aniket\DA Projects\ref\tn.movie_budgets.csv', na_filter = True, na_values ='[]', encoding='utf-8', index_col=0)
tnMovieData.head(5)


# In[9]:


#frop null values
tnMovieData.dropna()

#converting datatype of release_date column to datetime
tnMovieData['release_date'] = pd.to_datetime(tnMovieData['release_date'])

#to drop duplicate values
tnMovieData = tnMovieData.drop_duplicates()

tnMovieData.head(5)


# In[10]:


#removing white space, '$', replace ',' and convert it to int 
tnMovieData['movie'] = tnMovieData['movie'].str.strip()
tnMovieData['production_budget'] = tnMovieData['production_budget'].str.strip('$').str.replace(",", "", regex=True).astype(int)
tnMovieData['domestic_gross'] = tnMovieData['domestic_gross'].str.strip('$').str.replace(",", "", regex=True).astype(int)
tnMovieData['worldwide_gross'] = tnMovieData['worldwide_gross'].str.strip("$").str.replace(",", "", regex=True).astype(float)
tnMovieData.head(5)


# In[11]:


tnMovieData['dom_gross / budget'] = tnMovieData['domestic_gross'] / tnMovieData['production_budget']
tnMovieData['ww_gross / budget'] = tnMovieData['worldwide_gross'] / tnMovieData['production_budget']
tnMovieData['dom_profit'] = tnMovieData['domestic_gross'] - tnMovieData['production_budget']
tnMovieData['profit'] = tnMovieData['worldwide_gross'] - tnMovieData['production_budget']
tnMovieData.head(5)


# In[12]:


tnMovieData.info()


# In[13]:


tnMovieData.isnull().sum()


# In[14]:


tnMovieData.duplicated().value_counts()


# In[15]:


tnMovieData.shape


# In[16]:


tnMovieData = tnMovieData.rename(columns={"movie":"title"})
tnMovieData.head(5)


# In[17]:


#special cases where movie titles do not match
tnMovieData['title'].replace({'Harry Potter and the Deathly Hallows: Part I' : 'Harry Potter and the Deathly Hallows: Part 1'}, inplace=True)
tnMovieData['title'].replace({'Harry Potter and the Deathly Hallows: Part II' : 'Harry Potter and the Deathly Hallows: Part 2'}, inplace=True)
tnMovieData['title'].replace({'Fast & Furious 6':'Fast and Furious 6'}, inplace=True)
tnMovieData['title'].replace({'Star Wars: The Force Awakens' : 'Star Wars Ep. VII: The Force Awakens'}, inplace=True)
tnMovieData['title'].replace({'Star Wars: The Last Jedi' : 'Star Wars Ep. VIII: The Last Jedi'}, inplace=True)


# In[18]:


#merge both datasets
movieDataset = tmbdMovieData.merge(tnMovieData, how='left')


# In[19]:


movieDataset = movieDataset.dropna()
movieDataset.head(5)


# In[20]:


#check to see if worldwide_gross and domestic_gross is 0 to remove outliers
domGrossCount = 0
for x in movieDataset['domestic_gross']:
    if x == 0:
        domGrossCount += 1
print(domGrossCount)

wwGrossCount = 0
for x in movieDataset['worldwide_gross']:
    if x == 0:
        wwGrossCount += 1
print(wwGrossCount)


# In[21]:


movieData = movieDataset.drop(movieDataset[movieDataset['domestic_gross'] <1 ].index)
movieData.head(5)


# In[22]:


movieData.dtypes


# In[23]:


def remove_outliers(data, column):
    p25 = np.percentile(data[column], 25)
    p75 = np.percentile(data[column], 75)
    iqr = p75 - p25
    outlier_threshold = p75 + (1.5 * iqr)
    outliers_indices = data[data[column] > outlier_threshold].index
    filtered_data = data.drop(index=outliers_indices)
    return filtered_data


# In[24]:


domGrossmovie = remove_outliers(movieData, 'domestic_gross')


# In[25]:


plt.scatter(domGrossmovie['vote_average'], domGrossmovie['domestic_gross'])
plt.xlabel('Rating')
plt.ylabel('Domestic Revenue')
plt.title('Domestic Revenue per Rating')

plt.show()


# In[26]:


#create line of best fit
sns.lmplot(x = 'vote_average', y = 'domestic_gross', data = domGrossmovie, height = 8, aspect = 1.5);


# In[27]:


wwGrossmovie = remove_outliers(movieData, 'worldwide_gross')


# In[28]:


plt.scatter(wwGrossmovie['vote_average'], wwGrossmovie['worldwide_gross'])
plt.xlabel('Rating')
plt.ylabel('Worldwide Revenue (hundred million $)')
plt.title('Worldwide Revenue per Rating')
plt.show()


# In[29]:


#create line of best fit
sns.lmplot(x = 'vote_average', y = 'worldwide_gross', data = wwGrossmovie, height = 8, aspect = 1.5);


# In[30]:


plt.scatter(movieData['vote_average'], movieData['dom_profit'])
plt.xlabel('Rating')
plt.ylabel('Domestic Profit')
plt.title('Domestic Profit per Rating')
plt.show()


# In[31]:


#create line of best fit
sns.lmplot(x = 'vote_average', y = 'dom_profit', data = movieData, height = 8, aspect = 1.5);


# In[32]:


#determine outliers
p25 = np.percentile(movieData['profit'], 25)
p75 = np.percentile(movieData['profit'], 75)
iqr = p75 - p25
outlier = p75 + (1.5 * iqr)
#create db that removes upper outliers
wwProfitmovie = movieData.drop(movieData[movieData['worldwide_gross'] > outlier].index)


#create db that removes upper outliers
voteAvgmovie = wwProfitmovie.drop(wwProfitmovie[wwProfitmovie['vote_average'] > outlier].index)
voteAvgmovie.head(5)


# In[33]:


plt.scatter(voteAvgmovie['vote_average'], voteAvgmovie['profit'])
plt.xlabel('Rating')
plt.ylabel('Profit (hundred million $)')
plt.title('Profit per Rating')
plt.show()


# In[34]:


#create line of best fit
sns.lmplot(x = 'vote_average', y = 'profit', data = voteAvgmovie, height = 8, aspect = 1.5);


# In[35]:


plt.scatter(movieData['production_budget'], movieData['domestic_gross'])
plt.xlabel('Budget')
plt.ylabel('Domestic Revenue')
plt.title('Domestic Revenue per Budget')
plt.show()


# In[36]:


#create line of best fit
sns.lmplot(x = 'production_budget', y = 'domestic_gross', data = movieData, height = 8, aspect = 1.5);


# In[37]:


plt.scatter(movieData['production_budget'], movieData['worldwide_gross'])
plt.xlabel('Budget')
plt.ylabel('Worldwide Revenue')
plt.title('Worldwide Revenue per Budget')
plt.show()


# In[38]:


#create line of best fit
sns.lmplot(x = 'production_budget', y = 'worldwide_gross', data = movieData, height = 8, aspect = 1.5);


# In[39]:


popularityMovie = remove_outliers(movieData, 'popularity')


# In[40]:


popularity1Movie = remove_outliers(popularityMovie, 'domestic_gross')


# In[41]:


plt.scatter(popularity1Movie['popularity'], popularity1Movie['domestic_gross'])
plt.xlabel('Popularity')
plt.ylabel('Domestic Revenue')
plt.title('Domestic Revenue per Popularity')
plt.show()


# In[42]:


#create line of best fit
sns.lmplot(x = 'popularity', y = 'domestic_gross', data = popularity1Movie, height = 8, aspect = 1.5);


# In[43]:


popularity2Movie = remove_outliers(popularityMovie, 'worldwide_gross')


# In[44]:


plt.scatter(popularity2Movie['popularity'], popularity2Movie['worldwide_gross'])
plt.xlabel('Popularity')
plt.ylabel('Worldwide Revenue (hundred million $)')
plt.title('Worldwide Revenue per Popularity')
plt.show()


# In[45]:


#create line of best fit
sns.lmplot(x = 'popularity', y = 'worldwide_gross', data = popularity2Movie, height = 8, aspect = 1.5);


# In[46]:


#determine outliers
p25 = np.percentile(popularityMovie['dom_profit'], 25)
p75 = np.percentile(popularityMovie['dom_profit'], 75)
iqr = p75 - p25
outlier = p75 + (1.5 * iqr)
#create db that removes upper outliers
popularity3Movie = popularityMovie.drop(popularityMovie[popularityMovie['dom_profit'] > outlier].index)
popularity3Movie.head(5)


# In[47]:


#determine outliers
p25 = np.percentile(popularity3Movie['dom_profit'], 25)
p75 = np.percentile(popularity3Movie['dom_profit'], 75)
iqr = p75 - p25
outlier = p25 - (1.5 * iqr)
#create db that removes lower outliers
popularity4Movie = popularity3Movie.drop(popularity3Movie[popularity3Movie['dom_profit'] < outlier].index)
popularity4Movie.head(5)


# In[48]:


plt.scatter(popularity4Movie['popularity'], popularity4Movie['dom_profit'])
plt.xlabel('Popularity')
plt.ylabel('Domestic Profit')
plt.title('Domestic Profit per Popularity')
plt.show()


# In[49]:


#create line of best fit
sns.lmplot(x = 'popularity', y = 'dom_profit', data = popularity4Movie, height = 8, aspect = 1.5);


# In[50]:


popularity5Movie = remove_outliers(popularityMovie, 'profit')


# In[51]:


plt.scatter(popularity5Movie['popularity'], popularity5Movie['profit'],)
plt.xlabel('Popularity')
plt.ylabel('Profit (hundred million $)')
plt.title('Profit per Popularity')
plt.show()


# In[52]:


#create line of best fit
sns.lmplot(x = 'popularity', y = 'profit', data = popularity5Movie, height = 8, aspect = 1.5);


# In[53]:


#population must have more than 100 votes
populationMovie = movieData.drop(movieData[movieData['vote_count']<100].index)
populationMovie.head(5)


# In[54]:


populationMovie.shape


# In[55]:


#sample must have a vote average higher than 6
sampleMovie = movieData.drop(movieData[movieData['vote_average']<6].index)
sampleMovie.head(5)


# ### To determine if there is a significant difference between the sample mean and the population mean, in either direction (greater than or less than).

# In[56]:


#Run a two tail z test using a 95% confidence level
alpha = 0.025
x_bar = sampleMovie['worldwide_gross'].mean() #sample mean 
n = sampleMovie['worldwide_gross'].count() #number of samples

sigma = populationMovie['worldwide_gross'].std() #std of population
mu = populationMovie['worldwide_gross'].mean() #population mean 


# In[57]:


#calculalte z-score
z = (x_bar - mu) / (sigma / sqrt(n))
z


# In[58]:


#calculate p-value
p_val = 1 - stats.norm.cdf(z)
#p_val = 2* (1 - stats.norm.cdf(z))
p_val


# In[59]:


#view results to identify if statistically significant
print('p-value', p_val)
print('alpha', alpha)


# In[62]:


if p_val < alpha:
    print('Reject Null Hypothesis')
elif p_val >= alpha:
    print('Fail to Reject Null Hypothesis')


# ### Since the p-value is greater than alpha, it fail to reject the null hypothesis, meaning there's not enough evidence to say that the sample mean is significantly different from the population mean.

# In[64]:


alpha = 0.025
x_bar = sampleMovie['profit'].mean()
n = sampleMovie['profit'].count()

sigma = populationMovie['profit'].std()
mu = populationMovie['profit'].mean()


# In[65]:


z = (x_bar - mu) / (sigma / sqrt(n))
z


# In[66]:


p_val = 1 - stats.norm.cdf(z)
#p_val = 2* (1 - stats.norm.cdf(z))
p_val


# In[67]:


print('p-value', p_val)
print('alpha', alpha)


# In[70]:


if p_val < alpha:
    print('Reject Null Hypothesis')
elif p_val >= alpha:
    print('Fail to Reject Null Hypothesis')


# ### The p-value is less than alpha, so reject the null hypothesis. This means there is enough evidence to suggest that the sample mean is significantly different from the population mean

# In[85]:


sampleStdMean = [(x-mu) / sigma for x in sampleMovie['profit']]
stdZMean = np.mean(sampleStdMean)
print(f'Sample Standarized Mean: {stdZMean}')


# In[82]:


fig, ax = plt.subplots(figsize=(10,8))
ax.hist(sampleStdMean, bins=15, density = True, color='r')
ax.set_xlabel('Z Scores')
ax.set_title('Profit per Movie Rating')


# In[83]:


#applying normalized curve over histogram
sns.kdeplot(sampleStdMean, ax=ax, color='k')

# marking standardized_mean
ax.vlines(x=stdZMean, ymin=0, ymax=0.7, color='b', label='Mean: 0.2331')
ax.legend()


# In[84]:


#standardizing population for secondary visualizations to follow
mu = populationMovie['profit'].mean()
std = populationMovie['profit'].std()


# In[86]:


popStdMean = [(x-mu) / sigma for x in populationMovie['profit']]
popZmean = np.mean(popStdMean)
print(f'Population Standarized Mean: {popZmean}')


# In[92]:


# plotting tested runtime sample against population
fig, ax = plt.subplots(figsize = (10,8))

plt.hist(popStdMean, bins=45, density=True, color='b', alpha=0.5)
sns.kdeplot(popStdMean, ax=ax, color='y')

plt.hist(sampleStdMean, bins=15, density=True, color='r', alpha=0.5)
sns.kdeplot(sampleStdMean, ax=ax, color='k')

plt.vlines(x=stdZMean, ymin=0, ymax=0.7, color='m', label='Sample Mean: 0.2331')
plt.vlines(x=popZmean, ymin=0, ymax=0.7, label='Population Mean: 0.0')

plt.xlabel('Z Scores')
plt.title('Profit per Movie Rating')
plt.legend()


# In[89]:


#graph visual
x = sampleMovie['vote_average']

y = sampleMovie['profit']

plt.bar(x, y)
plt.xlabel('Movie Rating')
plt.ylabel('Profit (in hundred million $)')
plt.title('Profit per Movie Rating')
plt.show()


# In[94]:


#sample must have a popularity rating higher than 10
sampleMovie = movieData.drop(movieData[movieData['popularity'] < 10].index)
sampleMovie.head(5)


# In[95]:


alpha = 0.025
x_bar = sampleMovie['worldwide_gross'].mean()
n = sampleMovie['worldwide_gross'].count()

sigma = movieData['worldwide_gross'].std()
mu = movieData['worldwide_gross'].mean()


# In[96]:


z = (x_bar - mu) / (sigma / sqrt(n))
z


# In[98]:


p_val = 1 - stats.norm.cdf(z)
p_val


# In[ ]:




