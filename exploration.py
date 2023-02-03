#!/usr/bin/env python
# coding: utf-8

# In[205]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prepare as prep

import warnings
warnings.filterwarnings('ignore')

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# In[206]:


df = pd.read_csv('wine_df.csv')


# In[207]:


df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'sulphates'], inplace=True)


# In[208]:


df.rename(columns={'fixed acidity':'fixed','volatile acidity':'volatile','citric acid':'citric',
           'residual sugar':'sugar','free sulfur dioxide':'fso2','total sulfur dioxide':'tso2'}, inplace=True)


# In[209]:


df, var_fences = prep.remove_outliers(df)


# In[210]:


df = df[(df['quality']!=3) & (df['quality']!=9)]


# In[211]:


df.info()


# In[212]:


df['alcohol'].min()


# In[213]:


cont = df.select_dtypes(include=['float64','int64'])
cat = df.select_dtypes(include='object')


# In[214]:


for col in cont:
    
    plt.hist(df[col])
    plt.title(f'distribution of {col}')
    plt.show()


# In[215]:


df.info()


# In[216]:


sns.catplot(x='quality', y='fso2', data=df)
plt.axhline(df['fso2'].mean(), linestyle='--', label='FSO2 Mean')
plt.ylabel('FSO2')
plt.title('FSO2 Content for Each Quality')
plt.legend()
plt.show()


# In[217]:


fig, ax = plt.subplots()
bplot = sns.barplot(x='type', y='tso2', data=df, palette='magma')
plt.axhline(df['tso2'].mean(), linestyle='--', label='TSO2 Mean')
plt.xlabel('Wine Type')
plt.ylabel('TSO2')
ax.bar_label(bplot.containers[0], padding=7, fmt='%.2f')
plt.ylim(0,150)
plt.legend()
plt.show()


# In[218]:


sns.violinplot(x='quality', y='citric', data=df, palette='Set2', saturation=1)
plt.axhline(df['citric'].mean(), linestyle='--', label='Citric Acid Mean')
plt.legend()


# In[219]:


sns.regplot(x='pH', y='tso2', data=df.sample(2000), line_kws={'color':'red'})
plt.title('Comparison of pH Level to Total Sulfur Dioxide')
plt.ylabel('Total Sulfur Dioxide')
plt.xlabel('pH Level')
plt.show()


# In[220]:


fso2_6 = df[df['quality']>=6]['fso2']
fso2_mean = df['fso2'].mean()


# In[221]:


t, p = stats.ttest_1samp(fso2_6, fso2_mean)


# In[222]:


a = .05


# In[223]:


p/2<a
t>0


# In[224]:


p


# In[225]:


red = df[df['type']=='red']['tso2']
tso2_mean = df['tso2'].mean()


# In[226]:


t, p = stats.ttest_1samp(red, tso2_mean)


# In[227]:


p/2<a
t<0


# In[228]:


citric_6 = df[df['quality']>=6]['citric']
citric = df['citric'].mean()


# In[229]:


t, p = stats.ttest_1samp(citric_6, citric)


# In[230]:


p/2<a


# In[231]:


t>0


# In[232]:


corr, p = stats.pearsonr(df['tso2'], df['pH'])


# In[233]:


p<a


# In[234]:


corr


# In[235]:


col_list = df.select_dtypes(include=['float64','int64']).columns[:-1]


# In[236]:


X_train, y_train, X_val, y_val, X_test, y_test = prep.x_y_split(df, 'quality')


# In[237]:


X_train, X_val, X_test = prep.mm_scaler(X_train, X_val, X_test, col_list)


# In[238]:


X_train.drop(columns=['type'], inplace=True)


# In[239]:


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(9, 6))
    pd.Series({k: KMeans(k).fit(X_train).inertia_ for k in range(2, 12)}).plot(marker='x')
    plt.xticks(range(2, 12))
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.title('Change in inertia as k increases')


# In[240]:


kmeans = KMeans(n_clusters=4)


# In[241]:


kmeans.fit(X_train)
kmeans.predict(X_train)
X_train['cluster'] = kmeans.predict(X_train)


# In[242]:


sns.lmplot(x='pH',y='tso2',data=X_train, hue='cluster')
plt.show()


# In[243]:


def find_k(X_train, cluster_vars, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(X_train[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

    return k_comparisons_df


# In[244]:


X = [X_train, X_val, X_test]


# In[245]:


X[0]


# In[246]:


# list of variables I will cluster on. 
cluster_vars = ['fso2', 'tso2']
cluster_name = 'sulfur_dioxide'
k_range = range(2,20)


# In[247]:


find_k(X[0], cluster_vars, k_range)


# In[248]:


k = 5


# In[249]:


def create_clusters(X_train, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 42)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X_train[cluster_vars])

    return kmeans


# In[250]:


kmeans = create_clusters(X_train, k, cluster_vars)


# In[251]:


# get the centroids for each distinct cluster...

def get_centroids(kmeans, cluster_vars, cluster_name):
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df


# In[252]:


centroid_df_so2 = get_centroids(kmeans, cluster_vars, cluster_name)


# In[253]:


# label cluster for each observation in X_train (X[0] in our X list of dataframes), 
# X_validate (X[1]), & X_test (X[2])

def assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df):
    for i in range(len(X)):
        clusters = pd.DataFrame(kmeans.predict(X[i][cluster_vars]), 
                            columns=[cluster_name], index=X[i].index)

        clusters_centroids = clusters.merge(centroid_df, on=cluster_name, copy=False).set_index(clusters.index.values)

        X[i] = pd.concat([X[i], clusters_centroids], axis=1)
    return X


# In[255]:


X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_so2)


# In[256]:


X[2]


# In[257]:


cluster_vars = ['fixed', 'volatile', 'citric']
cluster_name = 'acids'
k_range = range(2,20)


# In[258]:


find_k(X[0], cluster_vars, k_range)


# In[259]:


k = 3


# In[260]:


kmeans = create_clusters(X[0], k, cluster_vars)


# In[261]:


centroid_df_acid = get_centroids(kmeans, cluster_vars, cluster_name)


# In[262]:


X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_acid)


# In[263]:


X[0]


# In[264]:


cluster_vars = ['density', 'alcohol']
cluster_name = 'viscosity'
k_range = range(2,20)


# In[265]:


find_k(X[0], cluster_vars, k_range)


# In[266]:


k = 5


# In[267]:


kmeans = create_clusters(X[0], k, cluster_vars)


# In[268]:


centroid_df_visc = get_centroids(kmeans, cluster_vars, cluster_name)


# In[269]:


X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_visc)


# In[270]:


X[0]


# In[277]:


X[2].drop(columns=['centroid_fso2', 'centroid_tso2','centroid_fixed',
                   'centroid_volatile','centroid_citric','centroid_density','centroid_alcohol'], inplace=True)


# In[280]:


X[0]


# In[293]:


sns.scatterplot(x='density',y='alcohol',data=X[0].sample(2000), hue='viscosity')


# In[296]:


def cluster_so2(L):
    
    cluster_vars = ['fso2', 'tso2']
    cluster_name = 'so2'
    
    k = 5
    
    kmeans = create_clusters(L[0], k, cluster_vars)
    
    centroid_df_so2 = get_centroids(kmeans, cluster_vars, cluster_name)
    
    L = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_so2)
    
    return L


# In[297]:


def cluster_acids(L):
    
    cluster_vars = ['fixed', 'volatile', 'citric']
    cluster_name = 'acids'
    
    k = 3
    
    kmeans = create_clusters(L[0], k, cluster_vars)
    
    centroid_df_acid = get_centroids(kmeans, cluster_vars, cluster_name)
    
    L = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_acid)
    
    return L


# In[298]:


def cluster_visc(L):
    
    cluster_vars = ['density', 'alcohol']
    cluster_name = 'visc'
    
    k = 5 
    
    kmeans = create_clusters(L[0], k, cluster_vars)
    
    centroid_df_visc = get_centroids(kmeans, cluster_vars, cluster_name)
    
    L = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_visc)
    
    return L


# In[ ]:




