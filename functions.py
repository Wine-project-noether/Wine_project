#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


def create_clusters(X, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 42)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X[cluster_vars])

    return kmeans


# In[4]:


def get_centroids(kmeans, cluster_vars, cluster_name):
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df


# In[5]:


def assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df, X):
    for i in range(len(X)):
        clusters = pd.DataFrame(kmeans.predict(X[i][cluster_vars]), 
                            columns=[cluster_name], index=X[i].index)

        clusters_centroids = clusters.merge(centroid_df, on=cluster_name, copy=False).set_index(clusters.index.values)

        X[i] = pd.concat([X[i], clusters_centroids], axis=1)
    return X


# In[7]:


def cluster_so2(X):
    
    cluster_vars = ['fso2', 'tso2']
    cluster_name = 'so2'
    
    k = 5
    
    kmeans = create_clusters(X[0], k, cluster_vars)
    
    centroid_df_so2 = get_centroids(kmeans, cluster_vars, cluster_name)
    
    X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_so2, X)
    
    return X


# In[8]:


def cluster_acids(X):
    
    cluster_vars = ['fixed', 'volatile', 'citric']
    cluster_name = 'acids'
    
    k = 3
    
    kmeans = create_clusters(X[0], k, cluster_vars)
    
    centroid_df_acid = get_centroids(kmeans, cluster_vars, cluster_name)
    
    X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_acid, X)
    
    return X


# In[9]:


def cluster_visc(X):
    
    cluster_vars = ['density', 'alcohol']
    cluster_name = 'visc'
    
    k = 5 
    
    kmeans = create_clusters(X[0], k, cluster_vars)
    
    centroid_df_visc = get_centroids(kmeans, cluster_vars, cluster_name)
    
    X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_visc, X)
    
    return X


# In[ ]:




