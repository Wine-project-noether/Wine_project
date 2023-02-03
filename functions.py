#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


# In[11]:


def wrangle_wine():
    
    '''
    This function gets the wine dataset from the csv file and drops unnecessary columns,
    renames columns to make the data frame easier to read and removes outliers.
    '''
    
    # creating initial Data Frame
    df = pd.read_csv('wine_df.csv')
    
    # dropping columns
    df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'sulphates'], inplace=True)
    
    # renaming columns
    df.rename(columns={'fixed acidity':'fixed','volatile acidity':'volatile',
                       'citric acid':'citric','residual sugar':'sugar','free sulfur dioxide':'fso2',
                       'total sulfur dioxide':'tso2'}, inplace=True)
    
    # removing outliers
    df, var_fences = prep.remove_outliers(df)
    df = df[(df['quality']!=3) & (df['quality']!=9)]
    
    return df


# In[12]:


def fso2_plot(df, col):
    
    '''
    This function takes the data frame and the fso2 column and returns a catplot of amount of 
    free sulfur dioxide in each wine split up by quality.
    '''
    
    sns.catplot(x='quality', y=col, data=df)
    plt.axhline(df[col].mean(), linestyle='--', label='FSO2 Mean')
    plt.ylabel('FSO2')
    plt.title('FSO2 Content for Each Quality')
    plt.legend()
    plt.show()


# In[13]:


def tso2_plot(df, col):
    
    '''
    This function takes the data frame and the tso2 column and returns a bar plot of the 
    average total sulfur dioxide in the types of wines.
    '''
    
    fig, ax = plt.subplots()
    bplot = sns.barplot(x='type', y=col, data=df, palette='magma')
    plt.axhline(df[col].mean(), linestyle='--', label='TSO2 Mean')
    plt.xlabel('Wine Type')
    plt.ylabel('TSO2')
    ax.bar_label(bplot.containers[0], padding=7, fmt='%.2f')
    plt.ylim(0,150)
    plt.legend()
    plt.show()


# In[14]:


def citric_plot(df, col):
    
    '''
    This function takes the data frame and the citric column and returns a violin plot of the 
    amount of citric acid for each quality of wine.
    '''
    
    sns.violinplot(x='quality', y=col, data=df, palette='Set2', saturation=1)
    plt.axhline(df[col].mean(), linestyle='--', label='Citric Acid Mean')
    plt.ylabel('Citric Content')
    plt.xlabel('Quality')
    plt.title('Comparing Citric Content to the Wine Quality')
    plt.legend()
    plt.show()
    


# In[15]:


def so2_ph_plot(df, col1, col2):
    
    '''
    The function takes the data frame, the tso2 column and the pH column and returns a regression plot
    of the correlation between total sulfur dioxide and pH level.
    '''
    
    sns.regplot(x=col1, y=col2, data=df.sample(2000), line_kws={'color':'red'})
    plt.title('Comparison of pH Level to Total Sulfur Dioxide')
    plt.ylabel('Total Sulfur Dioxide')
    plt.xlabel('pH Level')
    plt.show()
    


# In[22]:


def ttest(df, col, comp_col, comp_split):
    
    '''
    This function takes a data frame, column, comparison column and where to split the comparison column.
    It separates the comparison column on the split given and runs a one sample t-test comparing the 
    data of the column above the split to the overall mean.
    '''
    
    alpha = .05
    
    # determining the sample and the overall mean
    sample = df[df[comp_col]>=comp_split][col]
    overall_mean = df[col].mean()
    
    # running the ttest 
    t, p = stats.ttest_1samp(sample, overall_mean)
    
    # if statement to determine whether t value needs to be above or below 0
    if (p/2 < alpha) and (t > 0):
        print("We reject the null.")
    else:
        print("We fail to reject the null.")
        


# In[21]:


def ttest_type(df, col, comp_col, comp_split):    
    
    '''
    This function takes a data frame, column, comparison column and where to split the comparison column.
    It separates the comparison column on the split given and runs a one sample t-test comparing the 
    data of the column above the split to the overall mean.
    '''
    
    alpha = .05
    
    # determining the sample and the overall mean
    sample = df[df[comp_col]==comp_split][col]
    overall_mean = df[col].mean()
    
    # running the ttest 
    t, p = stats.ttest_1samp(sample, overall_mean)
    
    # if statement to determine whether t value needs to be above or below 0
    if (p/2 < alpha) and (t < 0):
        print("We reject the null.")
    else:
        print("We fail to reject the null.")
        


# In[17]:


def pearson_test(df, col1, col2):
    
    '''
    This function takes a data frame and 2 columns and runs a pearsonr stats test on the columns and 
    returns the correlation between the two columns.
    '''
    
    alpha = .05
    
    # running stats test on both columns
    corr, p = stats.pearsonr(df[col1], df[col2])
    
    # if statment to determine whether to reject the null
    if p < alpha:
        print("We reject the null.")
        print(f'{col1} and {col2} have a correlation of {corr}')
    else:
        print("We fail to reject the null.")
        


# In[18]:


def split_scale(df):
    
    '''
    This function gets dummies for the type column, splits and scales the data.
    '''
    
    # creating list of numeric columns and getting dummies
    col_list = df.select_dtypes(include=['float64','int64']).columns[:-1]
    df = pd.get_dummies(df, columns=['type'])
    
    # splitting and scaling data
    X_train, y_train, X_val, y_val, X_test, y_test = prep.x_y_split(df, 'quality')
    X_train, X_val, X_test = prep.mm_scaler(X_train, X_val, X_test, col_list)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# In[2]:


def find_k(X_train, cluster_vars, k_range):
    
    '''
    This function takes the X_trian data set and the variables to cluster on and creates a chart
    of the interia of the k value to determine which k value is the best.
    '''
    
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

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()



# In[3]:


def create_clusters(X, k, cluster_vars):
    
    '''
    This function creates the clusters using KMeans.
    '''
    
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 42)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X[0][cluster_vars])

    return kmeans


# In[4]:


def get_centroids(kmeans, cluster_vars, cluster_name):
    
    '''
    This function creates a dataframe of the centroids for the clusters
    '''
    
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df


# In[5]:


def assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df, X):
    
    '''
    This function adds the clusters and centroids to the X dataframes.
    '''
    
    for i in range(len(X)):
        clusters = pd.DataFrame(kmeans.predict(X[i][cluster_vars]), 
                            columns=[cluster_name], index=X[i].index)

        clusters_centroids = clusters.merge(centroid_df, on=cluster_name, copy=False).set_index(clusters.index.values)

        X[i] = pd.concat([X[i], clusters_centroids], axis=1)
    return X


# In[20]:


def cluster_so2(X):
    
    '''
    This function creates the clusters for the sulfur dioxide variables.
    '''
    
    # creating cluster variables and name
    cluster_vars = ['fso2', 'tso2']
    cluster_name = 'so2_cluster'
    
    k = 5
    
    # creatings clusters
    kmeans = create_clusters(X, k, cluster_vars)
    
    # creating centroids
    centroid_df_so2 = get_centroids(kmeans, cluster_vars, cluster_name)
    
    # adding clusters and centroids to X data frames
    X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_so2, X)
    
    X[0].drop(columns=['centroid_fso2','centroid_tso2'], inplace=True)
    X[1].drop(columns=['centroid_fso2','centroid_tso2'], inplace=True)
    X[2].drop(columns=['centroid_fso2','centroid_tso2'], inplace=True)
    
    return X


# In[21]:


def cluster_acids(X):
    
    '''
    This function creates the clusters for the acid variables.
    '''
    
    # creating cluster variables and name
    cluster_vars = ['fixed', 'volatile', 'citric']
    cluster_name = 'acids_cluster'
    
    k = 3
    
    # creating clusters
    kmeans = create_clusters(X, k, cluster_vars)
    
    #creating centroids
    centroid_df_acid = get_centroids(kmeans, cluster_vars, cluster_name)
    
    # adding clusters and centroids to X data frames
    X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_acid, X)
    
    X[0].drop(columns=['centroid_fixed','centroid_volatile','centroid_citric'], inplace=True)
    X[1].drop(columns=['centroid_fixed','centroid_volatile','centroid_citric'], inplace=True)
    X[2].drop(columns=['centroid_fixed','centroid_volatile','centroid_citric'], inplace=True)
    
    return X


# In[22]:


def cluster_visc(X):
    
    '''
    This function creates the clusters for the viscosity variables.
    '''
    
    # creating cluster variables and name
    cluster_vars = ['density', 'alcohol']
    cluster_name = 'visc_cluster'
    
    k = 5 
    
    # creating clusters
    kmeans = create_clusters(X, k, cluster_vars)
    
    # creating centroids
    centroid_df_visc = get_centroids(kmeans, cluster_vars, cluster_name)
    
    # addign clusters and centroids to X data frames
    X = assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df_visc, X)
    
    X[0].drop(columns=['centroid_density','centroid_alcohol'], inplace=True)
    X[1].drop(columns=['centroid_density','centroid_alcohol'], inplace=True)
    X[2].drop(columns=['centroid_density','centroid_alcohol'], inplace=True)
    
    return X


# In[24]:


def tso2_cluster_plot(X, y_train, col, cluster):
    
    '''
    This function plots a stripplot for total sulfur dioxide compared to quality, hued with the cluster
    '''
    
    plt.figure(figsize=(10,6))
    sns.stripplot(x=y_train, y=X[0][col], hue=X[0][cluster])
    plt.ylabel('Total Sulfur Dioxide')
    plt.xlabel('Quality')
    plt.title('Is there a distinction between clusters when visualizing TSO2 and Quality?')
    plt.show()


# In[25]:


def acid_cluster_plot(X, y_train, col, cluster):

    '''
    This function plots a stripplot for acidity compared to quality, hued with the cluster
    '''
    
    plt.figure(figsize=(10,6))
    sns.stripplot(x=y_train, y=X[0][col], hue=X[0][cluster])
    plt.ylabel('Acidity')
    plt.xlabel('Quality')
    plt.title('Is there a distinction between clusters when visualizing Acidity and Quality?')
    plt.show()


# In[26]:


def visc_cluster_plot(X, y_train, col, cluster):
    
    '''
    This function plots a stripplot for density compared to quality, hued with the cluster
    '''
    
    plt.figure(figsize=(10,6))
    sns.stripplot(x=y_train, y=X[0][col], hue=X[0][cluster])
    plt.title('Is there a distinction between clusters when visualizing Density and Quality?')
    plt.ylabel('Alcohol Amount')
    plt.xlabel('Quality')
    plt.show()


# In[ ]:




