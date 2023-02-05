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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


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
    


# In[6]:


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
    


# In[7]:


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
        


# In[8]:


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
        


# In[9]:


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
        


# In[10]:


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


# In[11]:


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


# In[12]:


def create_clusters(X, k, cluster_vars):
    
    '''
    This function creates the clusters using KMeans.
    '''
    
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 42)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X[0][cluster_vars])

    return kmeans


# In[13]:


def get_centroids(kmeans, cluster_vars, cluster_name):
    
    '''
    This function creates a dataframe of the centroids for the clusters
    '''
    
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df


# In[14]:


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


# In[15]:


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
    
    return X


# In[16]:


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
    
    return X


# In[17]:


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
    
    return X


# In[18]:


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


# In[19]:


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


# In[20]:


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


# # Decision Tree Model

# In[ ]:


def decision_tree(X_train, y_train, X_val, y_val):
    '''
    Takes in x and y to make a plot for each score of x and y train
    '''
    
    tree_train = []
    tree_val = []
    depth = []
    for i in range(2, 21):
        train_tree = DecisionTreeClassifier(max_depth=i, random_state=42)
        train_tree = train_tree.fit(X_train, y_train)

        tree_train.append(train_tree.score(X_train, y_train))
        tree_val.append(train_tree.score(X_val, y_val))
        depth.append(i)
    
        # Creating the rfc_dataframe
    tree_scores = pd.DataFrame({'score':tree_train,
                           'type':'train',
                           'depth':depth})
    #Creating val_rfc_score_df
    val_tree_scores = pd.DataFrame({'score':tree_val,
                                   'type':'val',
                                   'depth':depth})
    #Creating rfc_score_df
    tree_scores = tree_scores.append(val_tree_scores)
    # train scores loc of ref_score_df
    train_acc=tree_scores.loc[tree_scores['type'] == 'train']
    # val scores loc of ref_score_df
    val_acc=tree_scores.loc[tree_scores['type'] == 'val']
    # train depth loc of rfc_score_df
    train_depth=tree_scores.loc[tree_scores['type']== 'train']['depth']
    # val depth loc of rfc_score_df
    val_depth=tree_scores.loc[tree_scores['type']== 'val']['depth']
    # rfc_scre_df 
    tree_score_df= pd.DataFrame(
        {'train_score': train_acc.score,
        'val_score': val_acc.score,
         'train_depth': train_depth,
         'val_depth' : val_depth
        })
    # plot f, ax
    f, ax = plt.subplots(1, 1)
    
    #setting the title of chart
    plt.title("Decision Tree Accuracy and Depth Chart")
    # plotting the data
    sns.pointplot(data =tree_score_df, x='train_depth', y='train_score', label='Train',color='royalblue')
    sns.pointplot(data =tree_score_df, x='val_depth', y='val_score', label='Val', color="seagreen")
    # setting the labels
    plt.ylabel('score')
    plt.xlabel('depth')
    #Showing the graph
    plt.show()


# # Random Forest Classifier model function

# In[21]:


def random_for_class(X_train, y_train, X_val, y_val):
    '''
    Takes in x and y train to make a plot of each score of xand y train
    '''
    rfc_train = [] #Empty List
    rfc_val = []
    depth = []
    
    for i in range(2, 21): # Range of 2-21 and each value inbetween
        # Random Forest Classifier
        rf = RandomForestClassifier(bootstrap=True, 
                                    class_weight=None, 
                                    criterion='gini',
                                    min_samples_leaf=3,
                                    n_estimators=100,
                                    max_depth=i, 
                                    random_state=42)
        # Fitting the Xtrain and ytrain for the model
        rf.fit(X_train, y_train)
        # Creating values for empty lists
        rfc_train.append(rf.score(X_train, y_train))
        rfc_val.append(rf.score(X_val, y_val))
        depth.append(i)
    # Creating the rfc_dataframe
    rfc_scores = pd.DataFrame({'score':rfc_train,
                           'type':'train',
                           'depth':depth})
    #Creating val_rfc_score_df
    val_rfc_scores = pd.DataFrame({'score':rfc_val,
                                   'type':'val',
                                   'depth':depth})
    #Creating rfc_score_df
    rfc_scores = rfc_scores.append(val_rfc_scores)
    # train scores loc of ref_score_df
    train_acc=rfc_scores.loc[rfc_scores['type'] == 'train']
    # val scores loc of ref_score_df
    val_acc=rfc_scores.loc[rfc_scores['type'] == 'val']
    # train depth loc of rfc_score_df
    train_depth=rfc_scores.loc[rfc_scores['type']== 'train']['depth']
    # val depth loc of rfc_score_df
    val_depth=rfc_scores.loc[rfc_scores['type']== 'val']['depth']
    # rfc_scre_df 
    rfc_score_df= pd.DataFrame(
        {'train_score': train_acc.score,
        'val_score': val_acc.score,
         'train_depth': train_depth,
         'val_depth' : val_depth
        })
    # plot f, ax
    f, ax = plt.subplots(1, 1)
    
    #setting the title of chart
    plt.title("Random Forest Classifier Accuracy and Depth Chart")
    # plotting the data
    sns.pointplot(data =rfc_score_df, x='train_depth', y='train_score', label='Train',color='royalblue')
    sns.pointplot(data =rfc_score_df, x='val_depth', y='val_score', label='Val', color="seagreen")
    # setting the labels
    plt.ylabel('score')
    plt.xlabel('depth')
    #Showing the graph
    plt.show()


# # XGB Classifer Model Function

# In[22]:


def xgb_score(X_train, y_train, X_val, y_val):
    
    train_scores = []
    val_scores = []
    depth = []
    
    le = LabelEncoder()
    
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    
    for i in range(2,10):
        
        xgb = XGBClassifier(objective='multi:softmax',
                           seed=42,
                           max_depth=i,
                           learning_rate=.2,
                           gamma=.5,
                           reg_alpha=.75,
                           reg_lambda=.25,
                           min_child_weight=5,
                           max_leaves=4,
                           subsample=.6,
                           n_estimators=300)

        xgb.fit(X_train, y_train)
        
        train_scores.append(xgb.score(X_train, y_train))
        val_scores.append(xgb.score(X_val, y_val))
        depth.append(i)
        
        
    xgb_scores = pd.DataFrame({'score':train_scores,
                               'type':'train',
                               'depth':depth})
    #Creating val_rfc_score_df
    val_xgb_scores = pd.DataFrame({'score':val_scores,
                                   'type':'val',
                                   'depth':depth})
    
    xgb_scores = xgb_scores.append(val_xgb_scores)
    
    train_acc=xgb_scores.loc[xgb_scores['type'] == 'train']
    # val scores loc of ref_score_df
    val_acc=xgb_scores.loc[xgb_scores['type'] == 'val']
    # train depth loc of rfc_score_df
    train_depth=xgb_scores.loc[xgb_scores['type']== 'train']['depth']
    # val depth loc of rfc_score_df
    val_depth=xgb_scores.loc[xgb_scores['type']== 'val']['depth']
    # rfc_scre_df 
    xgb_score_df= pd.DataFrame(
        {'train_score': train_acc.score,
        'val_score': val_acc.score,
         'train_depth': train_depth,
         'val_depth' : val_depth
        })

    plt.title('XGBoost Classifier Accuracy and Depth Chart')
    
    sns.pointplot(data =xgb_score_df, x='train_depth', y='train_score', label='Train',color='royalblue')
    sns.pointplot(data =xgb_score_df, x='val_depth', y='val_score', label='Val', color="seagreen")
    
    # setting the labels
    plt.ylabel('score')
    plt.xlabel('depth')
    plt.show()


# # KNN Scores

# In[23]:


def knn_scores(X_train, y_train, X_val, y_val):
    '''
    Takes in x-train, y-train, x-val, and y-val 
    to make a plot of each score of x/y train and x/y val
    '''
    #Empty lists
    knn_train = []
    knn_val = []
    depth = []
    # Range of 2-21 and each value inbetween
    for i in range(2, 51):
         # KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')
        #fitting the model
        knn.fit(X_train, y_train)
        #y_pred array from prediciting X_train
        y_pred = knn.predict(X_train)
        #probability of y_pred array from prediciting X_train
        y_pred_proba = knn.predict_proba(X_train)
        #adding values to depth list
        depth.append(i)
        # adding values to knn_train and knn.val lists
        knn_train.append(knn.score(X_train, y_train))
        knn_val.append(knn.score(X_val, y_val))
    # knn_scores DataFrame
    knn_scores = pd.DataFrame({'score':knn_train,
                                   'type':'train',
                                   'depth':depth})
    #val_knn_scores DataFrame
    val_knn_scores = pd.DataFrame({'score':knn_val,
                                       'type':'val',
                                       'depth':depth})
    # Knn scores values added to val_knn_scores DataFrame
    knn_scores = knn_scores.append(val_knn_scores)
    # Creating the acc_dataframe from values of above DataFrames
    acc_df= pd.DataFrame(
            {'train_depth': knn_scores.depth[0:49], 
             'val_depth' : val_knn_scores.depth[0:49],
             'train_knn_score': knn_scores.score.values[0:49],
            'val_knn_score': val_knn_scores.score.values[0:49]
                  })
    # plot f, ax
    f, ax=plt.subplots(1,1)
    #setting the title of chart
    plt.title("KNN Scores Accuracy with Depth")
    # plotting the data
    sns.pointplot(x='train_depth', y='train_knn_score', data=acc_df, color='royalblue', label='Train')
    sns.pointplot(x='val_depth', y='val_knn_score', data=acc_df, color='seagreen', label='Val')
    # setting the labels
    plt.xlabel('Depth')
    plt.ylabel('Score')
    plt.xlim(0, 20)
    #Showing the graph
    plt.show()


# # Model Function

# In[24]:


def model_function_train_val(X_train, y_train, X_val, y_val):
    '''
    Function that returns the plot the accuracy of each model with the x/y train and x/y validate
    '''
    tree = decision_tree(X_train, y_train, X_val, y_val)
    rfc = random_for_class(X_train, y_train, X_val, y_val)
    xgb = xgb_score(X_train, y_train, X_val, y_val)
    knn = knn_scores(X_train, y_train, X_val, y_val)
    


# # Test Scores

# In[25]:


def test_score(X_train, y_train, X_test, y_test):
    '''
    Test Score function that 
    shows the Test Score Accuracy from XGB Classifier
    '''
    
    le = LabelEncoder()
    
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # XGB Boost Classifer
    xgb = XGBClassifier(objective='multi:softmax',
                           seed=42,
                           max_depth=8,
                           learning_rate=.2,
                           gamma=.5,
                           reg_alpha=.75,
                           reg_lambda=.25,
                           min_child_weight=5,
                           max_leaves=4,
                           subsample=.6,
                           n_estimators=300)
    
    # fitting the model on the x/y-train
    xgb.fit(X_train, y_train)
    # Score of the x/y-test
    score= xgb.score(X_test, y_test)
    #returning the score
    print(f'The final test score is {score:.2f}')


# # Test Against Baseline function

# In[26]:


def test_baseline(X_train, y_train, X_test, y_test, wine):
    '''
    Test Score Against Baseline function that 
    shows the plot of the Test Score Accuracy from RFC to our Baseline Model
    '''
    
    le = LabelEncoder()
    
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # XGB Boost Classifer
    xgb = XGBClassifier(objective='multi:softmax',
                           seed=42,
                           max_depth=8,
                           learning_rate=.2,
                           gamma=.5,
                           reg_alpha=.75,
                           reg_lambda=.25,
                           min_child_weight=5,
                           max_leaves=4,
                           subsample=.6,
                           n_estimators=300)
    
    # fitting the model on the x/y-train
    xgb.fit(X_train, y_train)
    # Score of the x/y-test
    score= xgb.score(X_test, y_test)
    
    baseline = (wine['quality']==6).mean()
    # Creating DataFrame of test score and baseline columns and values
    test_base_df=pd.DataFrame({'Test Score': [score],
                          'Baseline': [baseline]})
    # plot f, ax
    f, ax = plt.subplots(figsize=(8,6))
    #Setting the title of chart
    plt.title("Test Score Against Baseline Score")
    # plotting the data
    bplot = sns.barplot(data=test_base_df, palette='viridis')
    # setting the labels
    plt.ylabel('Score')
    ax.bar_label(bplot.containers[0], padding=4, fmt='%.2f')
    #Showing the graph
    plt.show()

