

# # Clustering & GDP of World Happiness Report
# 
# This kernel shows basic visualization of data using Choropleth maps. Further, it tries to cluster the data using few clustering algorithms including K-means and Guassian Mixture Model based on several factors such as GDP per capita, life expectancy, corruption etc. We have considered 2017 data only.


#Call required libraries
import time                   # To time processes
import warnings               # To suppress warnings

import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics
import seaborn as sns
import chart_studio.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn.preprocessing import StandardScaler  # For scaling dataset
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering
from sklearn.mixture import GaussianMixture #For GMM clustering

import os                     # For os related operations
import sys                    # For data size


data = pd.read_csv("TestData1.csv") #Read the dataset



data


data.describe()


data.info()




print("Dimension of dataset: wh.shape")
data.dtypes


# ### Basic Visualization
#
# Correlation among variables
# 
# First, we will try to understand the correlation between few variables. For this, first compute the correlation matrix among the variables and plotted as heat map.
# FIrst of all we are converting or checking the numarical values that we have in our dataset.



num_vars = data.select_dtypes(exclude="O").columns.to_list()
print("Numerical variables:", num_vars)



wh1 = data[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.', 'Freedom', 
          'Generosity','Trust..Government.Corruption.','Dystopia.Residual']] #Subsetting the data
cor = wh1.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = True) #Plot the correlation as heat map


# We have obtained the heatmap of correlation among the variables. The color palette in the side represents the amount of correlation among the variables. The lighter shade represents high correlation. We can see that happiness score is highly correlated with GDP per capita, family and life expectancy. It is least correlated with generosity.
# 
# add Codeadd Markdown
# Visualization of Happiness Score: Using Choropleth feature
# 
# add Codeadd Markdown
# We will try to plot the happiness score of countries in the world map. Hovering the mouse over the country shows the name of the country as well as its happiness score.







# # User-defined function in Python
# 
# Before we proceed further, we will see the basics of defining a function (or user-defined function) in Python:
# 
# The user-defined function starts with a keyword def to declare the function and this is followed by adding the function name.
# def plus
# 
# Pass the arguments in the function and is provided within parantheses () and close the statement with colon: .
# def plus(x,y):
# 
# Add the statements that need to be executed.
# z = sum(x,y)
# 
# End the function with return statement to see the output. If return statement is not provided, there will be no output.
# return (z)
# 
# So, the complete function is
# 
# def plus(x,y):
# 
#  z = sum (x,y)
#   return(z)
# a = plus(2,5) #Calling the function to add two numbers
# 
# So, whenever we execute a = plus(2,5), it would return a = 7.
# 
# For more details, refer https://www.datacamp.com/community/tutorials/functions-python-tutorial
# 
# add Codeadd Markdown
# Clustering Of Countries
# 
# We are considering eight parameters, namely, happiness score, GDP per capita, family, life expectancy, freedom, generosity, corruption and Dystopia residual for clustering the countries. Since the clustering is sensitive to range of data. It is advisable to scale the data before proceeding.



#Scaling of data
ss = StandardScaler()
ss.fit_transform(wh1)


# ### **(1) k-means clustering**
# 
# In general, k-means is the first choice for clustering because of its simplicity. Here, the user has to define the number of clusters (Post on how to decide the number of clusters would be dealt later). The clusters are formed based on the closeness to the center value of the clusters. The initial center value is chosen randomly.  K-means clustering is top-down approach, in the sense, we decide the number of clusters (k) and then group the data points into k clusters.




#K means Clustering 
def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(wh1, 2)
kmeans = pd.DataFrame(clust_labels)
wh1.insert((wh1.shape[1]),'kmeans',kmeans)



#Plot the clusters obtained using k means
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)


# ----------------------------------------------------------------------------------

# ### (2) Agglomerative Clustering
# 
# Also known as Hierarchical clustering, does not require the user to specify the number of clusters. Initially, each point is considered as a separate cluster, then it recursively clusters the points together depending upon the distance between them. The points are clustered in such a way that the distance between points within a cluster is minimum and distance between the cluster is maximum. Commonly used distance measures are Euclidean distance, Manhattan distance or Mahalanobis distance. Unlike k-means clustering, it is "bottom-up" approach.
# 
# Python Tip: Though providing the number of clusters is not necessary but Python provides an option of providing the same for easy and simple use.



def doAgglomerative(X, nclust=2):
    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'euclidean', linkage = 'ward')
    clust_labels1 = model.fit_predict(X)
    return (clust_labels1)

clust_labels1 = doAgglomerative(wh1, 2)
agglomerative = pd.DataFrame(clust_labels1)
wh1.insert((wh1.shape[1]),'agglomerative',agglomerative)



# Plot the clusters obtained using Agglomerative clustering or Hierarchical clustering
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=agglomerative[0],s=50)
ax.set_title('Agglomerative Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)


# # ----------------------------------------------------------------------------------------
# 
# 
# (3) Affinity Propagation
# 
# It does not require the number of cluster to be estimated and provided before starting the algorithm. It makes no assumption regarding the internal structure of the data points. For further details on clustering, refer http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/

# In[80]:


def doAffinity(X):
    model = AffinityPropagation(damping = 0.5, max_iter = 250, affinity = 'euclidean')
    model.fit(X)
    clust_labels2 = model.predict(X)
    cent2 = model.cluster_centers_
    return (clust_labels2, cent2)

clust_labels2, cent2 = doAffinity(wh1)
affinity = pd.DataFrame(clust_labels2)
wh1.insert((wh1.shape[1]),'affinity',affinity)


# In[81]:


#Plotting the cluster obtained using Affinity algorithm
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=affinity[0],s=50)
ax.set_title('Affinity Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)


# ### (4) Guassian Mixture Modelling
# 
# It is probabilistic based clustering or kernel density estimation based clusterig. The clusters are formed based on the Gaussian distribution of the centers. For further details and pictorial description, refer https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/mixture.html

# 


def doGMM(X, nclust=2):
    model = GaussianMixture(n_components=nclust,init_params='kmeans')
    model.fit(X)
    clust_labels3 = model.predict(X)
    return (clust_labels3)

clust_labels3 = doGMM(wh1,2)
gmm = pd.DataFrame(clust_labels3)
wh1.insert((wh1.shape[1]),'gmm',gmm)




#Plotting the cluster obtained using GMM
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(wh1['Economy..GDP.per.Capita.'],wh1['Trust..Government.Corruption.'],
                     c=gmm[0],s=50)
ax.set_title('Affinity Clustering')
ax.set_xlabel('GDP per Capita')
ax.set_ylabel('Corruption')
plt.colorbar(scatter)


# ### Visualization of countries based on the clustering results
# 
# (1) k-Means algorithm



# wh1.insert(0,'Country',data.iloc[:,0])
wh1.iloc[:,[0,9,10,11,12]]
data = [dict(type='choropleth',
             locations = wh1['Country'],
             locationmode = 'country names',
             z = wh1['kmeans'],
             text = wh1['Country'],
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Clustering of Countries based on K-Means',
              geo=dict(showframe = False,
                       projection = {'type':'airy'}))
map1 = go.Figure(data = data, layout=layout)
iplot(map1)


# ### (2) Agglomerative Clustering

# In[90]:


data = [dict(type='choropleth',
             locations = wh1['Country'],
             locationmode = 'country names',
             z = wh1['agglomerative'],
             text = wh1['Country'],
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Grouping of Countries based on Agglomerative Clustering',
              geo=dict(showframe = False, 
                       projection = {'type':'airy'}))
map2 = dict(data=data, layout=layout)
iplot(map2)


# ### (3) Affinity Propagation

# In[93]:


data = [dict(type='choropleth',
             locations = wh1['Country'],
             locationmode = 'country names',
             z = wh1['affinity'],
             text = wh1['Country'],
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Grouping of Countries based on Affinity Clustering',
              geo=dict(showframe = False, projection = {'type':'airy'}))
map3 = dict(data=data, layout=layout)
iplot(map3)


# ### *(4) GMM*
# 

# In[122]:


data = [dict(type='choropleth',
             locations = wh1['Country'],
             locationmode = 'country names',
             z = wh1['gmm'],
             text = wh1['Country'],
             colorbar = {'title':'Cluster Group'})]
layout = dict(title='Grouping of Countries based on GMM clustering',
              geo=dict(showframe = False, projection = {'type':'airy'}))
map4 = dict(data=data, layout=layout)
iplot(map4)


# # GDP



from ipywidgets import interact,widgets # for interactive visualization
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import ploting Libraries
import seaborn as sns
import matplotlib.pyplot as plt




num_vars = wh1.select_dtypes(exclude="O").columns.to_list()
print("Numerical variables:", num_vars)




#
plt.figure(figsize=(15,40))
nrows = len(num_vars)
ncols = 1
c = 1 

for var in num_vars:
    if var != 'Year':
        plt.subplot(nrows,ncols, c)
        wh1.groupby("Country").mean()[var].sort_values(ascending =False).plot(kind='bar')
        plt.title(f"{var} for last 10 Years")
    c=c+1
plt.tight_layout()
plt.show()






