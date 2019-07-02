import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('iris.csv')


print(data["species"].value_counts())


nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns  = ['Null Count']
nulls.index.name  = 'Feature'
print(nulls)



x = data.iloc[:,0:-1]
y = data.iloc[:,-1]
print(x.shape,y.shape)

#elbow method to know the number of clusters

wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,7),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

km = KMeans(n_clusters=3)
km.fit(x)
y_cluster_kmeans= km.predict(x)
plt.scatter(x.iloc[:,0],x.iloc[:,1],c=y_cluster_kmeans,s=50,cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5)
plt.show()
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("#"*50)
print(score)
print("#"*50)
# standardization


pca = PCA(2)
x_pca = pca.fit_transform(x)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2,data[['species']]],axis=1)
print("#"*50)
print(finaldf)
print("#"*50)

# KMeans after standarization

km = KMeans(n_clusters=4)
km.fit(x_pca)
y_cluster_kmeans= km.predict(x_pca)
from sklearn import metrics
score = metrics.silhouette_score(x_pca, y_cluster_kmeans)
print("#"*50)
print(score)
print("#"*50)