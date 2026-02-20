from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data  

        #  dimensionality reduction to 2 
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)  
print(x_pca)


sse = []
k_range = range(1, 10)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(x)  
    sse.append(model.inertia_)

plt.plot(k_range, sse, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE / Inertia")
plt.title("Elbow Method (Iris Dataset)")
plt.show()





optimal_k = 3           # from elbow method point
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(x)           # fit on all features
labels = kmeans.labels_
centers = kmeans.cluster_centers_

        # reduction , so that shape match to x_pca
centers_pca = pca.transform(centers)



        # plotting
plt.scatter(x_pca[labels == 0, 0], x_pca[labels == 0, 1], color='r', label=f'Cluster {1}')
plt.scatter(x_pca[labels == 1, 0], x_pca[labels == 1, 1], color='g', label=f'Cluster {2}')
plt.scatter(x_pca[labels == 2, 0], x_pca[labels == 2, 1], color='y', label=f'Cluster {3}')

        # Plot centroids, s for centorid size
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='+', s=200, color='black', label='Centroids')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering with PCA (Iris Dataset)")
plt.legend()
plt.show()
