from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()

X = iris.data[:, [0, 2]]   # sepal length & petal length

        #sum of square error,   measure distance of each data to it's nearest centroid
sse = []        # it will store inertia values 
k_range = range(1, 10)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)            # only on X, as it si unsupervosed algorithm
    sse.append(model.inertia_)

plt.plot(k_range, sse, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE / Inertia")
plt.title("Elbow Method")
plt.show()


            # we will take cluster number from the above elbow metho point, which will be best k 
kmeans = KMeans(n_clusters=3, random_state=66)
kmeans.fit(X)

labels = kmeans.labels_                 
centers = kmeans.cluster_centers_       # coordinates of centroids





                # plotting
        # means select only those data points that has labels 0,1,2 . from the column 0 which is sepal_length and 1 for petal_length
plt.scatter(X[labels == 0, 0], X[labels == 0, 1])
plt.scatter(X[labels == 1, 0], X[labels == 1, 1])
plt.scatter(X[labels == 2, 0], X[labels == 2, 1])

plt.scatter(centers[:, 0], centers[:, 1], marker='+', s=200)    # s for centroid size
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("K-Means Clustering (Iris Dataset)")
plt.show()
