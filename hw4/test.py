import numpy as np

def k_means(X, K, initial_centroids, max_iters=100):
    centroids = initial_centroids
    for i in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, K)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = np.sqrt(((x - centroids)**2).sum(axis=1))
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

def update_centroids(X, clusters, K):
    new_centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        new_centroids[k] = X[np.array(clusters) == k].mean(axis=0)
    return new_centroids

# Updated data points with new point (5, 6)
X = np.array([(1, 2), (3, 4), (7, 0), (10, 2), (5, 6)])

# Number of clusters
K = 2

# Initial centroids set 1
initial_centroids_1 = np.array([(1, 2), (10, 2)])
centroids_1, clusters_1 = k_means(X, K, initial_centroids_1)
print("Final centroids with initial centroids 1:", centroids_1)
print("Cluster assignments with initial centroids 1:", clusters_1)
err_1 = 0; err_2 = 0
for i in range(len(X)):
    if clusters_1[i] == 0:
        err_1 += ((X[i][0] - initial_centroids_1[0][0])**2 + (X[i][1] - initial_centroids_1[0][1])**2)
    else: 
        err_1 += ((X[i][0] - initial_centroids_1[1][0])**2 + (X[i][1] - initial_centroids_1[1][1])**2)        

print("minimum with initial centroids 1:", err_1)

# Initial centroids set 2
initial_centroids_2 = np.array([(1, 2), (3, 4)])
centroids_2, clusters_2 = k_means(X, K, initial_centroids_2)
print("Final centroids with initial centroids 2:", centroids_2)
print("Cluster assignments with initial centroids 2:", clusters_2)
for i in range(len(X)):
    if clusters_2[i] == 0:
        err_2 += ((X[i][0] - initial_centroids_2[0][0])**2 + (X[i][1] - initial_centroids_2[0][1])**2)
    else: 
        err_2 += ((X[i][0] - initial_centroids_2[1][0])**2 + (X[i][1] - initial_centroids_2[1][1])**2)        

print("minimum with initial centroids 2:", err_2)
