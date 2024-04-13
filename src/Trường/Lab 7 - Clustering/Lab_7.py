# K-means Clustering

# 1. implementing K-means
# 1.1. Finding closest centroids
import numpy as np
def findClosestCentroids(X, centroids):
    idx = []
    for xi in X:
        distances = [np.linalg.norm(xi-centroid) for centroid in centroids]
        j = np.argmin(distances)
        idx.append(j)

    return np.array(idx)

X = np.array([[1, 2], [3, 4], [5, 6], [9, 11]])
centroids = np.array([[7, 5], [0, 2], [3, 3]])
idx = findClosestCentroids(X, centroids)
print(idx)

# 1.2. Computing centroid means
def computeCentroids(X, idx, K):
    new_centroids = []
    for j in range(K):
        Cj = X[idx == j]
        mu_j = Cj.sum(axis=0) / len(Cj)
        new_centroids.append(mu_j)

    return np.array(new_centroids)

idx = np.array([1, 2, 0, 0])
centroids = computeCentroids(X, idx, 3)
print(centroids)

# 2.K-means on example dataset
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat('datasets/lab7data1.mat')
X = mat['X']
np.random.shuffle(X)
print("X.shape:", X.shape)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], marker='.', color='blue')
# plt.show()

def Kmeans(X, K, max_iterations):
    random_ids = np.random.choice(len(X), K, replace=False)
    centroids = X[random_ids]

    for itr in range(1, max_iterations):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)

    return centroids

K = 3 # number of clusters (and centroids) that we want to get
max_iterations = 10 # number of iterations to perform
centroids = Kmeans(X, K, max_iterations)

idx = findClosestCentroids(X, centroids)
fig, ax = plt.subplots()
ax.scatter(X[idx==0][:, 0], X[idx==0][:, 1], marker='1', color='blue', label='Cluster 1')
ax.scatter(X[idx==1][:, 0], X[idx==1][:, 1], marker='2', color='green', label='Cluster 2')
ax.scatter(X[idx==2][:, 0], X[idx==2][:, 1], marker='3', color='red', label='Cluster 3')
ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroids')
plt.legend()
# plt.show()

# 3. Image compression with K-means
# 3.1. K-means on pixels
A = plt.imread("datasets/bird_small.png")
print("A.shape:", A.shape)

X = A.reshape(A.shape[0] * A.shape[1], A.shape[2])
print("X.shape:", X.shape)

print("Performing Kmeans ... This may take some time ...")
centroids = Kmeans(X, 16, 20)

# For each pixel (X[i]) in X, we find its closest centroid (idx[i])
idx = findClosestCentroids(X, centroids)

# Create a new matrix XX where each data-point is replaced by its closest centroid
XX = np.array([centroids[i] for i in idx])

# Reshape XX back into an image (matrix AA of dimension p*q*d)
AA = XX.reshape(128, 128, 3)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(A)
ax1.axis("off")
ax1.set_title("Original image")

ax2.imshow(AA)
ax2.axis("off")
ax2.set_title("Compressed image")

plt.show()