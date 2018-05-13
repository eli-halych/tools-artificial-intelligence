import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# print(X)

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()

# clf = KMeans(n_clusters=6)
# clf.fit(X)
#
# centroids = clf.cluster_centers_
# labels = clf.labels_

colors = 10*["g", "r", "c", "b", "k"]
# print(colors)

# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=15)
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
# plt.show()


class K_Means:
    def __init__(self, k=3, tol=0.001, max_iter=300): # tolerance - how much the centroid is gonna move
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]
            # print(self.centroids)

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in X: # X will work because it is defined above, it should be data
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                print(distances)
                classification = distances.index(min(distances))
                print(classification)
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                # print(self.centroids)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # print(c)
                # print(original_centroid)
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    # print(np.sum((current_centroid-original_centroid)/original_centroid*100.0)) # num of iterations it went through in %
                    optimized = False

            if optimized:
                break




    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)


unknowns = np.array([[1, 3],
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]])

# for unknown in unknowns:
#     classification = clf.predict(unknown)
#     plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)



plt.show()

