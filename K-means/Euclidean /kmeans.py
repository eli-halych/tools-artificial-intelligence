from sklearn import datasets
from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt




class Kmeans:
    def __init__(self, k):

        iris = datasets.load_iris()
        self.X = iris.data[:, :3]
        self.y = iris.target
        self.features = []
        for p in self.X:
            self.features.append(Feature(p))

        # self.features.append(Feature([-1, -1, -1]))
        # self.features.append(Feature([1, 1, 1]))
        # self.features.append(Feature([0, 1, 2]))
        # self.features.append(Feature([3, 2, 4]))
        # self.features.append(Feature([4, 1, 3]))

        self.clusters = []
        for i in range(k):
            self.clusters.append(Cluster())

        # self.clusters[0].addElement(self.features[0])
        # self.clusters[0].addElement(self.features[1])
        # self.clusters[0].addElement(self.features[4])
        # self.clusters[1].addElement(self.features[2])
        # self.clusters[1].addElement(self.features[3])

        for i in range(k):
            for j in range(50):
                self.clusters[i].addElement(self.features[j])

    def placeCentroids(self):
        for cluster in self.clusters:
            for element in cluster.getElements():
                cluster.centroid.addInNumerator(element)  # print([x + y for x, y in zip([1, 2], [1, 2])])
                cluster.centroid.lengthOfNumerator += 1
            cluster.centroid.setCoordinates()


    def calculateDistances(self):
        for p in self.features:
            dist = 0
            for i in range(k):
                centroid = self.clusters[i].getCentroid()
                centr_coord = centroid.coordinates
                dist_temp = self.euclideanDistance(p, centr_coord)
                if (dist != 0 and dist_temp < dist) or (dist == 0):
                    dist = dist_temp
                    p.addOrUpdateCentroid(centroid)

    def euclideanDistance(self, feature, centroid):
        sum = 0
        for i in range(len(feature.coordinates)):
            sum += (feature.coordinates[i]-centroid[i])**2
        return sum

    def updateClusters(self):
        self.emptyClusters()

        for cluster in self.clusters:
            for feature in self.features:
                if cluster.centroid == feature.centroid:
                    cluster.addElement(feature)

    def emptyClusters(self):
        for cluster in self.clusters:
            cluster.removeElements()

    def showClusters(self):
        for cluster in self.clusters:
            for el in cluster.elements:
                print(el.coordinates)
            print("-")

    def getSumInCluster(self):
        for i in range(len(self.clusters)):
            self.clusters[i].showSqSumDistances(i)

    def centroidsDontMove(self):
        return False


class Centroid:
    def __init__(self):
        self.numerator = [0, 0, 0]
        self.lengthOfNumerator = 0
        self.coordinates = [0, 0, 0]

    def addInNumerator(self, feature):
        self.numerator = [x + y for x, y in zip(self.numerator, feature.coordinates)]

    def setCoordinates(self):
        self.coordinates = [x / self.lengthOfNumerator for x in self.numerator]
        # print(self.coordinates)







class Cluster:
    def __init__(self):
        self.centroid = Centroid()
        self.elements = []

    def addElement(self, element):
        self.elements.append(element)

    def getElements(self):
        return self.elements

    def getCentroid(self):
        return self.centroid

    def removeElements(self):
        self.elements.clear()

    def showSqSumDistances(self, cl_num):
        sq_sum = 0
        for e in self.elements:
            sq_sum += self.euclideanDistance(e, self.centroid)
        print(str(cl_num)+"cluster: "+str(sq_sum))

    def euclideanDistance(self, element, centroid):
        dist = 0
        for i in range(len(element.coordinates)):
            dist += (element.coordinates[i] - centroid.coordinates[i]) ** 2
        return dist






class Feature(object):
    def __init__(self, vector):
        self.coordinates = vector

    def addOrUpdateCentroid(self, c):
        self.centroid = c











k = 3
clf = Kmeans(k)
X = clf.X
y = clf.y

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()
colors = 10*["g", "r", "c", "b", "k"]
















# fit, possibly just one iteration needed
for i in range(50):
    clf.placeCentroids()
    clf.calculateDistances()
    clf.updateClusters()
    clf.getSumInCluster()
    # clf.showClusters()
    print("-----")

    # if clf.centroidsDontMove():
    #     break





















# plot fitted centroids
centroids = []
for cluster in clf.clusters:
    centroids.append(cluster.centroid.coordinates)
for centroid in centroids:
    plt.scatter(centroid[0], centroid[1],
                marker="o", color="k", s=150, linewidths=5)


# plot flowers
i=1
for cluster in clf.clusters:
    i+=1
    color = colors[i]
    for p in cluster.elements:
        plt.scatter(p.coordinates[0], p.coordinates[1], marker="x", color=color, s=150, linewidths=5)

plt.show()

