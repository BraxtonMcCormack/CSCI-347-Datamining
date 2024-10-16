#Please do not change function names and number of parameters. 
#Please check the assignment requirements what function should return. 
import random
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def k_means_clustering(data_matrix, k, epsilon):
    k_means = np.array([[3.90304975, 2.51839422] ,
                        [-1.21226502 , 3.6765421 ]])
    # # 1) randomly place k means
    # minRange = data_matrix.min()
    # maxRange = data_matrix.max()
    # # Get range min and max range from m (in nxm) and randomly place means
    # k_means = []        # array of arrays
    # for i, k in enumerate(data_matrix):
    #     randomX = np.random.uniform(minRange, maxRange)     # generates random x in range
    #     randomY = np.random.uniform(minRange, maxRange)     # generates random y in range
    #     k_means.append(np.array([ randomX,randomY]))
    # k_means = np.array(k_means)

    # 1.2) get the current k_means average
    count = 0
    sum1 = 0
    for row in k_means:
        for col in row:
            sum1 += col
            count += 1
    currentAverage = sum1/count
    while True:
        # 2) For each data point, assign that point to the nearest Centroid
        # ==================================================================================================================
        # 2.1) create empty matrix to store each point's centroid assignment (0 being fist centroid, 1 being second)
        clusterIndexes = np.zeros((len(data_matrix),1)) #same size as data matrix
         # 2.2) Find the shortest centroid and assign point to that index
        for pointIndex, point in enumerate(data_matrix, 0):
            shortestDist = float('inf')
            #loops through every centroid then assigns the closest centroid to shortestDist
            for centroidIndex, center in enumerate(k_means, 0): #index stores which index of centroid that point is closes# t to
                # calculating Euclidean distance using linalg.norm()
                newDist = np.linalg.norm(point - center)
                # if distance from this centroid is shorter than current, replace current
                if newDist < shortestDist:
                    shortestDist = newDist
                    clusterIndexes[pointIndex] = centroidIndex
        #print(f"Cluster assignments: {clusterIndexes.T}") #.T transposes it so it prints horizontally instead of vertically

        # 3) Recalculate Centroids based on current cluster assignments
        # ==================================================================================================================
        # 3.1) Finds every point in single cluster
        for clusterIndex in range(k):
            newMean = np.copy(k_means[clusterIndex])
            pointsInCluster = []
            # if the assignment of the datapoint and the cluster index match, store the point
            for point, assignment in zip(data_matrix, clusterIndexes):
                if clusterIndex == assignment:
                    pointsInCluster.append(point)

            #iterate through every col, take mean, then reassign to the original k-mean
            for i in range(len(k_means[0])):
                col = [p[i] for p in pointsInCluster]
                mean = sum(col) / len(pointsInCluster)
                newMean[i] = mean
            k_means[clusterIndex] = newMean
        #calculates new average of k_means
        count2 = 0
        sum2 = 0
        for row in k_means:
            for col in row:
                sum2 += col
                count2 += 1
        newAverage = sum2 / count2
        # compares old ave and new avg, breaks if < epsilon
        if (currentAverage - newAverage) < epsilon:
            break
        else:
            currentAverage = newAverage
    print(f"K-Means:\n{k_means}")
    print(f"Cluster assignments:\n{clusterIndexes.T}")

def dbscan_clustering(data_matrix, minpts, epsilon):
    # Function body left blank
    pass

def compute_clustering_precision(true_labels, cluster_labels):
    # Function body left blank
    pass


def main():
    # Test 1
    D = np.array([[-1, 2], [-2, -2], [3, 2], [5, 4.3], [-3, 3], [-3, -1], [5, -3], [3, 4], [2.3, 6.5]])
    k_means_clustering(D, 2, 0.000001)
    # # Test 2
    # D = np.array([[-1, 2], [-2, -2], [3, 2], [5, 4.3], [-3, 3], [-3, -1], [5, -3], [3, 4], [2.3, 6.5], [4, 2], [4, 4],
    #               [-2.3, 1.5]])
    # k_means_clustering(D, 3, 0.000001)
    # x, y = make_blobs(n_samples=300, centers=3, random_state=35)
    # data_matrix = np.column_stack((x, y))
    # k_means_clustering(data_matrix, 3, 0.00000001)
    pass



main()