# Please do not change function names and number of parameters.
# Please check the assignment requirements what function should return.
import random
import numpy as np


# Ben Heinze
def k_means_clustering(data_matrix, k, epsilon):
    # 1) randomly place k means, but she gives us initial centroid locations:
    mini = np.min(data_matrix) # gets min value
    maxi = np.max(data_matrix) # gets max value
    k_means = []
    # Loops through correct number of means and generates random values for them
    for _ in range(k):
        mean = []
        for _ in data_matrix[0]:
            mean.append(float(random.uniform(mini, maxi)))
        k_means.append(mean)
    # Converts generated values into a numpy array
    k_means = np.array(k_means)
    # 1.2) get the current k_means average
    currentMeanAverage = np.average(k_means)
    while True:
        # 2) For each data point, assign that point to the nearest Centroid
        # ==================================================================================================================
        # 2.1) create empty matrix to store each point's centroid assignment (0 being fist centroid, 1 being second)
        clusterIndexes = np.zeros((len(data_matrix),1)) # same size as data matrix
        # 2.2) Find the shortest centroid and assign point to that index
        for pointIndex, point in enumerate(data_matrix, 0):
            shortestDist = float('inf')
            # loops through every centroid then assigns the closest centroid to shortestDist
            for centroidIndex, center in enumerate(k_means, 0): # index stores which index of centroid that point is-
                # -closest to

                # calculating Euclidean distance using linalg.norm()
                newDist = np.linalg.norm(point - center)
                # if distance from this centroid is shorter than current, replace current
                if newDist < shortestDist:
                    shortestDist = newDist
                    clusterIndexes[pointIndex] = centroidIndex
        print(f"Cluster assignments: {clusterIndexes.T}") # .T transposes it so it prints horizontally instead of
        # vertically

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

            # iterate through every col, take mean, then reassign to the original k-mean
            for i in range(len(k_means[0])):
                col = [p[i] for p in pointsInCluster]
                if sum(col) != 0:
                    mean = sum(col) / len(pointsInCluster)
                else:
                    mean = 0
                newMean[i] = mean
            k_means[clusterIndex] = newMean
        newMeanAverage = np.average(k_means)
        # Breaks if the change in averaages is less than epsilon
        if (currentMeanAverage - newMeanAverage) < epsilon:
            break
        else:
            print("nice")
            currentMeanAverage = newMeanAverage
    print(f"K-Means:\n{k_means}")
    print(f"Cluster assignments:\n {clusterIndexes.T}")

# Braxton McCormack
def dbscan_clustering(data_matrix, minpts, epsilon):
    n_points = len(data_matrix)
    labels = -1 * np.ones(n_points)  # Initialize all labels as -1 (noise points)
    
    def neighbors(point_idx):
        """Calculate neighbors within epsilon distance."""
        return np.where(np.linalg.norm(data_matrix - data_matrix[point_idx], axis=1) <= epsilon)[0]
    
    cluster_id = 0
    for point_idx in range(n_points):
        if labels[point_idx] != -1:
            continue  # Point already processed
        
        # Find all points within the epsilon radius
        point_neighbors = neighbors(point_idx)
        
        # Not enough points to form a core point
        if len(point_neighbors) < minpts:
            labels[point_idx] = -1  # Label as noise (unnecessary as it's initialized to -1)
        else:
            # Current point is a core point
            labels[point_idx] = cluster_id  # Assign to a new cluster
            i = 0
            while i < len(point_neighbors):  # Process every point in the neighborhood
                neighbor_idx = point_neighbors[i]
                if labels[neighbor_idx] == -1:  # Noise point becomes border point
                    labels[neighbor_idx] = cluster_id
                elif labels[neighbor_idx] != -1:  # Already processed and labeled
                    i += 1
                    continue
                else:  # New point found
                    labels[neighbor_idx] = cluster_id
                    new_neighbors = neighbors(neighbor_idx)
                    if len(new_neighbors) >= minpts:  # New core point
                        point_neighbors = np.concatenate([point_neighbors, new_neighbors])
                i += 1
            cluster_id += 1  # Increment for the next cluster
            
    return labels

# Braxton McCormack
def compute_clustering_precision(true_labels, cluster_labels):
    from collections import Counter

    # Check if the input lists have the same length
    if len(true_labels) != len(cluster_labels):
        raise ValueError("The length of true_labels and cluster_labels must be the same.")

    # Mapping from cluster label to true labels in that cluster
    cluster_to_true_labels = {}
    for true_label, cluster_label in zip(true_labels, cluster_labels):
        if cluster_label not in cluster_to_true_labels:
            cluster_to_true_labels[cluster_label] = []
        cluster_to_true_labels[cluster_label].append(true_label)

    # Calculate precision for each cluster
    total_correct = 0
    total_points = 0
    for cluster, labels in cluster_to_true_labels.items():
        if cluster == -1:  # Skip noise points if using DBSCAN or similar
            continue
        label_count = Counter(labels)
        most_common_label, most_common_count = label_count.most_common(1)[0]
        total_correct += most_common_count
        total_points += len(labels)

    # Overall precision is the sum of correct labels divided by total labels (excluding noise points)
    precision = total_correct / total_points if total_points > 0 else 0
    return precision


def main():
    pass

main()