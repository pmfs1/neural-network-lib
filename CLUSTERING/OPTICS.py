import numpy as np

def optics_clustering(data, min_samples, epsilon):
    def calculate_distances(X):
        return squareform(pdist(X))

    def calculate_core_distances(distances, min_samples):
        return np.sort(distances, axis=0)[min_samples]

    def calculate_reachability_distances(distances, core_distances, epsilon):
        n_points = distances.shape[0]
        reachability_distances = np.full(n_points, np.inf)

        for point in range(n_points):
            if core_distances[point] < np.inf:
                neighborhood = np.where(distances[point] <= epsilon)[0]
                reachability_distances[point] = np.max(
                    np.maximum(core_distances[point], distances[point][neighborhood])
                )

        return reachability_distances

    def extract_clusters(distances, reachability_distances, epsilon, core_distances):
        n_points = distances.shape[0]
        ordered_points = np.argsort(reachability_distances)
        clusters = []

        for point in ordered_points:
            if reachability_distances[point] > epsilon:
                if core_distances[point] <= epsilon:
                    current_cluster = [point]

                    # Expand the cluster
                    index = np.where(ordered_points == point)[0][0] + 1
                    while index < n_points and reachability_distances[ordered_points[index]] <= epsilon:
                        current_cluster.append(ordered_points[index])
                        index += 1

                    clusters.append(current_cluster)

        return clusters

    # Calculate distances
    distances = calculate_distances(data)

    # Calculate core distances
    core_distances = calculate_core_distances(distances, min_samples)

    # Calculate reachability distances
    reachability_distances = calculate_reachability_distances(distances, core_distances, epsilon)

    # Extract clusters
    clusters = extract_clusters(distances, reachability_distances, epsilon, core_distances)

    return clusters

def pdist(X):
    n = X.shape[0]
    pairwise_distances = np.zeros((n * (n - 1)) // 2)

    index = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            distance = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            pairwise_distances[index] = distance
            index += 1

    return pairwise_distances

def squareform(pairwise_distances):
    n = int(np.sqrt(2 * len(pairwise_distances)) + 0.5)
    square_distances = np.zeros((n, n))

    index = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            square_distances[i, j] = pairwise_distances[index]
            square_distances[j, i] = pairwise_distances[index]
            index += 1

    return square_distances