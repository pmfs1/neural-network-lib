import numpy as np

def OPTICS(X, MIN_SAMPLES, EPSILON=None):

    def __CALCULATE_DISTANCES__(X):
        return SQUAREFORM(PDIST(X))

    def __CALCULATE_CORE_DISTANCES__(DISTANCES, MIN_SAMPLES):
        return np.sort(DISTANCES, axis=0)[MIN_SAMPLES]

    def __CALCULATE_REACHABILITY_DISTANCES__(DISTANCES, CORE_DISTANCES, EPSILON):
        N_POINTS = DISTANCES.shape[0]
        REACHABILITY_DISTANCES = np.full(N_POINTS, np.inf)
        for POINT in range(N_POINTS):
            if CORE_DISTANCES[POINT] < np.inf:
                NEIGHBORHOOD = np.where(DISTANCES[POINT] <= EPSILON)[0]
                REACHABILITY_DISTANCES[POINT] = np.max(np.maximum(CORE_DISTANCES[POINT], DISTANCES[POINT][NEIGHBORHOOD]))
        return REACHABILITY_DISTANCES

    def __EXTRACT_CLUSTERS__(DISTANCES, REACHABILITY_DISTANCES, EPSILON, CORE_DISTANCES):
        N_POINTS = DISTANCES.shape[0]
        ORDERED_POINTS = np.argsort(REACHABILITY_DISTANCES)
        CLUSTERS = []
        for POINT in ORDERED_POINTS:
            if REACHABILITY_DISTANCES[POINT] > EPSILON:
                if CORE_DISTANCES[POINT] <= EPSILON:
                    CURRENT_CLUSTER = [POINT]
                    IDX = np.where(ORDERED_POINTS == POINT)[0][0] + 1
                    while IDX < N_POINTS and REACHABILITY_DISTANCES[ORDERED_POINTS[IDX]] <= EPSILON:
                        CURRENT_CLUSTER.append(ORDERED_POINTS[IDX])
                        IDX += 1
                    CLUSTERS.append(CURRENT_CLUSTER)
        return CLUSTERS

    DISTANCES = __CALCULATE_DISTANCES__(X)
    if EPSILON is None:
        AVG_DISTANCE = np.mean(DISTANCES)
        EPSILON = AVG_DISTANCE * 0.5
    CORE_DISTANCES = __CALCULATE_CORE_DISTANCES__(DISTANCES, MIN_SAMPLES)
    REACHABILITY_DISTANCES = __CALCULATE_REACHABILITY_DISTANCES__(DISTANCES, CORE_DISTANCES, EPSILON)
    CLUSTERS = __EXTRACT_CLUSTERS__(DISTANCES, REACHABILITY_DISTANCES, EPSILON, CORE_DISTANCES)
    return CLUSTERS

def PDIST(X):
    N = X.shape[0]
    PAIRWISE_DISTANCES = np.zeros((N * (N - 1)) // 2)
    IDX = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            DISTANCE = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            PAIRWISE_DISTANCES[IDX] = DISTANCE
            IDX += 1
    return PAIRWISE_DISTANCES

def SQUAREFORM(PAIRWISE_DISTANCES):
    N = int(np.sqrt(2 * len(PAIRWISE_DISTANCES)) + 0.5)
    SQUARE_DISTANCES = np.zeros((N, N))
    IDX = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            SQUARE_DISTANCES[i, j] = PAIRWISE_DISTANCES[IDX]
            SQUARE_DISTANCES[j, i] = PAIRWISE_DISTANCES[IDX]
            IDX += 1
    return SQUARE_DISTANCES