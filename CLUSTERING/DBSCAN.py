import numpy as np

def DBSCAN(X, MIN_SAMPLES=5, EPSILON=0.5):
    N_SAMPLES, N_FEATURES = X.shape
    LABELS = np.zeros(N_SAMPLES, dtype=int)
    CLUSTER_ID = 0

    def __EXPAND_CLUSTER__(VISITED, DISTANCES, NEIGHBORS, CLUSTER_ID):
        for NEIGHBOR_INDEX in NEIGHBORS:
            if not VISITED[NEIGHBOR_INDEX]:
                VISITED[NEIGHBOR_INDEX] = True
                NEIGHBOR_NEIGHBORS = DISTANCES[NEIGHBOR_INDEX, 1:]
                NUM_NEIGHBOR_NEIGHBORS = len(NEIGHBOR_NEIGHBORS)
                if NUM_NEIGHBOR_NEIGHBORS >= MIN_SAMPLES:
                    NEIGHBORS.extend(NEIGHBOR_NEIGHBORS)
                if LABELS[NEIGHBOR_INDEX] == 0:
                    LABELS[NEIGHBOR_INDEX] = CLUSTER_ID

    DISTANCES = PAIRWISE_DISTANCES(X)
    VISITED = np.zeros(N_SAMPLES, dtype=bool)
    for i in range(N_SAMPLES):
        if VISITED[i]:
            continue
        VISITED[i] = True
        NEIGHBORS = np.where(DISTANCES[i, :] <= EPSILON)[0]
        NUM_NEIGHBORS = len(NEIGHBORS)
        if NUM_NEIGHBORS < MIN_SAMPLES:
            LABELS[i] = -1
        else:
            CLUSTER_ID += 1
            LABELS[i] = CLUSTER_ID
            __EXPAND_CLUSTER__(VISITED, DISTANCES, NEIGHBORS.tolist(), CLUSTER_ID)
    return LABELS

def PAIRWISE_DISTANCES(X):
    N_SAMPLES = X.shape[0]
    DISTANCES = np.zeros((N_SAMPLES, N_SAMPLES))
    for i in range(N_SAMPLES):
        for j in range(i + 1, N_SAMPLES):
            DISTANCES[i, j] = DISTANCES[j, i] = np.linalg.norm(X[i] - X[j])
    return DISTANCES