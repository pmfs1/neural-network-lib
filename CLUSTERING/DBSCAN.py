import numpy as np

def DBSCAN(X, MIN_SAMPLES=5, EPSILON=0.5):
    N_SAMPLES = X.shape[0]
    LABELS = np.zeros(N_SAMPLES, dtype=int)
    CLUSTER_ID = 0

    def __EXPAND_CLUSTER__(VISITED, DISTANCES, NEIGHBOURS, CLUSTER_ID):
        for NEIGHBOUR_INDEX in NEIGHBOURS:
            if not VISITED[NEIGHBOUR_INDEX]:
                VISITED[NEIGHBOUR_INDEX] = True
                NEIGHBOUR_NEIGHBOURS = DISTANCES[NEIGHBOUR_INDEX, 1:]
                NUM_NEIGHBOUR_NEIGHBOURS = len(NEIGHBOUR_NEIGHBOURS)
                if NUM_NEIGHBOUR_NEIGHBOURS >= MIN_SAMPLES:
                    NEIGHBOURS.extend(NEIGHBOUR_NEIGHBOURS)
                if LABELS[NEIGHBOUR_INDEX] == 0:
                    LABELS[NEIGHBOUR_INDEX] = CLUSTER_ID

    DISTANCES = PAIRWISE_DISTANCES(X)
    VISITED = np.zeros(N_SAMPLES, dtype=bool)
    for i in range(N_SAMPLES):
        if VISITED[i]:
            continue
        VISITED[i] = True
        NEIGHBOURS = np.where(DISTANCES[i, :] <= EPSILON)[0]
        NUM_NEIGHBOURS = len(NEIGHBOURS)
        if NUM_NEIGHBOURS < MIN_SAMPLES:
            LABELS[i] = -1
        else:
            CLUSTER_ID += 1
            LABELS[i] = CLUSTER_ID
            __EXPAND_CLUSTER__(VISITED, DISTANCES, NEIGHBOURS.tolist(), CLUSTER_ID)
    return LABELS

def PAIRWISE_DISTANCES(X):
    N_SAMPLES = X.shape[0]
    DISTANCES = np.zeros((N_SAMPLES, N_SAMPLES))
    for i in range(N_SAMPLES):
        for j in range(i + 1, N_SAMPLES):
            DISTANCES[i, j] = DISTANCES[j, i] = np.linalg.norm(X[i] - X[j])
    return DISTANCES