import numpy as np

def HDBSCAN(X, MIN_CLUSTER_SIZE=5, MIN_SAMPLES=5):

    def __COMPUTE_CORE_DISTANCES__(X):
        DISTANCES = PAIRWISE_DISTANCES(X)
        SORTED_DISTANCES = np.sort(DISTANCES, axis=0)
        CORE_DISTANCES = SORTED_DISTANCES[MIN_SAMPLES - 1]
        return CORE_DISTANCES

    def __COMPUTE_NEIGHBORS__(X, CORE_DISTANCES):
        NEIGHBORS = np.argwhere(PAIRWISE_DISTANCES(X) <= CORE_DISTANCES)[:, 1]
        return NEIGHBORS

    def __SINGLE_LINKAGE__(X, NEIGHBORS):
        CLUSTERER = {}
        for IDX, POINT in enumerate(X):
            if IDX in CLUSTERER:
                continue
            CLUSTER = set()
            CLUSTER.add(IDX)
            SEED_SET = set(NEIGHBORS[NEIGHBORS == IDX])
            while len(SEED_SET) > 0:
                CURRENT_POINT = SEED_SET.pop()
                if CURRENT_POINT in CLUSTERER:
                    continue
                CLUSTER.add(CURRENT_POINT)
                SEED_SET.update(NEIGHBORS[NEIGHBORS == CURRENT_POINT])
            for P in CLUSTER:
                CLUSTERER[P] = CLUSTER
        return CLUSTERER

    def __CONDENSE_TREE__(X, NEIGHBORS, CORE_DISTANCES):
        CONDENSED_TREE = {}
        for IDX, POINT in enumerate(X):
            NEIGHBORS = NEIGHBORS[NEIGHBORS == IDX]
            if len(NEIGHBORS) > 0:
                MIN_DISTANCE_IDX = np.argmin(CORE_DISTANCES[NEIGHBORS])
                NEAREST_NEIGHBOR = NEIGHBORS[MIN_DISTANCE_IDX]
                EDGE_LENGTH = CORE_DISTANCES[NEAREST_NEIGHBOR]
                CONDENSED_TREE[(IDX, NEAREST_NEIGHBOR)] = EDGE_LENGTH
        return CONDENSED_TREE

    def __EXTRACT_CLUSTERS__(CONDENSED_TREE, N_SAMPLES):
        LABELS = np.full(N_SAMPLES, -1)
        CLUSTER_ID = 0
        for EDGE, EDGE_LENGTH in sorted(CONDENSED_TREE.items(), key=lambda x: x[1]):
            CLUSTER_A, CLUSTER_B = EDGE
            if LABELS[CLUSTER_A] == -1 and LABELS[CLUSTER_B] == -1:
                LABELS[CLUSTER_A] = CLUSTER_ID
                LABELS[CLUSTER_B] = CLUSTER_ID
                CLUSTER_ID += 1
            elif LABELS[CLUSTER_A] == -1:
                LABELS[CLUSTER_A] = LABELS[CLUSTER_B]
            elif LABELS[CLUSTER_B] == -1:
                LABELS[CLUSTER_B] = LABELS[CLUSTER_A]
            elif LABELS[CLUSTER_A] != LABELS[CLUSTER_B]:
                MERGE_LABEL = min(LABELS[CLUSTER_A], LABELS[CLUSTER_B])
                LABELS[np.logical_or(LABELS == LABELS[CLUSTER_A], LABELS == LABELS[CLUSTER_B])] = MERGE_LABEL
        return __POST_PROCESS_LABELS__(LABELS)

    def __POST_PROCESS_LABELS__(LABELS):
        UNIQUE_LABELS = np.unique(LABELS)
        NEW_LABELS = -1 * np.ones_like(LABELS)
        for IDX, LABEL in enumerate(UNIQUE_LABELS):
            NEW_LABELS[LABELS == LABEL] = IDX
        return NEW_LABELS

    N_SAMPLES = X.shape[0]
    CORE_DISTANCES = __COMPUTE_CORE_DISTANCES__(X)
    NEIGHBORS = __COMPUTE_NEIGHBORS__(X, CORE_DISTANCES)
    CLUSTERER = __SINGLE_LINKAGE__(X, NEIGHBORS)
    CONDENSED_TREE = __CONDENSE_TREE__(X, NEIGHBORS, CORE_DISTANCES)
    LABELS = __EXTRACT_CLUSTERS__(CONDENSED_TREE, N_SAMPLES)
    return LABELS

def PAIRWISE_DISTANCES(X):
    N_SAMPLES = X.shape[0]
    DISTANCES = np.zeros((N_SAMPLES, N_SAMPLES))
    for i in range(N_SAMPLES):
        for j in range(i + 1, N_SAMPLES):
            DISTANCES[i, j] = DISTANCES[j, i] = np.linalg.norm(X[i] - X[j])
    return DISTANCES