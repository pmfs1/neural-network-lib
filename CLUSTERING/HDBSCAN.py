import numpy as np

def HDBSCAN(X, MIN_SAMPLES=5):

    def __COMPUTE_CORE_DISTANCES__(X):
        DISTANCES = PAIRWISE_DISTANCES(X)
        SORTED_DISTANCES = np.sort(DISTANCES, axis=0)
        CORE_DISTANCES = SORTED_DISTANCES[MIN_SAMPLES - 1]
        return CORE_DISTANCES

    def __COMPUTE_NEIGHBOURS__(X, CORE_DISTANCES):
        NEIGHBOURS = np.argwhere(PAIRWISE_DISTANCES(X) <= CORE_DISTANCES)[:, 1]
        return NEIGHBOURS

    def __CONDENSE_TREE__(X, NEIGHBOURS, CORE_DISTANCES):
        CONDENSED_TREE = {}
        for IDX, _ in enumerate(X):
            NEIGHBOURS = NEIGHBOURS[NEIGHBOURS == IDX]
            if len(NEIGHBOURS) > 0:
                MIN_DISTANCE_IDX = np.argmin(CORE_DISTANCES[NEIGHBOURS])
                NEAREST_NEIGHBOUR = NEIGHBOURS[MIN_DISTANCE_IDX]
                EDGE_LENGTH = CORE_DISTANCES[NEAREST_NEIGHBOUR]
                CONDENSED_TREE[(IDX, NEAREST_NEIGHBOUR)] = EDGE_LENGTH
        return CONDENSED_TREE

    def __EXTRACT_CLUSTERS__(CONDENSED_TREE, N_SAMPLES):
        LABELS = np.full(N_SAMPLES, -1)
        CLUSTER_ID = 0
        for EDGE, _ in sorted(CONDENSED_TREE.items(), key=lambda x: x[1]):
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
    NEIGHBOURS = __COMPUTE_NEIGHBOURS__(X, CORE_DISTANCES)
    CONDENSED_TREE = __CONDENSE_TREE__(X, NEIGHBOURS, CORE_DISTANCES)
    LABELS = __EXTRACT_CLUSTERS__(CONDENSED_TREE, N_SAMPLES)
    return LABELS

def PAIRWISE_DISTANCES(X):
    N_SAMPLES = X.shape[0]
    DISTANCES = np.zeros((N_SAMPLES, N_SAMPLES))
    for i in range(N_SAMPLES):
        for j in range(i + 1, N_SAMPLES):
            DISTANCES[i, j] = DISTANCES[j, i] = np.linalg.norm(X[i] - X[j])
    return DISTANCES