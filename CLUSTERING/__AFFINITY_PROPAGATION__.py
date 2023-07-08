import numpy as np

def AFFINITY_PROPAGATION(X, MAX_ITERATIONS=200, CONVERGENCE_ITERATIONS=15, D_AMPING=0.5):
    N_SAMPLES = X.shape[0]
    S = -np.inf * np.ones((N_SAMPLES, N_SAMPLES))
    A = np.zeros((N_SAMPLES, N_SAMPLES))
    R = np.zeros((N_SAMPLES, N_SAMPLES))
    for i in range(N_SAMPLES):
        for j in range(N_SAMPLES):
            S[i, j] = -np.linalg.norm(X[i] - X[j])
    for _ in range(MAX_ITERATIONS):
        R_OLD = R.copy()
        AS = A + S
        MAX_IDXS = np.argmax(AS, axis=1)
        MAX_VALUES = AS[np.arange(N_SAMPLES), MAX_IDXS]
        AS[np.arange(N_SAMPLES), MAX_IDXS] = -np.inf
        SECOND_MAX_VALUES = np.amax(AS, axis=1)
        R = S - np.repeat(MAX_VALUES[:, np.newaxis], N_SAMPLES, axis=1)
        R[np.arange(N_SAMPLES), MAX_IDXS] = S[np.arange(N_SAMPLES), MAX_IDXS] - SECOND_MAX_VALUES
        A_OLD = A.copy()
        R_P = np.maximum(R, 0)
        R_P[np.arange(N_SAMPLES), np.arange(N_SAMPLES)] = R[np.arange(N_SAMPLES), np.arange(N_SAMPLES)]
        A = np.sum(R_P, axis=0)
        D_A = np.diag(A)
        A = np.minimum(A, 0)
        A[np.arange(N_SAMPLES), np.arange(N_SAMPLES)] = np.diag(D_A)
        if np.allclose(R, R_OLD, atol=1e-6) and np.allclose(A, A_OLD, atol=1e-6):
            CONVERGENCE_ITERATIONS -= 1
            if CONVERGENCE_ITERATIONS == 0:
                break
    CLUSTER_CENTER_IDXS = np.where(np.diag(A + R) > 0)[0]
    LABELS = np.argmax(S[:, CLUSTER_CENTER_IDXS], axis=1)
    LABELS[CLUSTER_CENTER_IDXS] = np.arange(len(CLUSTER_CENTER_IDXS))
    return CLUSTER_CENTER_IDXS, LABELS