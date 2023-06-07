import numpy as np

# LINEAR_DISCRIMINANT_ANALYSIS: CLASS THAT IMPLEMENTS LINEAR DISCRIMINANT ANALYSIS MODEL
class LINEAR_DISCRIMINANT_ANALYSIS:
    # INITIALIZES THE LINEAR DISCRIMINANT ANALYSIS MODEL
    def __init__(self, N_COMPONENTS):
        self.N_COMPONENTS = N_COMPONENTS # NUMBER OF COMPONENTS: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF COMPONENTS IN THE DATASET.
        self.LINEAR_DISCRIMINANTS = None # LINEAR DISCRIMINANTS: IT'S THE MATRIX THAT CONTAINS THE LINEAR DISCRIMINANTS.

    # FIT(): METHOD THAT TRAINS THE LINEAR DISCRIMINANT ANALYSIS MODEL
    def FIT(self, X, Y):
        N_FEATURES = X.shape[1] # NUMBER OF FEATURES: IT'S THE NUMBER OF FEATURES IN THE DATASET.
        CLASS_LABELS = np.unique(Y) # CLASS LABELS: IT'S THE ARRAY THAT CONTAINS THE UNIQUE CLASS LABELS IN THE DATASET.
        MEAN_OVERALL = np.mean(X, axis=0) # MEAN OVERALL: IT'S THE MEAN OF THE DATASET.
        SW = np.zeros((N_FEATURES, N_FEATURES)) # SW: IT'S THE WITHIN-CLASS SCATTER MATRIX.
        SB = np.zeros((N_FEATURES, N_FEATURES)) # SB: IT'S THE BETWEEN-CLASS SCATTER MATRIX.
        for C in CLASS_LABELS: # FOR EVERY CLASS LABEL IN THE CLASS LABELS ARRAY
            X_C = X[Y == C] # X_C: IT'S THE SUBSET OF THE DATASET THAT CONTAINS THE SAMPLES WITH THE CLASS LABEL C.
            MEAN_C = np.mean(X_C, axis=0) # MEAN_C: IT'S THE MEAN OF THE SUBSET OF THE DATASET THAT CONTAINS THE SAMPLES WITH THE CLASS LABEL C.
            SW += (X_C - MEAN_C).T.dot((X_C - MEAN_C)) # SW = SW + (X_C - MEAN_C).T * (X_C - MEAN_C)
            N_C = X_C.shape[0] # N_C: IT'S THE NUMBER OF SAMPLES WITH THE CLASS LABEL C.
            MEAN_DIFF = (MEAN_C - MEAN_OVERALL).reshape(N_FEATURES, 1) # MEAN_DIFF: IT'S THE DIFFERENCE BETWEEN THE MEAN_C AND THE MEAN_OVERALL.
            SB += N_C * (MEAN_DIFF).dot(MEAN_DIFF.T) # SB = SB + N_C * (MEAN_DIFF).T * (MEAN_DIFF)
        A = np.linalg.inv(SW).dot(SB) # A: IT'S THE MATRIX THAT CONTAINS THE EIGENVECTORS OF THE LINEAR DISCRIMINANTS.
        EIGENVALUES, EIGENVECTORS = np.linalg.eig(A) # EIGENVALUES: IT'S THE ARRAY THAT CONTAINS THE EIGENVALUES OF THE LINEAR DISCRIMINANTS.
        EIGENVECTORS = EIGENVECTORS.T # EIGENVECTORS: IT'S THE MATRIX THAT CONTAINS THE EIGENVECTORS OF THE LINEAR DISCRIMINANTS.
        INDEX_LENGTHS = np.argsort(abs(EIGENVALUES))[::-1] # INDEX_LENGTHS: IT'S THE ARRAY THAT CONTAINS THE INDEXES OF THE EIGENVALUES IN DESCENDING ORDER.
        EIGENVALUES = EIGENVALUES[INDEX_LENGTHS] # EIGENVALUES: IT'S THE ARRAY THAT CONTAINS THE EIGENVALUES OF THE LINEAR DISCRIMINANTS IN DESCENDING ORDER.
        EIGENVECTORS = EIGENVECTORS[INDEX_LENGTHS] # EIGENVECTORS: IT'S THE MATRIX THAT CONTAINS THE EIGENVECTORS OF THE LINEAR DISCRIMINANTS IN DESCENDING ORDER.
        self.LINEAR_DISCRIMINANTS = EIGENVECTORS[0:self.N_COMPONENTS] # LINEAR_DISCRIMINANTS = EIGENVECTORS[0:N_COMPONENTS]

    # TRANSFORM(): METHOD THAT TRANSFORMS THE DATASET
    def TRANSFORM(self, X):
        return np.dot(X, self.LINEAR_DISCRIMINANTS.T) # RETURNS THE TRANSFORMED DATASET: X * LINEAR_DISCRIMINANTS.T