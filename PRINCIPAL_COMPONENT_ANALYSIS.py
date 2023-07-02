import numpy as np

# PRINCIPAL_COMPONENT_ANALYSIS: CLASS THAT IMPLEMENTS PRINCIPAL COMPONENT ANALYSIS
class PRINCIPAL_COMPONENT_ANALYSIS:
    # INITIALIZES THE PRINCIPAL COMPONENT ANALYSIS MODEL
    def __init__(self, N_COMPONENTS):
        # N_COMPONENTS: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF COMPONENTS.
        self.N_COMPONENTS = N_COMPONENTS
        # COMPONENTS: IT'S THE LIST THAT CONTAINS THE COMPONENTS OF THE MODEL.
        self.COMPONENTS = None
        # MEAN: IT'S THE LIST THAT CONTAINS THE MEAN OF THE MODEL.
        self.MEAN = None

    # FIT(): METHOD THAT TRAINS THE PRINCIPAL COMPONENT ANALYSIS MODEL
    def FIT(self, X):
        self.MEAN = np.mean(X, axis=0)  # CALCULATES THE MEAN
        X = X - self.MEAN  # CENTERS THE DATA: X - MEAN
        COV = np.cov(X.T)  # CALCULATES THE COVARIANCE MATRIX: X.T * X
        # CALCULATES THE EIGENVALUES AND THE EIGENVECTORS: COV * EIGENVECTORS = EIGENVALUES * EIGENVECTORS
        EIGENVALUES, EIGENVECTORS = np.linalg.eig(COV)
        EIGENVECTORS = EIGENVECTORS.T  # TRANSPOSES THE EIGENVECTORS
        # CALCULATES THE INDEXES OF THE SORTED EIGENVALUES
        INDEXES = np.argsort(EIGENVALUES)[::-1]
        EIGENVALUES = EIGENVALUES[INDEXES]  # SORTS THE EIGENVALUES
        EIGENVECTORS = EIGENVECTORS[INDEXES]  # SORTS THE EIGENVECTORS
        # SELECTS THE COMPONENTS
        self.COMPONENTS = EIGENVECTORS[0:self.N_COMPONENTS]

    # TRANSFORM(): METHOD THAT TRANSFORMS THE DATASET INTO A MATRIX OF COMPONENTS
    def TRANSFORM(self, X):
        X = X - self.MEAN  # CENTERS THE DATA: X - MEAN
        # RETURNS THE MATRIX OF COMPONENTS: X * COMPONENTS.T
        return np.dot(X, self.COMPONENTS.T)