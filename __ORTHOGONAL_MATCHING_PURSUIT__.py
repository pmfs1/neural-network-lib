import numpy as np

# ORTHOGONAL_MATCHING_PURSUIT: IMPLEMENTATION OF THE ORTHOGONAL MATCHING PURSUIT ALGORITHM. ORTHOGONAL MATCHING PURSUIT (OMP) IS A GREEDY ALGORITHM THAT SOLVES THE LINEAR APPROXIMATION PROBLEM. IN EACH ITERATION, IT CHOOSES THE COLUMN OF THE MATRIX THAT IS MOST HIGHLY CORRELATED WITH THE CURRENT RESIDUAL. THE ALGORITHM TERMINATES WHEN THE NUMBER OF SELECTED COLUMNS REACHES THE DESIRED NUMBER OF FEATURES.
class ORTHOGONAL_MATCHING_PURSUIT:
    # INITIALIZES THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
    def __init__(self, NUMBER_OF_FEATURES):
        # NUMBER_OF_FEATURES: IT'S THE NUMBER OF FEATURES.
        self.NUMBER_OF_FEATURES = NUMBER_OF_FEATURES
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
        self.WEIGHTS = None
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
        self.BIAS = 0

    # FIT(): IT'S THE FUNCTION THAT TRAINS THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
    def FIT(self, X, Y):
        # X: IT'S THE MATRIX OF FEATURES.
        X = np.array(X)
        # Y: IT'S THE VECTOR OF TARGETS.
        Y = np.array(Y)
        # NUMBER_OF_FEATURES: IT'S THE NUMBER OF FEATURES.
        NUMBER_OF_FEATURES = len(X[0])
        # W: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
        W = np.zeros(NUMBER_OF_FEATURES)
        # R: IT'S THE RESIDUAL.
        R = Y
        # I: IT'S THE SET OF SELECTED FEATURES.
        I = []
        # FOR EACH FEATURE.
        for _ in range(self.NUMBER_OF_FEATURES):
            # C: IT'S THE VECTOR OF CORRELATIONS.
            C = np.dot(X.T, R)
            # J: IT'S THE INDEX OF THE MOST CORRELATED FEATURE.
            J = np.argmax(np.abs(C))
            # I: IT'S THE SET OF SELECTED FEATURES.
            I.append(J)
            # X_I: IT'S THE MATRIX OF SELECTED FEATURES.
            X_I = X[:, I]
            # W_I: IT'S THE VECTOR OF WEIGHTS.
            W_I = np.dot(np.linalg.inv(np.dot(X_I.T, X_I)), np.dot(X_I.T, Y))
            # R: IT'S THE RESIDUAL.
            R = Y - np.dot(X_I, W_I)
            # W: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
            W[I] = W_I
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
        self.WEIGHTS = W
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE ORTHOGONAL MATCHING PURSUIT ALGORITHM.
        self.BIAS = np.mean(Y) - np.dot(np.mean(X, axis=0), W)

    # PREDICT(): IT'S THE FUNCTION THAT USES THE ORTHOGONAL MATCHING PURSUIT ALGORITHM FOR PREDICTION.
    def PREDICT(self, X):
        # RETURNS THE VECTOR OF PREDICTIONS.
        return np.dot(X, self.WEIGHTS) + self.BIAS