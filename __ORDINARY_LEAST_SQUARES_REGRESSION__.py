import numpy as np

# ORDINARY_LEAST_SQUARES_REGRESSION: IMPLEMENTATION OF ORDINARY LEAST SQUARES REGRESSION. ORDINARY LEAST SQUARES (OLS) REGRESSION IS A METHOD FOR ESTIMATING THE UNKNOWN PARAMETERS IN A LINEAR REGRESSION MODEL. OLS REGRESSION FINDS THE PARAMETER VALUES THAT MINIMIZE THE SUM OF THE SQUARED DIFFERENCES BETWEEN THE OBSERVED AND PREDICTED VALUES.
class ORDINARY_LEAST_SQUARES_REGRESSION:
    # INITIALIZES THE ORDINARY LEAST SQUARES REGRESSION MODEL.
    def __init__(self):
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ORDINARY LEAST SQUARES REGRESSION MODEL.
        self.WEIGHTS = None
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE ORDINARY LEAST SQUARES REGRESSION MODEL.
        self.BIAS = 0

    # FIT(): IT'S THE FUNCTION THAT TRAINS THE ORDINARY LEAST SQUARES REGRESSION MODEL.
    def FIT(self, X, Y):
        # X: IT'S THE MATRIX OF FEATURES.
        X = np.array(X)
        # Y: IT'S THE VECTOR OF TARGETS.
        Y = np.array(Y)
        # X_T: IT'S THE TRANSPOSED MATRIX OF FEATURES.
        X_T = X.T
        # X_T_X: IT'S THE PRODUCT BETWEEN THE TRANSPOSED MATRIX OF FEATURES AND THE MATRIX OF FEATURES.
        X_T_X = np.dot(X_T, X)
        # X_T_Y: IT'S THE PRODUCT BETWEEN THE TRANSPOSED MATRIX OF FEATURES AND THE VECTOR OF TARGETS.
        X_T_Y = np.dot(X_T, Y)
        # W: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ORDINARY LEAST SQUARES REGRESSION MODEL.
        W = np.dot(np.linalg.inv(X_T_X), X_T_Y)
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ORDINARY LEAST SQUARES REGRESSION MODEL.
        self.WEIGHTS = W
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE ORDINARY LEAST SQUARES REGRESSION MODEL.
        self.BIAS = np.mean(Y) - np.dot(np.mean(X, axis=0), W)

    # PREDICT(): IT'S THE FUNCTION THAT USES THE ORDINARY LEAST SQUARES REGRESSION MODEL IN ORDER TO PREDICT NEW OUTPUT VALUES.
    def PREDICT(self, X):
        # RETURN THE PREDICTED OUTPUT VALUE.
        return np.dot(X, self.WEIGHTS) + self.BIAS