import numpy as np

from .NEURAL_NETWORKS.REGULARIZERS import L2

# RIDGE_REGRESSION: IMPLEMENTS THE RIDGE REGRESSION MODEL. (ALSO KNOWN AS TIKHONOV REGULARIZATION.) IT'S A LINEAR REGRESSION MODEL WHERE THE LOSS FUNCTION IS MODIFIED TO MINIMIZE THE COMPLEXITY OF THE MODEL. THIS IS DONE BY ADDING A PENALTY TERM TO THE LOSS FUNCTION. THE COEFFICIENTS ARE ESTIMATED USING THE ORDINARY LEAST SQUARES METHOD. THE RIDGE REGRESSION MODEL IS PARTICULARLY USEFUL TO ALLEVIATE THE PROBLEM OF MULTICOLLINEARITY IN LINEAR REGRESSION, WHICH COMMONLY OCCURS IN MODELS WITH LARGE NUMBERS OF PARAMETERS. IN RIDGE REGRESSION, THE COST FUNCTION IS ALTERED BY ADDING A PENALTY EQUIVALENT TO SQUARE OF THE MAGNITUDE OF THE COEFFICIENTS. MEANING, IT'S A LINEAR LEAST SQUARES MODEL WITH L2 REGULARIZATION.
class RIDGE_REGRESSION:
    # INITIALIZES THE RIDGE REGRESSION MODEL.
    def __init__(self, FIT_INTERCEPT=True, ALPHA=1.0):
        # FIT_INTERCEPT: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO WHETHER TO CALCULATE THE INTERCEPT FOR THIS MODEL. IF SET TO FALSE, NO INTERCEPT WILL BE USED IN CALCULATIONS (E.G. DATA IS EXPECTED TO BE ALREADY CENTERED).
        self.FIT_INTERCEPT = FIT_INTERCEPT
        # ALPHA: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE REGULARIZATION STRENGTH. IT MUST BE A POSITIVE FLOAT. REGULARIZATION IMPROVES THE CONDITIONING OF THE PROBLEM AND REDUCES THE VARIANCE OF THE ESTIMATES. LARGER VALUES SPECIFY STRONGER REGULARIZATION.
        self.ALPHA = ALPHA
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        self.WEIGHTS = None
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE LINEAR REGRESSION MODEL.
        self.BIAS = None

    # FIT(): IT'S THE FUNCTION THAT TRAINS THE RIDGE REGRESSION MODEL.
    def FIT(self, X, Y):
        # X: IT'S THE PARAMETER THAT CORRESPONDS TO THE FEATURES OF THE TRAINING SET.
        # Y: IT'S THE PARAMETER THAT CORRESPONDS TO THE TARGETS OF THE TRAINING SET.
        # N: IT'S THE NUMBER OF SAMPLES IN THE TRAINING SET.
        N = len(X)
        # D: IT'S THE NUMBER OF FEATURES IN THE TRAINING SET.
        D = len(X[0])
        # L2_REGULARIZATION: IT'S THE L2 REGULARIZATION OBJECT.
        L2_REGULARIZATION = L2(C=self.ALPHA)
        # IF FIT_INTERCEPT IS TRUE
        if self.FIT_INTERCEPT:
            # ADD A COLUMN OF ONES TO THE FEATURES OF THE TRAINING SET.
            X = np.hstack((np.ones((N, 1)), X))
            # D: IT'S THE NUMBER OF FEATURES IN THE TRAINING SET.
            D += 1
        # COMPUTE THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        self.WEIGHTS = np.linalg.inv(X.T @ X + L2_REGULARIZATION(N=D)) @ X.T @ Y
        # IF FIT_INTERCEPT IS TRUE
        if self.FIT_INTERCEPT:
            # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE LINEAR REGRESSION MODEL.
            self.BIAS = self.WEIGHTS[0]
            # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
            self.WEIGHTS = self.WEIGHTS[1:]

    # TRANSFORM(): IT'S THE FUNCTION THAT USES THE LINEAR REGRESSION MODEL TO MAKE PREDICTIONS.
    def TRANSFORM(self, X):
        # X: IT'S THE PARAMETER THAT CORRESPONDS TO THE FEATURES OF THE TEST SET.
        # IF FIT_INTERCEPT IS TRUE
        if self.FIT_INTERCEPT:
            # RETURN THE PREDICTIONS MADE BY THE LINEAR REGRESSION MODEL.
            return self.BIAS + X @ self.WEIGHTS
        # RETURN THE PREDICTIONS MADE BY THE LINEAR REGRESSION MODEL.
        return X @ self.WEIGHTS