import numpy as np

# RIDGE_REGRESSION: IMPLEMENTS THE RIDGE REGRESSION MODEL. IT'S A LINEAR REGRESSION MODEL WHERE THE LOSS FUNCTION IS MODIFIED TO MINIMIZE THE COMPLEXITY OF THE MODEL. THIS IS DONE BY ADDING A PENALTY TERM TO THE LOSS FUNCTION. THE COEFFICIENTS ARE ESTIMATED USING THE ORDINARY LEAST SQUARES METHOD. THE RIDGE REGRESSION MODEL IS PARTICULARLY USEFUL TO ALLEVIATE THE PROBLEM OF MULTICOLLINEARITY IN LINEAR REGRESSION, WHICH COMMONLY OCCURS IN MODELS WITH LARGE NUMBERS OF PARAMETERS. IN RIDGE REGRESSION, THE COST FUNCTION IS ALTERED BY ADDING A PENALTY EQUIVALENT TO SQUARE OF THE MAGNITUDE OF THE COEFFICIENTS. MEANING, IT'S A LINEAR LEAST SQUARES MODEL WITH L2 REGULARIZATION.
class RIDGE_REGRESSION:
    # INITIALIZES THE RIDGE REGRESSION MODEL.
    def __init__(self, FIT_INTERCEPT=True, ALPHA=1.0, N_ITER=300):
        # FIT_INTERCEPT: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO WHETHER TO CALCULATE THE INTERCEPT FOR THIS MODEL. IF SET TO FALSE, NO INTERCEPT WILL BE USED IN CALCULATIONS (E.G. DATA IS EXPECTED TO BE ALREADY CENTERED).
        self.FIT_INTERCEPT = FIT_INTERCEPT
        # ALPHA: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE REGULARIZATION STRENGTH. IT MUST BE A POSITIVE FLOAT. REGULARIZATION IMPROVES THE CONDITIONING OF THE PROBLEM AND REDUCES THE VARIANCE OF THE ESTIMATES. LARGER VALUES SPECIFY STRONGER REGULARIZATION.
        self.ALPHA = ALPHA
        # N_ITER: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS FOR THE OPTIMIZATION.
        self.N_ITER = N_ITER
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        self.WEIGHTS = None
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE LINEAR REGRESSION MODEL.
        self.BIAS = None

    # FIT(): FITS THE RIDGE REGRESSION MODEL TO THE TRAINING DATA
    def FIT(self, X, Y):
        # CHECK IF THE FIT_INTERCEPT PARAMETER IS SET TO TRUE.
        if self.FIT_INTERCEPT:
            # IF FIT_INTERCEPT IS SET TO TRUE, THEN ADD A COLUMN OF ONES TO THE TRAINING DATA MATRIX.
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # CALCULATE THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        self.WEIGHTS = np.linalg.inv(X.T.dot(X) + self.ALPHA * np.identity(X.shape[1])).dot(X.T).dot(Y)
        # IF FIT_INTERCEPT IS SET TO TRUE, THEN THE FIRST ELEMENT OF THE WEIGHTS ARRAY IS THE BIAS.
        if self.FIT_INTERCEPT:
            self.BIAS = self.WEIGHTS[0]
            self.WEIGHTS = self.WEIGHTS[1:]
    
    # TRANSFORM(): PREDICTS THE LABELS OF THE TRAINING DATA USING THE LINEAR REGRESSION MODEL.
    def TRANSFORM(self, X):
        # CHECK IF THE FIT_INTERCEPT PARAMETER IS SET TO TRUE.
        if self.FIT_INTERCEPT:
            # IF FIT_INTERCEPT IS SET TO TRUE, THEN ADD A COLUMN OF ONES TO THE TRAINING DATA MATRIX.
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # RETURN THE PREDICTIONS OF THE LINEAR REGRESSION MODEL.
        return X.dot(self.WEIGHTS) + self.BIAS if self.FIT_INTERCEPT else X.dot(self.WEIGHTS)