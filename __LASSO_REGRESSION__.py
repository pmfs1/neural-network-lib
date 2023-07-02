import numpy as np

# LASSO_REGRESSION: IMPLEMENTS THE LASSO REGRESSION MODEL. WHICH IS A REGRESSION ANALYSIS METHOD THAT PERFORMS BOTH VARIABLE SELECTION AND REGULARIZATION IN ORDER TO ENHANCE THE PREDICTION ACCURACY AND INTERPRETABILITY OF THE RESULTING STATISTICAL MODEL. THE LASSO REGRESSION MODEL IS PARTICULARLY USEFUL TO ALLEVIATE THE PROBLEM OF MULTICOLLINEARITY IN LINEAR REGRESSION, WHICH COMMONLY OCCURS IN MODELS WITH LARGE NUMBERS OF PARAMETERS. MEANING, IT'S A LINEAR MODEL TRAINED WITH L1 PRIOR AS REGULARIZER.
class LASSO_REGRESSION:
    # INITIALIZES THE LASSO REGRESSION MODEL.
    def __init__(self, FIT_INTERCEPT=True, ALPHA=1.0, N_ITER=1000):
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

    # FIT(): FITS THE LASSO REGRESSION MODEL TO THE TRAINING DATA
    def FIT(self, X, Y):
        # CHECK IF THE FIT_INTERCEPT PARAMETER IS SET TO TRUE.
        if self.FIT_INTERCEPT:
            # IF FIT_INTERCEPT IS SET TO TRUE, THEN ADD A COLUMN OF ONES TO THE TRAINING DATA MATRIX.
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # CALCULATE THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        self.WEIGHTS = np.zeros(X.shape[1])
        # CALCULATE THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        for _ in range(self.N_ITER):
            # CALCULATE THE GRADIENT OF THE WEIGHTS.
            GRADIENT = self.__GRADIENT__(X, Y)
            # UPDATE THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
            self.WEIGHTS = self.WEIGHTS - self.ALPHA * GRADIENT
        # IF FIT_INTERCEPT IS SET TO TRUE, THEN THE FIRST ELEMENT OF THE WEIGHTS ARRAY IS THE BIAS.
        if self.FIT_INTERCEPT:
            self.BIAS = self.WEIGHTS[0]
            self.WEIGHTS = self.WEIGHTS[1:]
    
    # TRANSFORM(): PREDICTS THE LABELS OF THE DATA SAMPLES.
    def TRANSFORM(self, X):
        # CHECK IF THE FIT_INTERCEPT PARAMETER IS SET TO TRUE.
        if self.FIT_INTERCEPT:
            # IF FIT_INTERCEPT IS SET TO TRUE, THEN ADD A COLUMN OF ONES TO THE DATA MATRIX.
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # CALCULATE THE PREDICTIONS.
        PREDICTIONS = np.dot(X, self.WEIGHTS)
        # RETURN THE PREDICTIONS.
        return PREDICTIONS
    
    # __GRADIENT__() [PRIVATE FUNCTION]: CALCULATES THE GRADIENT OF THE WEIGHTS.
    def __GRADIENT__(self, X, Y):
        # CALCULATE THE PREDICTIONS.
        PREDICTIONS = np.dot(X, self.WEIGHTS)
        # CALCULATE THE DIFFERENCE BETWEEN THE PREDICTIONS AND THE ACTUAL LABELS.
        DIFFERENCE = PREDICTIONS - Y
        # CALCULATE THE GRADIENT OF THE WEIGHTS.
        GRADIENT = np.dot(X.T, DIFFERENCE) / X.shape[0]
        # RETURN THE GRADIENT OF THE WEIGHTS.
        return GRADIENT