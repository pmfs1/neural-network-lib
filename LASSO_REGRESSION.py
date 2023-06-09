import numpy as np

from .NEURAL_NETWORKS.REGULARIZERS import L1

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

    # FIT(): IT'S THE FUNCTION THAT TRAINS THE LASSO REGRESSION MODEL.
    def FIT(self, X, Y):
        # N: NUMBER OF DATA POINTS.
        # M: NUMBER OF FEATURES.
        N, M = X.shape
        # IF FIT_INTERCEPT IS TRUE THEN
        if self.FIT_INTERCEPT:
            # X: AUGMENTED X.
            X = np.hstack((X, np.ones((N, 1))))
            # M: NUMBER OF FEATURES.
            M += 1
        # INITIALIZE WEIGHTS.
        self.WEIGHTS = np.random.uniform(-1.0, 1.0, M)
        # INITIALIZE BIAS.
        self.BIAS = np.random.uniform(-1.0, 1.0)
        # INITIALIZE L1 REGULARIZER.
        L1_REGULARIZER = L1(C=self.ALPHA)
        # INITIALIZE GRADIENT.
        GRADIENT = L1_REGULARIZER.__GRAD__(self.WEIGHTS)
        # FOR EACH ITERATION
        for _ in range(self.N_ITER):
            # COMPUTE THE PREDICTIONS.
            Y_HAT = X @ self.WEIGHTS + self.BIAS
            # COMPUTE THE ERROR.
            ERROR = Y_HAT - Y
            # COMPUTE THE GRADIENT.
            GRADIENT = L1_REGULARIZER.__GRAD__(self.WEIGHTS)
            # UPDATE THE WEIGHTS.
            self.WEIGHTS -= 0.01 * GRADIENT
            # UPDATE THE BIAS.
            self.BIAS -= 0.01 * np.sum(ERROR)
        # RETURN THE LASSO REGRESSION MODEL.
        return self
    
    # TRANSFORM(): IT'S THE FUNCTION THAT USES THE LASSO REGRESSION MODEL IN ORDER TO PREDICT NEW DATA.
    def TRANSFORM(self, X):
        # IF FIT_INTERCEPT IS TRUE THEN
        if self.FIT_INTERCEPT:
            # X: AUGMENTED X.
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        # RETURN THE PREDICTIONS.
        return X @ self.WEIGHTS + self.BIAS