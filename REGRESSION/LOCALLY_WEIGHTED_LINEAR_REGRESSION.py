import numpy as np

# LOCALLY_WEIGHTED_LINEAR_REGRESSION: IMPLEMENTS THE LOCALLY WEIGHTED LINEAR REGRESSION ALGORITHM. WHICH IS A NON-PARAMETRIC ALGORITHM THAT USES A GAUSSIAN KERNEL TO CALCULATE THE WEIGHTS FOR EACH TRAINING EXAMPLE. THE WEIGHTS ARE USED TO CALCULATE THE PARAMETERS OF THE LINEAR REGRESSION MODEL FOR EACH PREDICTION.


class LOCALLY_WEIGHTED_LINEAR_REGRESSION:
    """IMPLEMENTS THE LOCALLY WEIGHTED LINEAR REGRESSION ALGORITHM. WHICH IS A NON-PARAMETRIC ALGORITHM THAT USES A GAUSSIAN KERNEL TO CALCULATE THE WEIGHTS FOR EACH TRAINING EXAMPLE. THE WEIGHTS ARE USED TO CALCULATE THE PARAMETERS OF THE LINEAR REGRESSION MODEL FOR EACH PREDICTION.

    ATTRIBUTES
    ----------
    TAU: FLOAT (DEFAULT: 0.001)
        IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE TAU OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.

    METHODS
    -------
    [PRIVATE] __WEIGHT_MATRIX__(POINT, X): [PRIVATE FUNCTION] RETURNS THE WEIGHT MATRIX OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL. IT'S A DIAGONAL MATRIX.
    PREDICT(X): RETURNS THE PREDICTIONS OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL FOR THE INPUT DATA X.
    """
    # INITIALIZES THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL

    def __init__(self, TAU=0.001):
        """INITIALIZES THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.

        PARAMETERS
        ----------
        TAU: FLOAT (DEFAULT: 0.001)
            IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE TAU OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.

        ATTRIBUTES
        ----------
        TAU: FLOAT
            IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE TAU OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.
        """
        # TAU: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE TAU OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.
        self.TAU = TAU

    # WEIGHT_MATRIX(): [PRIVATE FUNCTION] RETURNS THE WEIGHT MATRIX OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL. IT'S A DIAGONAL MATRIX.
    def __WEIGHT_MATRIX__(self, POINT, X):
        """[PRIVATE FUNCTION] RETURNS THE WEIGHT MATRIX OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL. IT'S A DIAGONAL MATRIX.

        PARAMETERS
        ----------
        POINT: NUMPY ARRAY SHAPE (1, NUMBER OF FEATURES)
            IT'S THE POINT THAT WE WANT TO PREDICT ITS LABEL.
        X: NUMPY ARRAY SHAPE (NUMBER OF TRAINING EXAMPLES, NUMBER OF FEATURES)
            IT'S THE TRAINING DATA.

        RETURNS
        -------
        WEIGHT_MATRIX: NUMPY MATRIX SHAPE (NUMBER OF TRAINING EXAMPLES, NUMBER OF TRAINING EXAMPLES)
            IT'S THE WEIGHT MATRIX OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.
        """
        NUMBER_OF_SAMPLES = X.shape[0]  # NUMBER OF TRAINING EXAMPLES
        # INITIALIZES THE WEIGHT MATRIX AS AN IDENTITY MATRIX
        WEIGHT_MATRIX = np.mat(np.eye(NUMBER_OF_SAMPLES))
        # CALCULATES THE WEIGHTS FOR EACH TRAINING EXAMPLE
        for EXAMPLE_INDEX in range(NUMBER_OF_SAMPLES):
            # CALCULATES THE DENOMINATOR OF THE GAUSSIAN KERNEL
            DENOMINATOR = -2 * self.TAU * self.TAU
            WEIGHT_MATRIX[EXAMPLE_INDEX, EXAMPLE_INDEX] = np.exp(np.dot(
                (X[EXAMPLE_INDEX] - POINT), (X[EXAMPLE_INDEX] - POINT).T) / DENOMINATOR)  # CALCULATES THE WEIGHT FOR THE TRAINING EXAMPLE
        return WEIGHT_MATRIX  # RETURNS THE WEIGHT MATRIX

    # PREDICT(): RETURNS THE PREDICTIONS OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL
    def PREDICT(self, X, Y, POINT):
        """RETURNS THE PREDICTIONS OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.

        PARAMETERS
        ----------
        X: NUMPY ARRAY SHAPE (NUMBER OF TRAINING EXAMPLES, NUMBER OF FEATURES)
            IT'S THE TRAINING DATA.
        Y: NUMPY ARRAY SHAPE (NUMBER OF TRAINING EXAMPLES, 1)
            IT'S THE LABELS OF THE TRAINING DATA.
        POINT: NUMPY ARRAY SHAPE (1, NUMBER OF FEATURES)
            IT'S THE POINT THAT WE WANT TO PREDICT ITS LABEL.

        RETURNS
        -------
        THETA: NUMPY ARRAY SHAPE (NUMBER OF FEATURES, 1)
            IT'S THE PARAMETERS OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.
        PREDICTION: FLOAT
            IT'S THE PREDICTION OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL.
        """
        NUMBER_OF_SAMPLES = X.shape[0]  # NUMBER OF TRAINING EXAMPLES
        X_ = np.append(X, np.ones(NUMBER_OF_SAMPLES).reshape(
            NUMBER_OF_SAMPLES, 1), axis=1)  # ADDS A COLUMN OF ONES TO THE X MATRIX
        POINT_ = np.array([POINT, 1])  # ADDS A ONE TO THE POINT
        WEIGHT_MATRIX = self.__WEIGHT_MATRIX__(
            POINT_, X_)  # CALCULATES THE WEIGHT MATRIX
        # CALCULATES THE PARAMETERS, THETA, OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL: THETA IS A VECTOR OF WEIGHTS, THETA_0 IS THE BIAS (SCALAR).
        THETA = np.linalg.pinv(X_.T*(WEIGHT_MATRIX * X_)) * \
            (X_.T*(WEIGHT_MATRIX * Y))
        # CALCULATES THE PREDICTION OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL
        PREDICTION = np.dot(POINT_, THETA)
        # RETURNS THE PARAMETERS AND THE PREDICTION OF THE LOCALLY WEIGHTED LINEAR REGRESSION MODEL
        return THETA, PREDICTION
