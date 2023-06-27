import numpy as np

# LINEAR_REGRESSION: IMPLEMENTS THE LINEAR REGRESSION MODEL. WHICH IS A STATISTICAL METHOD FOR FINDING A LINE THAT BEST FITS A DATA SET. IT IS USED TO PREDICT OR VISUALIZE A RELATIONSHIP BETWEEN TWO VARIABLES. ONE VARIABLE IS CALLED THE DEPENDENT VARIABLE, WHICH IS THE OUTCOME OR RESPONSE THAT IS MEASURED. THE OTHER VARIABLE IS CALLED THE INDEPENDENT VARIABLE, WHICH IS THE FACTOR OR PREDICTOR THAT INFLUENCES THE OUTCOME.


class LINEAR_REGRESSION:
    """IMPLEMENTS THE LINEAR REGRESSION MODEL. WHICH IS A STATISTICAL METHOD FOR FINDING A LINE THAT BEST FITS A DATA SET. IT IS USED TO PREDICT OR VISUALIZE A RELATIONSHIP BETWEEN TWO VARIABLES. ONE VARIABLE IS CALLED THE DEPENDENT VARIABLE, WHICH IS THE OUTCOME OR RESPONSE THAT IS MEASURED. THE OTHER VARIABLE IS CALLED THE INDEPENDENT VARIABLE, WHICH IS THE FACTOR OR PREDICTOR THAT INFLUENCES THE OUTCOME.

    ATTRIBUTES
    ----------
    LEARNING_RATE: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE LEARNING RATE OF THE ALGORITHM.
    EPOCHS: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
    WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
    BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE LINEAR REGRESSION MODEL.

    METHODS
    -------
    FIT(X, Y)
        FITS THE LINEAR REGRESSION MODEL TO THE TRAINING DATA.
    PREDICT(X)
        PREDICTS THE OUTPUT OF THE GIVEN INPUT DATA.
    """
    # INITIALIZES THE LINEAR REGRESSION MODEL

    def __init__(self, LEARNING_RATE=0.001, EPOCHS=1000):
        """INITIALIZES THE LINEAR REGRESSION MODEL.

        PARAMETERS
        ----------
        LEARNING_RATE: FLOAT, DEFAULT=0.001
            IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE LEARNING RATE OF THE ALGORITHM.
        EPOCHS: INT, DEFAULT=1000
            IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.

        ATTRIBUTES
        ----------
        LEARNING_RATE: FLOAT
            IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE LEARNING RATE OF THE ALGORITHM.
        EPOCHS: INT
            IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
        WEIGHTS: NUMPY ARRAY SHAPE (N_FEATURES, 1)
            IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        BIAS: FLOAT
            IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE LINEAR REGRESSION MODEL.
        """
        # LEARNING RATE: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE LEARNING RATE OF THE ALGORITHM.
        self.LEARNING_RATE = LEARNING_RATE
        # NUMBER OF ITERATIONS: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
        self.EPOCHS = EPOCHS
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        self.WEIGHTS = None
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE LINEAR REGRESSION MODEL.
        self.BIAS = None

    # FIT(): FITS THE LINEAR REGRESSION MODEL TO THE TRAINING DATA
    def FIT(self, X, Y):
        """FITS THE LINEAR REGRESSION MODEL TO THE TRAINING DATA.

        PARAMETERS
        ----------
        X: NUMPY ARRAY SHAPE (N_SAMPLES, N_FEATURES)
            IT'S THE INPUT DATA.
        Y: NUMPY ARRAY SHAPE (N_SAMPLES, 1)
            IT'S THE TARGET DATA.

        RETURNS
        -------
        NONE
        """
        # INITIALIZES THE WEIGHTS AND THE BIAS
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE LINEAR REGRESSION MODEL.
        self.WEIGHTS = np.zeros(X.shape[1])
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE LINEAR REGRESSION MODEL.
        self.BIAS = 0
        # IMPLEMENTS THE GRADIENT DESCENT ALGORITHM
        for _ in range(self.EPOCHS):  # FOR EACH ITERATION
            # CALCULATES THE PREDICTIONS OF THE LINEAR REGRESSION MODEL
            PREDICTIONS = self.PREDICT(X)
            # CALCULATES THE DERIVATIVES OF THE WEIGHTS AND THE BIAS
            # WEIGHTS_DERIVATIVE: IT'S THE DERIVATIVE OF THE WEIGHTS.
            WEIGHTS_DERIVATIVE = - \
                (2/X.shape[0]) * np.dot(X.T, (Y - PREDICTIONS))
            # BIAS_DERIVATIVE: IT'S THE DERIVATIVE OF THE BIAS.
            BIAS_DERIVATIVE = -(2/X.shape[0]) * np.sum(Y - PREDICTIONS)
            # UPDATES THE WEIGHTS AND THE BIAS
            self.WEIGHTS -= self.LEARNING_RATE * WEIGHTS_DERIVATIVE
            self.BIAS -= self.LEARNING_RATE * BIAS_DERIVATIVE

    # PREDICT(): RETURNS THE PREDICTIONS OF THE LINEAR REGRESSION MODEL
    def PREDICT(self, X):
        """RETURNS THE PREDICTIONS OF THE LINEAR REGRESSION MODEL.

        PARAMETERS
        ----------
        X: NUMPY ARRAY SHAPE (N_SAMPLES, N_FEATURES)
            IT'S THE INPUT DATA.

        RETURNS
        -------
        PREDICTIONS: NUMPY ARRAY SHAPE (N_SAMPLES, 1)
            IT'S THE PREDICTIONS OF THE LINEAR REGRESSION MODEL.

        EXCEPTIONS
        ----------
        ASSERTION ERROR: IF FIT() IS NOT RUN BEFORE
        """
        assert self.WEIGHTS is not None and self.BIAS is not None, "RUN FIT() FIRST"  # ASSERTS THAT FIT() IS RUN BEFORE
        # RETURNS THE PREDICTIONS OF THE LINEAR REGRESSION MODEL
        return np.dot(X, self.WEIGHTS) + self.BIAS
