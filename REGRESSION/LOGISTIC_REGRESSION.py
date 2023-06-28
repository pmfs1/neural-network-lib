import numpy as np

# LOGISTIC_REGRESSION: IMPLEMENTS THE LOGISTIC REGRESSION MODEL. WHICH IS A CLASSIFICATION ALGORITHM USED TO ASSIGN OBSERVATIONS TO A DISCRETE SET OF CLASSES.


class LOGISTIC_REGRESSION:
    """IMPLEMENTS THE LOGISTIC REGRESSION MODEL. WHICH IS A CLASSIFICATION ALGORITHM USED TO ASSIGN OBSERVATIONS TO A DISCRETE SET OF CLASSES.

    ATTRIBUTES
    ----------
    LEARNING_RATE: FLOAT (DEFAULT=0.001)
        IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE LEARNING RATE OF THE ALGORITHM.
    BATCH_SIZE: INT (DEFAULT=32)
        IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF SAMPLES TO BE USED IN EACH ITERATION.
    EPOCHS: INT (DEFAULT=1000)
        IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
    WEIGHTS: NUMPY ARRAY
        IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE POLYNOMIAL REGRESSION MODEL.
    BIAS: FLOAT
        IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE POLYNOMIAL REGRESSION MODEL.

    METHODS
    -------
    FIT(X, Y)
        FITS THE LOGISTIC REGRESSION MODEL TO THE TRAINING DATA.
    PREDICT(X)
        PREDICTS THE OUTPUT OF THE GIVEN INPUT DATA.
    __LOSS_FUNCTION__(Y, Y_PRED)
        RETURNS THE LOSS VALUE OF THE GIVEN PREDICTIONS.
    __GRADIENTS__(X, Y, Y_PRED)
        RETURNS THE GRADIENTS OF THE WEIGHTS AND THE BIAS.
    __NORMALIZE__(X)
        NORMALIZES THE GIVEN INPUT DATA.
    __SIGMOID__(Z)
        RETURNS THE SIGMOID OF THE GIVEN INPUT.
    """
    # INITIALIZES THE LOGISTIC REGRESSION MODEL

    def __init__(self, LEARNING_RATE=0.001, BATCH_SIZE=32, EPOCHS=1000):
        """INITIALIZES THE LOGISTIC REGRESSION MODEL.

        PARAMETERS
        ----------
        LEARNING_RATE: FLOAT (DEFAULT=0.001)
            HYPERPARAMETER THAT CONTROLS THE STEP SIZE AT EACH ITERATION WHILE MOVING TOWARDS A MINIMUM OF A LOSS FUNCTION.
        BATCH_SIZE: INT (DEFAULT=32)
            HYPERPARAMETER THAT CONTROLS THE NUMBER OF SAMPLES TO BE USED IN EACH ITERATION.
        EPOCHS: INT (DEFAULT=1000)
            IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
        """
        # LEARNING RATE: HYPERPARAMETER THAT CONTROLS THE STEP SIZE AT EACH ITERATION WHILE MOVING TOWARDS A MINIMUM OF A LOSS FUNCTION.
        self.LEARNING_RATE = LEARNING_RATE
        # BATCH SIZE: HYPERPARAMETER THAT CONTROLS THE NUMBER OF SAMPLES TO BE USED IN EACH ITERATION.
        self.BATCH_SIZE = BATCH_SIZE
        # NUMBER OF ITERATIONS: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
        self.EPOCHS = EPOCHS
        # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE POLYNOMIAL REGRESSION MODEL.
        self.WEIGHTS = None
        # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE POLYNOMIAL REGRESSION MODEL.
        self.BIAS = None

    # FIT(): TRAINS THE POLYNOMIAL REGRESSION MODEL
    def FIT(self, X, Y):
        """TRAIN THE POLYNOMIAL REGRESSION MODEL.

        PARAMETERS
        ----------
        X: NUMPY ARRAY
            IT'S THE INPUT DATA.
        Y: NUMPY ARRAY
            IT'S THE TARGET VALUES.

        RETURNS
        -------
        NONE
        """
        _X = self.__NORMALIZE__(X)  # NORMALIZE THE INPUTS
        # NUMBER OF TRAINING EXAMPLES AND NUMBER OF FEATURES
        NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES = X.shape
        # INITIALIZE WEIGHTS TO ZERO
        self.WEIGHTS = np.zeros((NUMBER_OF_FEATURES, 1))
        self.BIAS = 0  # INITIALIZE BIAS TO ZERO
        # RESHAPES Y TO (NUMBER_OF_SAMPLES, 1)
        Y = Y.reshape(NUMBER_OF_SAMPLES, 1)
        LOSS = []  # LIST TO STORE LOSS
        for _ in range(self.EPOCHS):  # FOR EACH EPOCH
            # FOR EACH BATCH
            for EXAMPLE_INDEX in range((NUMBER_OF_SAMPLES - 1) // self.BATCH_SIZE + 1):
                START_INDEX = EXAMPLE_INDEX * self.BATCH_SIZE  # START INDEX OF THE BATCH
                END_INDEX = START_INDEX + self.BATCH_SIZE  # END INDEX OF THE BATCH
                X_BATCH = _X[START_INDEX:END_INDEX]  # X_BATCH: BATCH OF INPUTS
                # Y_BATCH: BATCH OF TARGET VALUES
                Y_BATCH = Y[START_INDEX:END_INDEX]
                # PREDICTION: PREDICTED TARGET VALUES
                PREDICTION = LOGISTIC_REGRESSION.__SIGMOID__(
                    np.dot(X_BATCH, self.WEIGHTS) + self.BIAS)
                # DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE PARAMETERS: WEIGHTS AND BIAS
                DERIVATIVE_WEIGHTS, DERIVATIVE_BIAS = self.__GRADIENTS__(
                    X_BATCH, Y_BATCH, PREDICTION)
                self.WEIGHTS -= self.LEARNING_RATE * DERIVATIVE_WEIGHTS  # UPDATES THE WEIGHTS
                self.BIAS -= self.LEARNING_RATE * DERIVATIVE_BIAS  # UPDATES THE BIAS
            # CALCULATES THE LOSS AND APPENDS IT TO THE LIST
            LOSS.append(self.__LOSS_FUNCTION__(
                Y, LOGISTIC_REGRESSION.__SIGMOID__(np.dot(_X, self.WEIGHTS) + self.BIAS)))

    # PREDICT(): PREDICTS THE TARGET VALUE OF THE INPUTS
    def PREDICT(self, X):
        """PREDICTS THE TARGET VALUE OF THE INPUTS.

        PARAMETERS
        ----------
        X: NUMPY ARRAY
            IT'S THE INPUT DATA.

        RETURNS
        -------
        PREDICTION: NUMPY ARRAY
            IT'S THE PREDICTED TARGET VALUES.

        EXCEPTIONS
        ----------
        ASSERTION ERROR: IF THE MODEL IS NOT TRAINED.
        """
        assert self.WEIGHTS is not None and self.BIAS is not None, "RUN FIT() FIRST"  # ASSERTION TO CHECK IF THE MODEL IS TRAINED
        _X = self.__NORMALIZE__(X)  # NORMALIZE THE INPUTS
        # PREDICTION: PREDICTED TARGET VALUES
        PREDICTION = LOGISTIC_REGRESSION.__SIGMOID__(
            np.dot(_X, self.WEIGHTS) + self.BIAS)
        PREDICTION_CLASS = []  # LIST TO STORE THE PREDICTED TARGET VALUES
        # FOR EACH PREDICTED TARGET VALUE APPENDS 1 IF IT'S GREATER THAN 0.5, ELSE APPENDS 0
        PREDICTION_CLASS = [1 if i > 0.5 else 0 for i in PREDICTION]
        # RETURNS THE PREDICTED TARGET VALUES
        return np.array(PREDICTION_CLASS)

    # LOSS FUNCTION [PRIVATE & STATIC]: CALCULATES THE LOSS FUNCTION (IT'S THE FUNCTION THAT WE WANT TO MINIMIZE): BINARY CROSS ENTROPY
    @staticmethod
    def __LOSS_FUNCTION__(TRUE_TARGET_VALUE, PREDICTION):
        """[PRIVATE & STATIC] CALCULATES THE LOSS FUNCTION (IT'S THE FUNCTION THAT WE WANT TO MINIMIZE): BINARY CROSS ENTROPY.

        PARAMETERS
        ----------
        TRUE_TARGET_VALUE: NUMPY ARRAY
            IT'S THE TRUE TARGET VALUES.
        PREDICTION: NUMPY ARRAY
            IT'S THE PREDICTED TARGET VALUES.

        RETURNS
        -------
        LOSS: FLOAT
            IT'S THE LOSS VALUE.
        """
        PREDICTION = np.clip(PREDICTION, 1e-15, 1 -
                             1e-15)  # CLIPS THE PREDICTED VALUES
        # RETURN BINARY CROSS ENTROPY: -MEAN(TRUE_TARGET_VALUE * LOG(PREDICTION) + (1 - TRUE_TARGET_VALUE) * LOG(1 - PREDICTION))
        return np.mean(-np.sum(TRUE_TARGET_VALUE * np.log(PREDICTION) + (1 - TRUE_TARGET_VALUE) * np.log(1 - PREDICTION)))

    # GRADIENTS [PRIVATE FUNCTION]: CALCULATES THE GRADIENTS OF THE LOSS FUNCTION W.R.T. THE PARAMETERS: WEIGHTS AND BIAS
    def __GRADIENTS__(self, X, Y, PREDICTION):
        """[PRIVATE FUNCTION] CALCULATES THE GRADIENTS OF THE LOSS FUNCTION W.R.T. THE PARAMETERS: WEIGHTS AND BIAS.

        PARAMETERS
        ----------
        X: NUMPY ARRAY
            IT'S THE INPUT DATA.
        Y: NUMPY ARRAY
            IT'S THE TARGET VALUES.
        PREDICTION: NUMPY ARRAY
            IT'S THE PREDICTED TARGET VALUES.

        RETURNS
        -------
        DERIVATIVE_WEIGHTS: NUMPY ARRAY
            IT'S THE DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE WEIGHTS.
        DERIVATIVE_BIAS: FLOAT
            IT'S THE DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE BIAS.
        """
        NUMBER_OF_SAMPLES = X.shape[0]  # NUMBER OF TRAINING EXAMPLES
        # DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE WEIGHTS
        DERIVATIVE_WEIGHTS = (1 / NUMBER_OF_SAMPLES) * \
            np.dot(X.T, (PREDICTION - Y))
        # DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE BIAS
        DERIVATIVE_BIAS = (1 / NUMBER_OF_SAMPLES) * np.sum(PREDICTION - Y)
        # RETURNS THE GRADIENTS OF THE LOSS FUNCTION W.R.T. THE PARAMETERS: WEIGHTS AND BIAS
        return DERIVATIVE_WEIGHTS, DERIVATIVE_BIAS

    # NORMALIZE [PRIVATE FUNCTION]: NORMALIZES THE INPUTS (FEATURES) OF THE MODEL
    def __NORMALIZE__(self, X):
        """[PRIVATE FUNCTION] NORMALIZES THE INPUTS (FEATURES) OF THE MODEL.

        PARAMETERS
        ----------
        X: NUMPY ARRAY
            IT'S THE INPUT DATA.

        RETURNS
        -------
        X: NUMPY ARRAY
            IT'S THE NORMALIZED INPUTS.
        """
        # NUMBER OF TRAINING EXAMPLES AND NUMBER OF FEATURES
        _, NUMBER_OF_FEATURES = X.shape
        for _ in range(NUMBER_OF_FEATURES):  # FOR EACH FEATURE
            X = (X - X.mean(axis=0)) / X.std(axis=0)  # NORMALIZE THE FEATURE
        return X  # RETURNS THE NORMALIZED INPUTS

    # SIGMOID [PRIVATE & STATIC]: CALCULATES THE SIGMOID ACTIVATION
    @staticmethod
    def __SIGMOID__(Z):
        """[PRIVATE & STATIC] SIGMOID ACTIVATION FUNCTION

        PARAMETERS
        ----------
        Z : ARRAY
            LINEAR TRANSFORMATION

        RETURNS
        -------
        RETURN SIGMOID ACTIVATION
        """
        return 1.0 / (1.0 + np.exp(-Z))  # RETURN SIGMOID ACTIVATION: 1 / (1 + e^(-Z))
