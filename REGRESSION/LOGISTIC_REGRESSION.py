import numpy as np
from NEURAL_NETWORKS.ACTIVATIONS import SIGMOID
from NEURAL_NETWORKS.METRICS import BINARY_CROSSENTROPY

# LOGISTIC_REGRESSION: CLASS THAT IMPLEMENTS THE LOGISTIC REGRESSION MODEL
class LOGISTIC_REGRESSION:
    # INITIALIZES THE LOGISTIC REGRESSION MODEL
    def __init__(self, LEARNING_RATE=0.001, BATCH_SIZE=32, EPOCHS=1000):
        self.LEARNING_RATE = LEARNING_RATE # LEARNING RATE: HYPERPARAMETER THAT CONTROLS THE STEP SIZE AT EACH ITERATION WHILE MOVING TOWARDS A MINIMUM OF A LOSS FUNCTION.
        self.BATCH_SIZE = BATCH_SIZE # BATCH SIZE: HYPERPARAMETER THAT CONTROLS THE NUMBER OF SAMPLES TO BE USED IN EACH ITERATION.
        self.EPOCHS = EPOCHS # NUMBER OF ITERATIONS: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
        self.WEIGHTS = None # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE POLYNOMIAL REGRESSION MODEL.
        self.BIAS = None # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE POLYNOMIAL REGRESSION MODEL.
    
    # FIT(): TRAINS THE POLYNOMIAL REGRESSION MODEL
    def FIT(self, X, Y):
        _X = self.__NORMALIZE__(X) # NORMALIZE THE INPUTS
        NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES = X.shape # NUMBER OF TRAINING EXAMPLES AND NUMBER OF FEATURES
        self.WEIGHTS = np.zeros((NUMBER_OF_FEATURES, 1)) # INITIALIZE WEIGHTS TO ZERO
        self.BIAS = 0 # INITIALIZE BIAS TO ZERO
        Y = Y.reshape(NUMBER_OF_SAMPLES, 1) # RESHAPES Y TO (NUMBER_OF_SAMPLES, 1)
        LOSS = [] # LIST TO STORE LOSS
        for _ in range(self.EPOCHS): # FOR EACH EPOCH
            for EXAMPLE_INDEX in range((NUMBER_OF_SAMPLES - 1) // self.BATCH_SIZE + 1): # FOR EACH BATCH
                START_INDEX = EXAMPLE_INDEX * self.BATCH_SIZE # START INDEX OF THE BATCH
                END_INDEX = START_INDEX + self.BATCH_SIZE # END INDEX OF THE BATCH
                X_BATCH = _X[START_INDEX:END_INDEX] # X_BATCH: BATCH OF INPUTS
                Y_BATCH = Y[START_INDEX:END_INDEX] # Y_BATCH: BATCH OF TARGET VALUES
                HYPOTHESIS = SIGMOID(np.dot(X_BATCH, self.WEIGHTS) + self.BIAS) # HYPOTHESIS: PREDICTED TARGET VALUES
                DERIVATIVE_WEIGHTS, DERIVATIVE_BIAS = self.__GRADIENTS__(X_BATCH, Y_BATCH, HYPOTHESIS) # DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE PARAMETERS: WEIGHTS AND BIAS
                self.WEIGHTS -= self.LEARNING_RATE * DERIVATIVE_WEIGHTS # UPDATES THE WEIGHTS
                self.BIAS -= self.LEARNING_RATE * DERIVATIVE_BIAS # UPDATES THE BIAS
            LOSS.append(self.__LOSS_FUNCTION__(Y, SIGMOID(np.dot(_X, self.WEIGHTS) + self.BIAS))) # CALCULATES THE LOSS AND APPENDS IT TO THE LIST
    
    # TRANSFORM(): PREDICTS THE TARGET VALUE OF THE INPUTS
    def TRANSFORM(self, X):
        _X = self.__NORMALIZE__(X) # NORMALIZE THE INPUTS
        HYPOTHESIS = SIGMOID(np.dot(_X, self.WEIGHTS) + self.BIAS) # HYPOTHESIS: PREDICTED TARGET VALUES
        HYPOTHESIS_CLASS = [] # LIST TO STORE THE PREDICTED TARGET VALUES
        HYPOTHESIS_CLASS = [1 if i > 0.5 else 0 for i in HYPOTHESIS] # FOR EACH PREDICTED TARGET VALUE APPENDS 1 IF IT'S GREATER THAN 0.5, ELSE APPENDS 0
        return np.array(HYPOTHESIS_CLASS) # RETURNS THE PREDICTED TARGET VALUES
    
    # LOSS FUNCTION [PRIVATE & STATIC]: CALCULATES THE LOSS FUNCTION (IT'S THE FUNCTION THAT WE WANT TO MINIMIZE): BINARY CROSS ENTROPY
    @staticmethod
    def __LOSS_FUNCTION__(TRUE_TARGET_VALUE, HYPOTHESIS):
        return BINARY_CROSSENTROPY(TRUE_TARGET_VALUE, HYPOTHESIS) # RETURNS THE LOSS FUNCTION
    
    # GRADIENTS [PRIVATE]: CALCULATES THE GRADIENTS OF THE LOSS FUNCTION W.R.T. THE PARAMETERS: WEIGHTS AND BIAS
    def __GRADIENTS__(self, X, Y, HYPOTHESIS):
        NUMBER_OF_SAMPLES = X.shape[0] # NUMBER OF TRAINING EXAMPLES
        DERIVATIVE_WEIGHTS = (1 / NUMBER_OF_SAMPLES) * np.dot(X.T, (HYPOTHESIS - Y)) # DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE WEIGHTS
        DERIVATIVE_BIAS = (1 / NUMBER_OF_SAMPLES) * np.sum(HYPOTHESIS - Y) # DERIVATIVE OF THE LOSS FUNCTION W.R.T. THE BIAS
        return DERIVATIVE_WEIGHTS, DERIVATIVE_BIAS # RETURNS THE GRADIENTS OF THE LOSS FUNCTION W.R.T. THE PARAMETERS: WEIGHTS AND BIAS
    
    # NORMALIZE [PRIVATE]: NORMALIZES THE INPUTS (FEATURES) OF THE MODEL
    def __NORMALIZE__(self, X):
        _, NUMBER_OF_FEATURES = X.shape # NUMBER OF TRAINING EXAMPLES AND NUMBER OF FEATURES
        for _ in range(NUMBER_OF_FEATURES): # FOR EACH FEATURE
            X = (X - X.mean(axis=0)) / X.std(axis=0) # NORMALIZE THE FEATURE
        return X # RETURNS THE NORMALIZED INPUTS