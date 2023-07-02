import numpy as np
from NEURAL_NETWORKS.ACTIVATIONS import SOFTMAX
from NEURAL_NETWORKS.METRICS import ACCURACY

# SOFTMAX_REGRESSION: CLASS THAT IMPLEMENTS SOFTMAX REGRESSION MODEL
class SOFTMAX_REGRESSION:
    # INITIALIZES THE SOFTMAX REGRESSION MODEL
    def __init__(self, LEARNING_RATE=0.01, C=2, EPOCHS=1000):
        self.LEARNING_RATE = LEARNING_RATE # LEARNING RATE: HYPERPARAMETER THAT CONTROLS THE STEP SIZE AT EACH ITERATION WHILE MOVING TOWARDS A MINIMUM OF A LOSS FUNCTION.
        self.C = C # NUMBER OF CLASSES: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF CLASSES IN THE DATASET.
        self.EPOCHS = EPOCHS # NUMBER OF ITERATIONS: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF ITERATIONS THAT THE ALGORITHM PASS THROUGH THE TRAINING DATA.
        self.WEIGHTS = None # WEIGHTS: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE POLYNOMIAL REGRESSION MODEL.
        self.BIAS = None # BIAS: IT'S THE PARAMETER THAT CORRESPONDS TO THE BIAS OF THE POLYNOMIAL REGRESSION MODEL.
    
    # ONE_HOT: CONVERTS THE LABELS TO ONE HOT ENCODING FORMAT
    def ONE_HOT(self, Y):
        # Y: LABELS/GROUND TRUTH
        # C: NUMBER OF CLASSES
        Y_HOT = np.zeros((len(Y), self.C)) # INITIALIZING THE ONE HOT ENCODING MATRIX
        Y_HOT[np.arange(len(Y)), Y] = 1 # CONVERTING THE LABELS TO ONE HOT ENCODING FORMAT
        return Y_HOT # RETURNING THE ONE HOT ENCODING MATRIX
    
    # FIT(): TRAINS THE SOFTMAX REGRESSION MODEL
    def FIT(self, X, Y):
        NUMBER_OF_TRAINING_EXAMPLES, NUMBER_OF_FEATURES = X.shape # NUMBER OF TRAINING EXAMPLES AND NUMBER OF FEATURES
        self.WEIGHTS = np.random.random((NUMBER_OF_FEATURES, self.C)) # INITIALIZING THE WEIGHTS
        self.BIAS = np.random.random(self.C) # INITIALIZING THE BIAS
        LOSSES = [] # LIST TO STORE THE LOSS AT EACH EPOCH
        for EPOCH in range(self.EPOCHS): # LOOPING THROUGH THE EPOCHS
            Z = X@self.WEIGHTS + self.BIAS # CALCULATING THE HYPOTHESIS/PREDICTION
            HYPOTHESIS = SOFTMAX(Z) # CALCULATING THE SOFTMAX OF THE HYPOTHESIS/PREDICTION
            Y_HOT = self.ONE_HOT(Y) # CONVERTING THE LABELS TO ONE HOT ENCODING FORMAT
            W_GRADIENT = (1 / NUMBER_OF_TRAINING_EXAMPLES) * np.dot(X.T, (HYPOTHESIS - Y_HOT)) # CALCULATING THE GRADIENT OF THE LOSS W.R.T WEIGHTS
            B_GRADIENT = (1 / NUMBER_OF_TRAINING_EXAMPLES) * np.sum(HYPOTHESIS - Y_HOT) # CALCULATING THE GRADIENT OF THE LOSS W.R.T BIAS
            self.WEIGHTS -= self.LEARNING_RATE * W_GRADIENT # UPDATING THE WEIGHTS
            self.BIAS -= self.LEARNING_RATE * B_GRADIENT # UPDATING THE BIAS
            LOSS = -np.mean(np.log(HYPOTHESIS[np.arange(len(Y)), Y])) # CALCULATING THE LOSS
            LOSSES.append(LOSS) # APPENDING THE LOSS TO THE LIST
        
    # PREDICT(): PREDICTS THE LABELS FOR THE GIVEN DATA    
    def PREDICT(self, X):
        Z = X@self.WEIGHTS + self.BIAS # CALCULATING THE HYPOTHESIS/PREDICTION
        HYPOTHESIS = SOFTMAX(Z) # CALCULATING THE SOFTMAX OF THE HYPOTHESIS/PREDICTION
        return np.argmax(HYPOTHESIS, axis=1) # RETURNING THE PREDICTED LABELS
    
    # ACCURACY [STATIC]: CALCULATES THE ACCURACY OF THE MODEL
    @staticmethod
    def ACCURACY(Y, HYPOTHESIS):
        return ACCURACY(Y, HYPOTHESIS) # RETURNS THE ACCURACY OF THE MODEL