import logging
import numpy as np
from autograd import elementwise_grad
from neural_network.base_estimator import BASE_ESTIMATOR
from neural_network.metrics import *
from neural_network.layers import PHASE_MIXIN
from neural_network.batch_iterator import BATCH_ITERATOR

np.random.seed(9999) # THIS LINE OF CODE SETS THE SEED FOR THE RANDOM NUMBER GENERATOR USED IN THE NEURAL NETWORK IMPLEMENTATION. THE NP.RANDOM.SEED() FUNCTION IS PART OF THE NUMPY LIBRARY, WHICH IS COMMONLY USED IN SCIENTIFIC COMPUTING AND DATA ANALYSIS. BY SETTING THE SEED TO A SPECIFIC VALUE (IN THIS CASE, 9999), WE ENSURE THAT THE RANDOM NUMBERS GENERATED BY THE NETWORK WILL BE THE SAME EVERY TIME THE CODE IS RUN. THIS IS IMPORTANT FOR REPRODUCIBILITY, AS IT ALLOWS US TO COMPARE THE RESULTS OF DIFFERENT RUNS OF THE NETWORK AND ENSURE THAT ANY DIFFERENCES ARE DUE TO CHANGES IN THE NETWORK ARCHITECTURE OR TRAINING DATA, RATHER THAN RANDOM VARIATION. IN MACHINE LEARNING, IT IS COMMON TO USE RANDOM INITIALIZATION OF MODEL PARAMETERS TO BREAK SYMMETRY AND PREVENT THE MODEL FROM GETTING STUCK IN LOCAL OPTIMA DURING TRAINING. HOWEVER, THIS RANDOMNESS CAN MAKE IT DIFFICULT TO COMPARE THE PERFORMANCE OF DIFFERENT MODELS OR TO REPRODUCE RESULTS FROM PREVIOUS EXPERIMENTS. BY SETTING THE RANDOM SEED, WE CAN ENSURE THAT THE SAME RANDOM INITIALIZATION IS USED EVERY TIME THE CODE IS RUN, MAKING IT EASIER TO COMPARE RESULTS AND REPRODUCE EXPERIMENTS. OVERALL, SETTING THE RANDOM SEED IS A SIMPLE BUT IMPORTANT STEP IN ENSURING THE REPRODUCIBILITY AND RELIABILITY OF MACHINE LEARNING EXPERIMENTS.

class NEURAL_NETWORK(BASE_ESTIMATOR):
    """NEURAL NETWORK CLASSIFIER.

    PARAMETERS
    ----------
    LAYERS : LIST
        LIST OF LAYERS TO BE USED IN THE NETWORK.
    OPTIMIZER : OBJECT
        OPTIMIZER TO BE USED IN THE NETWORK.
    LOSS : FUNCTION
        LOSS FUNCTION TO BE USED IN THE NETWORK.
    MAX_EPOCHS : INT
        MAXIMUM NUMBER OF EPOCHS TO TRAIN THE NETWORK.
    BATCH_SIZE : INT
        BATCH SIZE TO BE USED IN THE NETWORK.
    METRIC : FUNCTION
        METRIC TO BE USED IN THE NETWORK.
    SHUFFLE : BOOL
        WHETHER TO SHUFFLE THE TRAINING DATA BEFORE EACH EPOCH.
    VERBOSE : BOOL
        WHETHER TO PRINT TRAINING PROGRESS TO STDOUT.
    
    ATTRIBUTES
    ----------
    VERBOSE : BOOL
        WHETHER TO PRINT TRAINING PROGRESS TO STDOUT.
    SHUFFLE : BOOL
        WHETHER TO SHUFFLE THE TRAINING DATA BEFORE EACH EPOCH.
    OPTIMIZER : OBJECT
        OPTIMIZER TO BE USED IN THE NETWORK.
    LOSS : FUNCTION
        LOSS FUNCTION TO BE USED IN THE NETWORK.
    LOSS_GRAD : FUNCTION
        GRADIENT OF THE LOSS FUNCTION.
    METRIC : FUNCTION
        METRIC TO BE USED IN THE NETWORK.
    LAYERS : LIST
        LIST OF LAYERS TO BE USED IN THE NETWORK.
    BATCH_SIZE : INT
        BATCH SIZE TO BE USED IN THE NETWORK.
    MAX_EPOCHS : INT
        MAXIMUM NUMBER OF EPOCHS TO TRAIN THE NETWORK.
    __N_LAYERS__ : INT
        NUMBER OF LAYERS IN THE NETWORK.
    LOG_METRIC : BOOL
        WHETHER TO LOG THE METRIC TO TENSORBOARD.
    METRIC_NAME : STRING
        NAME OF THE METRIC TO BE LOGGED TO TENSORBOARD.
    BPROP_ENTRY : INT
        INDEX OF THE ENTRY LAYER FOR BACK PROPAGATION.
    TRAINING : BOOL
        WHETHER THE NETWORK IS CURRENTLY TRAINING.
    __INITIALIZED__ : BOOL
        WHETHER THE NETWORK HAS BEEN INITIALIZED.
    
    METHODS
    -------
    __SETUP_LAYERS__(X_SHAPE)
        INITIALIZE MODEL'S LAYERS.
    ___FIND_BPROP_ENTRY____()
        FIND ENTRY LAYER FOR BACK PROPAGATION.
    FIT(X, Y)
        FIT THE MODEL TO THE TRAINING DATA.
    _PREDICT__(X)
        MAKE PREDICTIONS ON THE TEST DATA.
    UPDATE(X, Y)
        UPDATE THE MODEL'S PARAMETERS USING GRADIENT DESCENT.
    SCORE(X, Y)
        CALCULATE THE SCORE BETWEEN THE MODEL'S PREDICTIONS AND THE TRUE LABELS.
    __SHUFFLE_DATASET__(X, Y)
        SHUFFLE THE TRAINING DATA.
    RESET()
        RESET THE MODEL'S PARAMETERS.
    __FORWARD__(X)
        PERFORM A __FORWARD__ PASS THROUGH THE NETWORK.
    
    PROPERTIES
    ----------
    N_PARAMS
        TOTAL NUMBER OF PARAMETERS IN THE NETWORK.
    N_LAYERS
        NUMBER OF LAYERS IN THE NETWORK.
    IS_TRAINING
        WHETHER THE NETWORK IS CURRENTLY TRAINING.
    PARAMETERS
        MODEL PARAMETERS.
    PARAMETRIC_LAYERS
        LIST OF PARAMETRIC LAYERS IN THE NETWORK.
    """
    FIT_REQUIRED = False # THIS LINE OF CODE SETS THE FIT_REQUIRED ATTRIBUTE TO FALSE. THIS ATTRIBUTE IS USED BY THE BASE_ESTIMATOR CLASS TO DETERMINE WHETHER THE MODEL NEEDS TO BE FIT TO DATA BEFORE MAKING PREDICTIONS. IN THIS CASE, THE NEURAL NETWORK CLASSIFIER DOES NOT NEED TO BE FIT TO DATA BEFORE MAKING PREDICTIONS, SO WE SET FIT_REQUIRED TO FALSE.

    def __init__(self, LAYERS, OPTIMIZER, LOSS, MAX_EPOCHS=10, BATCH_SIZE=64, METRIC=MEAN_SQUARED_ERROR, SHUFFLE=False, VERBOSE=True):
        """INITIALIZE THE NEURAL NETWORK CLASSIFIER.

        PARAMETERS
        ----------
        LAYERS : LIST
            LIST OF LAYERS TO BE USED IN THE NETWORK.
        OPTIMIZER : OBJECT
            OPTIMIZER TO BE USED IN THE NETWORK.
        LOSS : FUNCTION
            LOSS FUNCTION TO BE USED IN THE NETWORK.
        MAX_EPOCHS : INT
            MAXIMUM NUMBER OF EPOCHS TO TRAIN THE NETWORK.
        BATCH_SIZE : INT
            BATCH SIZE TO BE USED IN THE NETWORK.
        METRIC : FUNCTION
            METRIC TO BE USED IN THE NETWORK.
        SHUFFLE : BOOL
            WHETHER TO SHUFFLE THE TRAINING DATA BEFORE EACH EPOCH.
        VERBOSE : BOOL
            WHETHER TO PRINT TRAINING PROGRESS TO STDOUT.
        
        RETURNS
        -------
        NONE
        """
        self.VERBOSE = VERBOSE # THIS LINE OF CODE SETS THE VERBOSE ATTRIBUTE TO THE VALUE OF THE VERBOSE PARAMETER. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER TRAINING PROGRESS SHOULD BE PRINTED TO STDOUT.
        self.SHUFFLE = SHUFFLE # THIS LINE OF CODE SETS THE SHUFFLE ATTRIBUTE TO THE VALUE OF THE SHUFFLE PARAMETER. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE TRAINING DATA SHOULD BE SHUFFLED BEFORE EACH EPOCH.
        self.OPTIMIZER = OPTIMIZER # THIS LINE OF CODE SETS THE OPTIMIZER ATTRIBUTE TO THE VALUE OF THE OPTIMIZER PARAMETER. THIS ATTRIBUTE IS USED TO OPTIMIZE THE MODEL'S PARAMETERS.
        self.LOSS = LOSS # THIS LINE OF CODE SETS THE LOSS ATTRIBUTE TO THE VALUE OF THE LOSS PARAMETER. THIS ATTRIBUTE IS USED TO CALCULATE THE SCORE BETWEEN THE MODEL'S PREDICTIONS AND THE TRUE LABELS.
        if LOSS == CATEGORICAL_CROSSENTROPY: # THIS LINE OF CODE CHECKS IF THE LOSS FUNCTION IS CATEGORICAL_CROSSENTROPY.
            self.LOSS_GRAD = lambda actual, predicted: -(actual - predicted) # THIS LINE OF CODE SETS THE LOSS_GRAD ATTRIBUTE TO THE GRADIENT OF THE LOSS FUNCTION. THIS ATTRIBUTE IS USED TO CALCULATE THE GRADIENT OF THE LOSS FUNCTION.
        else: # THIS LINE OF CODE RUNS IF THE LOSS FUNCTION IS NOT CATEGORICAL_CROSSENTROPY.
            self.LOSS_GRAD = elementwise_grad(self.LOSS, 1) # THIS LINE OF CODE SETS THE LOSS_GRAD ATTRIBUTE TO THE GRADIENT OF THE LOSS FUNCTION. THIS ATTRIBUTE IS USED TO CALCULATE THE GRADIENT OF THE LOSS FUNCTION.
        self.METRIC = METRIC # THIS LINE OF CODE SETS THE METRIC ATTRIBUTE TO THE VALUE OF THE METRIC PARAMETER. THIS ATTRIBUTE IS USED TO CALCULATE THE MODEL'S PERFORMANCE.
        self.LAYERS = LAYERS # THIS LINE OF CODE SETS THE LAYERS ATTRIBUTE TO THE VALUE OF THE LAYERS PARAMETER. THIS ATTRIBUTE IS USED TO STORE THE MODEL'S LAYERS.
        self.BATCH_SIZE = BATCH_SIZE # THIS LINE OF CODE SETS THE BATCH_SIZE ATTRIBUTE TO THE VALUE OF THE BATCH_SIZE PARAMETER. THIS ATTRIBUTE IS USED TO STORE THE BATCH SIZE TO BE USED IN THE NETWORK.
        self.MAX_EPOCHS = MAX_EPOCHS # THIS LINE OF CODE SETS THE MAX_EPOCHS ATTRIBUTE TO THE VALUE OF THE MAX_EPOCHS PARAMETER. THIS ATTRIBUTE IS USED TO STORE THE MAXIMUM NUMBER OF EPOCHS TO TRAIN THE NETWORK.
        self.__N_LAYERS__ = 0 # THIS LINE OF CODE SETS THE __N_LAYERS__ ATTRIBUTE TO 0. THIS ATTRIBUTE IS USED TO STORE THE NUMBER OF LAYERS IN THE NETWORK.
        self.LOG_METRIC = True if LOSS != METRIC else False # THIS LINE OF CODE SETS THE LOG_METRIC ATTRIBUTE TO TRUE IF THE LOSS FUNCTION IS NOT THE SAME AS THE METRIC. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE METRIC SHOULD BE LOGGED TO TENSORBOARD.
        self.METRIC_NAME = METRIC if LOSS != METRIC else None # THIS LINE OF CODE SETS THE METRIC_NAME ATTRIBUTE TO THE VALUE OF THE METRIC PARAMETER IF THE LOSS FUNCTION IS NOT THE SAME AS THE METRIC. THIS ATTRIBUTE IS USED TO STORE THE NAME OF THE METRIC TO BE LOGGED TO TENSORBOARD.
        self.BPROP_ENTRY = self.__FIND_BPROP_ENTRY__() # THIS LINE OF CODE SETS THE BPROP_ENTRY ATTRIBUTE TO THE VALUE OF THE __FIND_BPROP_ENTRY__ METHOD. THIS ATTRIBUTE IS USED TO STORE THE INDEX OF THE ENTRY LAYER FOR BACK PROPAGATION.
        self.TRAINING = False # THIS LINE OF CODE SETS THE TRAINING ATTRIBUTE TO FALSE. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE NETWORK IS CURRENTLY TRAINING.
        self.__INITIALIZED__ = False # THIS LINE OF CODE SETS THE __INITIALIZED__ ATTRIBUTE TO FALSE. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE MODEL HAS BEEN INITIALIZED.

    def __SETUP_LAYERS__(self, X_SHAPE):
        """SETUP THE LAYERS IN THE NETWORK.

        PARAMETERS
        ----------
        X_SHAPE : LIST
            SHAPE OF THE INPUT DATA.
        
        RETURNS
        -------
        NONE
        """
        X_SHAPE = list(X_SHAPE) # THIS LINE OF CODE CONVERTS THE X_SHAPE PARAMETER TO A LIST.
        X_SHAPE[0] = self.BATCH_SIZE # THIS LINE OF CODE SETS THE FIRST ELEMENT OF THE X_SHAPE PARAMETER TO THE VALUE OF THE BATCH_SIZE PARAMETER.
        for LAYER in self.LAYERS: # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
            LAYER.setup(X_SHAPE) # THIS LINE OF CODE CALLS THE SETUP METHOD OF THE LAYER OBJECT.
            X_SHAPE = LAYER.shape(X_SHAPE) # THIS LINE OF CODE SETS THE X_SHAPE PARAMETER TO THE VALUE OF THE SHAPE METHOD OF THE LAYER OBJECT.
        self.__N_LAYERS__ = len(self.LAYERS) # THIS LINE OF CODE SETS THE __N_LAYERS__ ATTRIBUTE TO THE LENGTH OF THE LAYERS ATTRIBUTE.
        self.OPTIMIZER.SETUP(self) # THIS LINE OF CODE CALLS THE SETUP METHOD OF THE OPTIMIZER OBJECT.
        self.__INITIALIZED__ = True # THIS LINE OF CODE SETS THE __INITIALIZED__ ATTRIBUTE TO TRUE.
        logging.info("TOTAL NUMBER OF PARAMETERS: {}".format(self.N_PARAMS)) # THIS LINE OF CODE PRINTS THE TOTAL NUMBER OF PARAMETERS IN THE NETWORK TO STDOUT.

    def __FIND_BPROP_ENTRY__(self):
        """FIND THE INDEX OF THE ENTRY LAYER FOR BACK PROPAGATION.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        INT
            INDEX OF THE ENTRY LAYER FOR BACK PROPAGATION.
        """
        if len(self.LAYERS) > 0 and not hasattr(self.LAYERS[-1], "PARAMETERS"): # THIS LINE OF CODE CHECKS IF THE LAST LAYER IN THE LAYERS ATTRIBUTE HAS THE PARAMETERS ATTRIBUTE.
            return -1 # THIS LINE OF CODE RETURNS -1.
        return len(self.LAYERS) # THIS LINE OF CODE RETURNS THE LENGTH OF THE LAYERS ATTRIBUTE.

    def FIT(self, X, Y=None):
        """TRAIN THE NETWORK.

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT DATA.
        Y : NUMPY ARRAY
            TARGET DATA.
        
        RETURNS
        -------
        NONE
        """
        assert X is not None, "X CANNOT BE NONE" # THIS LINE OF CODE CHECKS IF THE X PARAMETER IS NOT NONE.
        assert Y is not None, "Y CANNOT BE NONE" # THIS LINE OF CODE CHECKS IF THE Y PARAMETER IS NOT NONE.
        assert len(X) == len(Y), "X AND Y MUST HAVE THE SAME LENGTH" # THIS LINE OF CODE CHECKS IF THE LENGTH OF THE X PARAMETER IS EQUAL TO THE LENGTH OF THE Y PARAMETER.
        if not self.__INITIALIZED__: # THIS LINE OF CODE CHECKS IF THE __INITIALIZED__ ATTRIBUTE IS FALSE.
            self.__SETUP_LAYERS__(X.shape) # THIS LINE OF CODE CALLS THE __SETUP_LAYERS__ METHOD.
        if Y.ndim == 1: # THIS LINE OF CODE CHECKS IF THE Y PARAMETER IS A 1D ARRAY.
            Y = Y[:, np.newaxis] # THIS LINE OF CODE ADDS A NEW AXIS TO THE Y PARAMETER.
        self.__SETUP_INPUT__(X, Y) # THIS LINE OF CODE CALLS THE __SETUP_INPUT__ METHOD.
        self.IS_TRAINING = True # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO TRUE.
        self.OPTIMIZER.optimize(self) # THIS LINE OF CODE CALLS THE OPTIMIZE METHOD OF THE OPTIMIZER OBJECT.
        self.IS_TRAINING = False # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO FALSE.

    def UPDATE(self, X, Y):
        """UPDATE THE PARAMETERS OF THE NETWORK.

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT DATA.
        Y : NUMPY ARRAY
            TARGET DATA.
        
        RETURNS
        -------
        NONE
        """
        Y_PREDICTION = self.__FORWARD__(X) # THIS LINE OF CODE SETS THE Y_PREDICTION VARIABLE TO THE VALUE OF THE __FORWARD__ METHOD.
        GRAD = self.LOSS_GRAD(Y, Y_PREDICTION) # THIS LINE OF CODE SETS THE GRAD VARIABLE TO THE VALUE OF THE LOSS_GRAD METHOD.
        for LAYER in reversed(self.LAYERS[: self.BPROP_ENTRY]): # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
            GRAD = LAYER.BACKWARD_PASS(GRAD) # THIS LINE OF CODE SETS THE GRAD VARIABLE TO THE VALUE OF THE BACKWARD_PASS METHOD OF THE LAYER OBJECT.
        return self.LOSS(Y, Y_PREDICTION) # THIS LINE OF CODE RETURNS THE VALUE OF THE LOSS METHOD.

    def __FORWARD__(self, X):
        """__FORWARD__ PASS THROUGH THE NETWORK.
        
        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT DATA.
        
        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT DATA.
        """
        for LAYER in self.LAYERS: # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
            X = LAYER.FORWARD_PASS(X) # THIS LINE OF CODE SETS THE X VARIABLE TO THE VALUE OF THE FORWARD_PASS METHOD OF THE LAYER OBJECT.
        return X # THIS LINE OF CODE RETURNS THE X VARIABLE.

    def PREDICT(self, X=None):
        """MAKE PREDICTIONS USING THE NETWORK.

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT DATA.
        
        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT DATA.
        """
        assert X is not None, "X CANNOT BE NONE" # THIS LINE OF CODE CHECKS IF THE X PARAMETER IS NOT NONE.
        if not self.__INITIALIZED__: # THIS LINE OF CODE CHECKS IF THE __INITIALIZED__ ATTRIBUTE IS FALSE.
            self.__SETUP_LAYERS__(X.shape) # THIS LINE OF CODE CALLS THE __SETUP_LAYERS__ METHOD.
        Y = [] # THIS LINE OF CODE CREATES AN EMPTY LIST.
        X_BATCH = BATCH_ITERATOR(X, self.BATCH_SIZE) # THIS LINE OF CODE SETS THE X_BATCH VARIABLE TO THE VALUE OF THE BATCH_ITERATOR FUNCTION.
        for XB in X_BATCH: # THIS LINE OF CODE ITERATES THROUGH EACH BATCH IN THE X_BATCH VARIABLE.
            Y.append(self.__FORWARD__(XB)) # THIS LINE OF CODE APPENDS THE VALUE OF THE __FORWARD__ METHOD TO THE Y LIST.
        return np.concatenate(Y) # THIS LINE OF CODE RETURNS THE CONCATENATED VALUE OF THE Y LIST.

    def SCORE(self, X=None, Y=None):
        """CALCULATE THE ERROR OF THE NETWORK.

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT DATA.
        Y : NUMPY ARRAY
            TARGET DATA.

        RETURNS
        -------
        FLOAT
            ERROR OF THE NETWORK.
        """
        TRAINING_PHASE = self.IS_TRAINING # THIS LINE OF CODE SETS THE TRAINING_PHASE VARIABLE TO THE VALUE OF THE IS_TRAINING ATTRIBUTE.
        if TRAINING_PHASE: # THIS LINE OF CODE CHECKS IF THE TRAINING_PHASE VARIABLE IS TRUE.
            self.IS_TRAINING = False # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO FALSE.
        if X is None and Y is None: # THIS LINE OF CODE CHECKS IF THE X AND Y PARAMETERS ARE NONE.
            Y_PREDICTION = self.PREDICT(self.X) # THIS LINE OF CODE SETS THE Y_PREDICTION VARIABLE TO THE VALUE OF THE PREDICT METHOD.
            ERROR = self.METRIC(self.Y, Y_PREDICTION) # THIS LINE OF CODE SETS THE SCORE VARIABLE TO THE VALUE OF THE METRIC METHOD.
        else: # THIS LINE OF CODE RUNS IF THE IF STATEMENT IS FALSE.
            Y_PREDICTION = self.PREDICT(X) # THIS LINE OF CODE SETS THE Y_PREDICTION VARIABLE TO THE VALUE OF THE PREDICT METHOD.
            ERROR = self.METRIC(Y, Y_PREDICTION) # THIS LINE OF CODE SETS THE SCORE VARIABLE TO THE VALUE OF THE METRIC METHOD.
        if TRAINING_PHASE: # THIS LINE OF CODE CHECKS IF THE TRAINING_PHASE VARIABLE IS TRUE.
            self.IS_TRAINING = True # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO TRUE.
        return ERROR # THIS LINE OF CODE RETURNS THE ERROR VARIABLE.
        
    def RESET(self):
        """RESET THE NETWORK.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        NONE
        """
        self.__INITIALIZED__ = False # THIS LINE OF CODE SETS THE __INITIALIZED__ ATTRIBUTE TO FALSE.

    def __SHUFFLE_DATASET__(self):
        """SHUFFLE THE DATASET.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        NONE
        """
        N_SAMPLES = self.X.shape[0] # THIS LINE OF CODE SETS THE N_SAMPLES VARIABLE TO THE VALUE OF THE X ATTRIBUTE.
        INDICES = np.arange(N_SAMPLES) # THIS LINE OF CODE SETS THE INDICES VARIABLE TO THE VALUE OF THE arange METHOD.
        np.random.shuffle(INDICES) # THIS LINE OF CODE SHUFFLES THE INDICES VARIABLE.
        self.X = self.X.take(INDICES, axis=0) # THIS LINE OF CODE SETS THE X ATTRIBUTE TO THE VALUE OF THE take METHOD.
        self.Y = self.Y.take(INDICES, axis=0) # THIS LINE OF CODE SETS THE y ATTRIBUTE TO THE VALUE OF THE take METHOD.

    @property
    def PARAMETRIC_LAYERS(self):
        """GET THE PARAMETRIC LAYERS.
        
        PARAMETERS
        ----------
        NONE
        
        RETURNS
        -------
        GENERATOR
            PARAMETRIC LAYERS.
        """
        for LAYER in self.LAYERS: # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
            if hasattr(LAYER, "PARAMETERS"): # THIS LINE OF CODE CHECKS IF THE LAYER OBJECT HAS THE PARAMETERS ATTRIBUTE.
                yield LAYER # THIS LINE OF CODE RETURNS THE LAYER OBJECT.

    @property
    def PARAMETERS(self):
        """GET THE PARAMETERS OF THE NETWORK.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        LIST
            PARAMETERS OF THE NETWORK.
        """
        PARAMS = [] # THIS LINE OF CODE CREATES AN EMPTY LIST.
        for LAYER in self.PARAMETRIC_LAYERS: # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE PARAMETRIC_LAYERS PROPERTY.
            PARAMS.append(LAYER.PARAMETERS) # THIS LINE OF CODE APPENDS THE VALUE OF THE PARAMETERS ATTRIBUTE TO THE PARAMS LIST.
        return PARAMS # THIS LINE OF CODE RETURNS THE PARAMS LIST.

    @property
    def IS_TRAINING(self):
        """GET THE TRAINING ATTRIBUTE.
        
        PARAMETERS
        ----------
        NONE
        
        RETURNS
        -------
        BOOL
            TRAINING ATTRIBUTE.
        """
        return self.TRAINING # THIS LINE OF CODE RETURNS THE TRAINING ATTRIBUTE.

    @IS_TRAINING.setter
    def IS_TRAINING(self, TRAIN):
        """SET THE TRAINING ATTRIBUTE.

        PARAMETERS
        ----------
        TRAIN : BOOL
            TRAINING ATTRIBUTE.
        
        RETURNS
        -------
        NONE
        """
        self.TRAINING = TRAIN # THIS LINE OF CODE SETS THE TRAINING ATTRIBUTE TO THE VALUE OF THE TRAIN PARAMETER.
        for LAYER in self.LAYERS: # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
            if isinstance(LAYER, PHASE_MIXIN): # THIS LINE OF CODE CHECKS IF THE LAYER OBJECT IS AN INSTANCE OF THE PHASE_MIXIN CLASS.
                LAYER.IS_TRAINING = TRAIN # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO THE VALUE OF THE TRAIN PARAMETER.

    @property
    def N_LAYERS(self):
        """GET THE NUMBER OF LAYERS.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        INT
            NUMBER OF LAYERS.
        """
        return self.__N_LAYERS__ # THIS LINE OF CODE RETURNS THE __N_LAYERS__ ATTRIBUTE.

    @property
    def N_PARAMS(self):
        """GET THE NUMBER OF PARAMETERS.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        INT
            NUMBER OF PARAMETERS.
        """
        return sum([LAYER.PARAMETERS.N_PARAMS for LAYER in self.PARAMETRIC_LAYERS]) # THIS LINE OF CODE RETURNS THE SUM OF THE N_PARAMS ATTRIBUTE OF EACH LAYER IN THE PARAMETRIC_LAYERS PROPERTY.