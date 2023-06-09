import secrets
import numpy as np
from autograd import elementwise_grad

from .BASE_ESTIMATOR import BASE_ESTIMATOR
from .BATCH_ITERATOR import BATCH_ITERATOR
from .LAYERS.BASIC import PHASE_MIXIN
from .METRICS import MEAN_SQUARED_ERROR


class NEURAL_NETWORK(BASE_ESTIMATOR):
    """NEURAL NETWORK

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

    ATTRIBUTES
    ----------
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
    FIT_REQUIRED = False  # THIS LINE OF CODE SETS THE FIT_REQUIRED ATTRIBUTE TO FALSE. THIS ATTRIBUTE IS USED BY THE BASE_ESTIMATOR CLASS TO DETERMINE WHETHER THE MODEL NEEDS TO BE FIT TO DATA BEFORE MAKING PREDICTIONS. IN THIS CASE, THE NEURAL NETWORK CLASSIFIER DOES NOT NEED TO BE FIT TO DATA BEFORE MAKING PREDICTIONS, SO WE SET FIT_REQUIRED TO FALSE.

    def __init__(self, LAYERS, OPTIMIZER, LOSS, MAX_EPOCHS=10, BATCH_SIZE=64, METRIC=MEAN_SQUARED_ERROR, SHUFFLE=False):
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

        RETURNS
        -------
        NONE
        """
        # THIS LINE OF CODE CALLS THE __INIT__ METHOD OF THE BASE_ESTIMATOR CLASS.
        super().__init__()
        # THIS LINE OF CODE SETS THE SHUFFLE ATTRIBUTE TO THE VALUE OF THE SHUFFLE PARAMETER. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE TRAINING DATA SHOULD BE SHUFFLED BEFORE EACH EPOCH.
        self.SHUFFLE = SHUFFLE
        # THIS LINE OF CODE SETS THE OPTIMIZER ATTRIBUTE TO THE VALUE OF THE OPTIMIZER PARAMETER. THIS ATTRIBUTE IS USED TO OPTIMIZE THE MODEL'S PARAMETERS.
        self.OPTIMIZER = OPTIMIZER
        # THIS LINE OF CODE SETS THE LOSS ATTRIBUTE TO THE VALUE OF THE LOSS PARAMETER. THIS ATTRIBUTE IS USED TO CALCULATE THE SCORE BETWEEN THE MODEL'S PREDICTIONS AND THE TRUE LABELS.
        self.LOSS = LOSS
        # THIS LINE OF CODE CHECKS IF THE LOSS FUNCTION IS CATEGORICAL_CROSSENTROPY.
        # if isinstance(LOSS, CATEGORICAL_CROSSENTROPY):  # THIS LINE OF CODE RUNS IF THE LOSS FUNCTION IS CATEGORICAL_CROSSENTROPY.
        #    # THIS LINE OF CODE SETS THE LOSS_GRAD ATTRIBUTE TO THE GRADIENT OF THE LOSS FUNCTION. THIS ATTRIBUTE IS USED TO CALCULATE THE GRADIENT OF THE LOSS FUNCTION.
        #    self.LOSS_GRAD = lambda actual, predicted: -(actual - predicted)
        # else:  # THIS LINE OF CODE RUNS IF THE LOSS FUNCTION IS NOT CATEGORICAL_CROSSENTROPY.
        # THIS LINE OF CODE SETS THE LOSS_GRAD ATTRIBUTE TO THE GRADIENT OF THE LOSS FUNCTION. THIS ATTRIBUTE IS USED TO CALCULATE THE GRADIENT OF THE LOSS FUNCTION.
        self.LOSS_GRAD = elementwise_grad(self.LOSS, 1)
        # THIS LINE OF CODE SETS THE METRIC ATTRIBUTE TO THE VALUE OF THE METRIC PARAMETER. THIS ATTRIBUTE IS USED TO CALCULATE THE MODEL'S PERFORMANCE.
        self.METRIC = METRIC
        # THIS LINE OF CODE SETS THE LAYERS ATTRIBUTE TO THE VALUE OF THE LAYERS PARAMETER. THIS ATTRIBUTE IS USED TO STORE THE MODEL'S LAYERS.
        self.LAYERS = LAYERS
        # THIS LINE OF CODE SETS THE BATCH_SIZE ATTRIBUTE TO THE VALUE OF THE BATCH_SIZE PARAMETER. THIS ATTRIBUTE IS USED TO STORE THE BATCH SIZE TO BE USED IN THE NETWORK.
        self.BATCH_SIZE = BATCH_SIZE
        # THIS LINE OF CODE SETS THE MAX_EPOCHS ATTRIBUTE TO THE VALUE OF THE MAX_EPOCHS PARAMETER. THIS ATTRIBUTE IS USED TO STORE THE MAXIMUM NUMBER OF EPOCHS TO TRAIN THE NETWORK.
        self.MAX_EPOCHS = MAX_EPOCHS
        # THIS LINE OF CODE SETS THE __N_LAYERS__ ATTRIBUTE TO 0. THIS ATTRIBUTE IS USED TO STORE THE NUMBER OF LAYERS IN THE NETWORK.
        self.__N_LAYERS__ = 0
        # THIS LINE OF CODE SETS THE LOG_METRIC ATTRIBUTE TO TRUE IF THE LOSS FUNCTION IS NOT THE SAME AS THE METRIC. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE METRIC SHOULD BE LOGGED TO TENSORBOARD.
        self.LOG_METRIC = True if LOSS != METRIC else False
        # THIS LINE OF CODE SETS THE METRIC_NAME ATTRIBUTE TO THE VALUE OF THE METRIC PARAMETER IF THE LOSS FUNCTION IS NOT THE SAME AS THE METRIC. THIS ATTRIBUTE IS USED TO STORE THE NAME OF THE METRIC TO BE LOGGED TO TENSORBOARD.
        self.METRIC_NAME = METRIC if LOSS != METRIC else None
        # THIS LINE OF CODE SETS THE BPROP_ENTRY ATTRIBUTE TO THE VALUE OF THE __FIND_BPROP_ENTRY__ METHOD. THIS ATTRIBUTE IS USED TO STORE THE INDEX OF THE ENTRY LAYER FOR BACK PROPAGATION.
        self.BPROP_ENTRY = self.__FIND_BPROP_ENTRY__()
        # THIS LINE OF CODE SETS THE TRAINING ATTRIBUTE TO FALSE. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE NETWORK IS CURRENTLY TRAINING.
        self.TRAINING = False
        # THIS LINE OF CODE SETS THE __INITIALIZED__ ATTRIBUTE TO FALSE. THIS ATTRIBUTE IS USED TO DETERMINE WHETHER THE MODEL HAS BEEN INITIALIZED.
        self.__INITIALIZED__ = False

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
        X_SHAPE = list(
            X_SHAPE)  # THIS LINE OF CODE CONVERTS THE X_SHAPE PARAMETER TO A LIST.
        # THIS LINE OF CODE SETS THE FIRST ELEMENT OF THE X_SHAPE PARAMETER TO THE VALUE OF THE BATCH_SIZE PARAMETER.
        X_SHAPE[0] = self.BATCH_SIZE
        # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
        for LAYER in self.LAYERS:
            # THIS LINE OF CODE CALLS THE SETUP METHOD OF THE LAYER OBJECT.
            LAYER.setup(X_SHAPE)
            # THIS LINE OF CODE SETS THE X_SHAPE PARAMETER TO THE VALUE OF THE SHAPE METHOD OF THE LAYER OBJECT.
            X_SHAPE = LAYER.shape(X_SHAPE)
        # THIS LINE OF CODE SETS THE __N_LAYERS__ ATTRIBUTE TO THE LENGTH OF THE LAYERS ATTRIBUTE.
        self.__N_LAYERS__ = len(self.LAYERS)
        # THIS LINE OF CODE CALLS THE SETUP METHOD OF THE OPTIMIZER OBJECT.
        self.OPTIMIZER.SETUP(self)
        # THIS LINE OF CODE SETS THE __INITIALIZED__ ATTRIBUTE TO TRUE.
        self.__INITIALIZED__ = True

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
        if len(self.LAYERS) > 0 and not hasattr(self.LAYERS[-1], "PARAMETERS"):  # THIS LINE OF CODE CHECKS IF THE LAST LAYER IN THE LAYERS ATTRIBUTE HAS THE PARAMETERS ATTRIBUTE.
            return -1  # THIS LINE OF CODE RETURNS -1.
        # THIS LINE OF CODE RETURNS THE LENGTH OF THE LAYERS ATTRIBUTE.
        return len(self.LAYERS)

    def FIT(self, X, Y):
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
        assert X is not None, "X CANNOT BE NONE"  # THIS LINE OF CODE CHECKS IF THE X PARAMETER IS NOT NONE.
        # THIS LINE OF CODE CHECKS IF THE Y PARAMETER IS NOT NONE.
        assert Y is not None, "Y CANNOT BE NONE"
        # THIS LINE OF CODE CHECKS IF THE LENGTH OF THE X PARAMETER IS EQUAL TO THE LENGTH OF THE Y PARAMETER.
        assert len(X) == len(Y), "X AND Y MUST HAVE THE SAME LENGTH"
        # THIS LINE OF CODE CHECKS IF THE __INITIALIZED__ ATTRIBUTE IS FALSE.
        if not self.__INITIALIZED__:
            # THIS LINE OF CODE CALLS THE __SETUP_LAYERS__ METHOD.
            self.__SETUP_LAYERS__(X.shape)
        if Y.ndim == 1:  # THIS LINE OF CODE CHECKS IF THE Y PARAMETER IS A 1D ARRAY.
            # THIS LINE OF CODE ADDS A NEW AXIS TO THE Y PARAMETER.
            Y = Y[:, np.newaxis]
        # THIS LINE OF CODE CALLS THE __SETUP_INPUT__ METHOD.
        self.__SETUP_INPUT__(X, Y)
        # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO TRUE.
        self.IS_TRAINING = True
        # THIS LINE OF CODE CALLS THE OPTIMIZE METHOD OF THE OPTIMIZER OBJECT.
        self.OPTIMIZER.optimize(self)
        # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO FALSE.
        self.IS_TRAINING = False

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
        Y_PREDICTION = self.__FORWARD__(
            X)  # THIS LINE OF CODE SETS THE Y_PREDICTION VARIABLE TO THE VALUE OF THE __FORWARD__ METHOD.
        # THIS LINE OF CODE SETS THE GRAD VARIABLE TO THE VALUE OF THE LOSS_GRAD METHOD.
        GRAD = self.LOSS_GRAD(Y, Y_PREDICTION)
        # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
        for LAYER in reversed(self.LAYERS[: self.BPROP_ENTRY]):
            # THIS LINE OF CODE SETS THE GRAD VARIABLE TO THE VALUE OF THE BACKWARD_PASS METHOD OF THE LAYER OBJECT.
            GRAD = LAYER.BACKWARD_PASS(GRAD)
        # THIS LINE OF CODE RETURNS THE VALUE OF THE LOSS METHOD.
        return self.LOSS(Y, Y_PREDICTION)

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
        for LAYER in self.LAYERS:  # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
            # THIS LINE OF CODE SETS THE X VARIABLE TO THE VALUE OF THE FORWARD_PASS METHOD OF THE LAYER OBJECT.
            X = LAYER.FORWARD_PASS(X)
        return X  # THIS LINE OF CODE RETURNS THE X VARIABLE.

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
        assert X is not None, "X CANNOT BE NONE"  # THIS LINE OF CODE CHECKS IF THE X PARAMETER IS NOT NONE.
        # THIS LINE OF CODE CHECKS IF THE __INITIALIZED__ ATTRIBUTE IS FALSE.
        if not self.__INITIALIZED__:
            # THIS LINE OF CODE CALLS THE __SETUP_LAYERS__ METHOD.
            self.__SETUP_LAYERS__(X.shape)
        Y = []  # THIS LINE OF CODE CREATES AN EMPTY LIST.
        # THIS LINE OF CODE SETS THE X_BATCH VARIABLE TO THE VALUE OF THE BATCH_ITERATOR FUNCTION.
        X_BATCH = BATCH_ITERATOR(X, self.BATCH_SIZE)
        for XB in X_BATCH:  # THIS LINE OF CODE ITERATES THROUGH EACH BATCH IN THE X_BATCH VARIABLE.
            # THIS LINE OF CODE APPENDS THE VALUE OF THE __FORWARD__ METHOD TO THE Y LIST.
            Y.append(self.__FORWARD__(XB))
        # THIS LINE OF CODE RETURNS THE CONCATENATED VALUE OF THE Y LIST.
        return np.concatenate(Y)

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
        TRAINING_PHASE = self.IS_TRAINING  # THIS LINE OF CODE SETS THE TRAINING_PHASE VARIABLE TO THE VALUE OF THE IS_TRAINING ATTRIBUTE.
        if TRAINING_PHASE:  # THIS LINE OF CODE CHECKS IF THE TRAINING_PHASE VARIABLE IS TRUE.
            # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO FALSE.
            self.IS_TRAINING = False
        # THIS LINE OF CODE CHECKS IF THE X AND Y PARAMETERS ARE NONE.
        if X is None and Y is None:
            # THIS LINE OF CODE SETS THE Y_PREDICTION VARIABLE TO THE VALUE OF THE PREDICT METHOD.
            Y_PREDICTION = self.PREDICT(self.X)
            # THIS LINE OF CODE SETS THE SCORE VARIABLE TO THE VALUE OF THE METRIC METHOD.
            ERROR = self.METRIC(self.Y, Y_PREDICTION)
        else:  # THIS LINE OF CODE RUNS IF THE IS STATEMENT IS FALSE.
            # THIS LINE OF CODE SETS THE Y_PREDICTION VARIABLE TO THE VALUE OF THE PREDICT METHOD.
            Y_PREDICTION = self.PREDICT(X)
            # THIS LINE OF CODE SETS THE SCORE VARIABLE TO THE VALUE OF THE METRIC METHOD.
            ERROR = self.METRIC(Y, Y_PREDICTION)
        if TRAINING_PHASE:  # THIS LINE OF CODE CHECKS IF THE TRAINING_PHASE VARIABLE IS TRUE.
            # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO TRUE.
            self.IS_TRAINING = True
        return ERROR  # THIS LINE OF CODE RETURNS THE ERROR VARIABLE.

    def RESET(self):
        """RESET THE NETWORK.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        NONE
        """
        self.__INITIALIZED__ = False  # THIS LINE OF CODE SETS THE __INITIALIZED__ ATTRIBUTE TO FALSE.

    def __SHUFFLE_DATASET__(self):
        """SHUFFLE THE DATASET.

        PARAMETERS
        ----------
        NONE

        RETURNS
        -------
        NONE
        """
        assert self.X is not None, "X CANNOT BE NONE"  # THIS LINE OF CODE CHECKS IF THE X ATTRIBUTE IS NOT NONE.
        # THIS LINE OF CODE CHECKS IF THE Y ATTRIBUTE IS NOT NONE.
        assert self.Y is not None, "Y CANNOT BE NONE"
        # THIS LINE OF CODE SETS THE N_SAMPLES VARIABLE TO THE VALUE OF THE X ATTRIBUTE.
        N_SAMPLES = self.X.shape[0]
        # THIS LINE OF CODE SETS THE INDICES VARIABLE TO THE VALUE OF THE arange METHOD.
        INDICES = np.arange(N_SAMPLES)
        # THIS LINE OF CODE SETS THE ORIGINAL_LIST VARIABLE TO THE VALUE OF THE INDICES VARIABLE.
        ORIGINAL_LIST = list(INDICES)
        SHUFFLED_LIST = []  # THIS LINE OF CODE CREATES AN EMPTY LIST.
        # THIS LINE OF CODE CHECKS IF THE LENGTH OF THE ORIGINAL_LIST VARIABLE IS GREATER THAN 0.
        while len(ORIGINAL_LIST) > 0:
            # THIS LINE OF CODE SETS THE picked_item VARIABLE TO THE VALUE OF THE choice METHOD.
            picked_item = secrets.choice(ORIGINAL_LIST)
            # THIS LINE OF CODE APPENDS THE VALUE OF THE picked_item VARIABLE TO THE SHUFFLED_LIST VARIABLE.
            SHUFFLED_LIST.append(picked_item)
            # THIS LINE OF CODE REMOVES THE VALUE OF THE picked_item VARIABLE FROM THE ORIGINAL_LIST VARIABLE.
            ORIGINAL_LIST.remove(picked_item)
        # THIS LINE OF CODE SETS THE INDICES VARIABLE TO THE VALUE OF THE SHUFFLED_LIST VARIABLE.
        INDICES = np.array(SHUFFLED_LIST)
        # THIS LINE OF CODE SETS THE X ATTRIBUTE TO THE VALUE OF THE take METHOD.
        self.X = self.X.take(INDICES, axis=0)
        # THIS LINE OF CODE SETS THE y ATTRIBUTE TO THE VALUE OF THE take METHOD.
        self.Y = self.Y.take(INDICES, axis=0)

    def __PREDICT__(self, X=None):
        """MAKE PREDICTIONS USING THE NETWORK.

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT DATA.

        RETURNS
        -------
        NUMPY ARRAY
            PREDICTED DATA.
        """
        raise NotImplementedError()  # THIS LINE OF CODE RAISES THE NOTIMPLEMENTEDERROR EXCEPTION.

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
        for LAYER in self.LAYERS:  # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
            # THIS LINE OF CODE CHECKS IF THE LAYER OBJECT HAS THE PARAMETERS ATTRIBUTE.
            if hasattr(LAYER, "PARAMETERS"):
                yield LAYER  # THIS LINE OF CODE RETURNS THE LAYER OBJECT.

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
        PARAMS = []  # THIS LINE OF CODE CREATES AN EMPTY LIST.
        # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE PARAMETRIC_LAYERS PROPERTY.
        for LAYER in self.PARAMETRIC_LAYERS:
            # THIS LINE OF CODE APPENDS THE VALUE OF THE PARAMETERS ATTRIBUTE TO THE PARAMS LIST.
            PARAMS.append(LAYER.PARAMETERS)
        return PARAMS  # THIS LINE OF CODE RETURNS THE PARAMS LIST.

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
        return self.TRAINING  # THIS LINE OF CODE RETURNS THE TRAINING ATTRIBUTE.

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
        self.TRAINING = TRAIN  # THIS LINE OF CODE SETS THE TRAINING ATTRIBUTE TO THE VALUE OF THE TRAIN PARAMETER.
        # THIS LINE OF CODE ITERATES THROUGH EACH LAYER IN THE LAYERS ATTRIBUTE.
        for LAYER in self.LAYERS:
            # THIS LINE OF CODE CHECKS IF THE LAYER OBJECT IS AN INSTANCE OF THE PHASE_MIXIN CLASS.
            if isinstance(LAYER, PHASE_MIXIN):
                # THIS LINE OF CODE SETS THE IS_TRAINING ATTRIBUTE TO THE VALUE OF THE TRAIN PARAMETER.
                LAYER.IS_TRAINING = TRAIN

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
        return self.__N_LAYERS__  # THIS LINE OF CODE RETURNS THE __N_LAYERS__ ATTRIBUTE.

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
        return sum([LAYER.PARAMETERS.N_PARAMS for LAYER in self.PARAMETRIC_LAYERS])  # THIS LINE OF CODE RETURNS THE SUM OF THE N_PARAMS ATTRIBUTE OF EACH LAYER IN THE PARAMETRIC_LAYERS PROPERTY.
