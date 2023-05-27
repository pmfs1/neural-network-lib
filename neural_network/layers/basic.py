from autograd import elementwise_grad

from neural_network.parameters import *

np.random.seed(9999)  # SET SEED FOR REPRODUCIBILITY OF RESULTS

class LAYER(object):
    """BASE CLASS FOR ALL LAYERS.
    
    METHODS:
    --------
    SETUP(X_SHAPE)
        ALLOCATES INITIAL WEIGHTS.
    FORWARD_PASS(X)
        FORWARD PROPAGATION.
    BACKWARD_PASS(DELTA)
        BACKWARD PROPAGATION.
    SHAPE(X_SHAPE)
        RETURNS SHAPE OF THE CURRENT LAYER.
    
    ATTRIBUTES:
    -----------
    LAST_INPUT: NUMPY ARRAY
        LAST INPUT TO THE LAYER.
    """

    def SETUP(self, X_SHAPE):
        """ALLOCATES INITIAL WEIGHTS.
        
        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.
        
        RETURNS:
        --------
        NONE
        """
        pass

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION.

        PARAMETERS:
        ----------
        X: NUMPY ARRAY
            INPUT TO THE LAYER.
        
        RETURNS:
        --------
        FORWARD_PASS: NUMPY ARRAY
            OUTPUT OF THE LAYER.
        """
        raise NotImplementedError()  # RAISE NOT IMPLEMENTED ERROR

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION.

        PARAMETERS:
        ----------
        DELTA: NUMPY ARRAY
            DELTA FROM THE NEXT LAYER.
        
        RETURNS:
        --------
        BACKWARD_PASS: NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER.
        """
        raise NotImplementedError()  # RAISE NOT IMPLEMENTED ERROR

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE CURRENT LAYER.

        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.
        
        RETURNS:
        --------
        SHAPE: TUPLE
            SHAPE OF THE CURRENT LAYER.
        """
        raise NotImplementedError()  # RAISE NOT IMPLEMENTED ERROR

class PARAM_MIXIN(object):
    """MIXIN CLASS FOR LAYERS WITH PARAMETERS.

    ATTRIBUTES:
    -----------
    PARAMETERS: PARAMETERS
        PARAMETERS OF THE LAYER.
    """

    @property
    def PARAMETERS(self):
        """PARAMETERS OF THE LAYER."""
        raise NotImplementedError()  # RAISE NOT IMPLEMENTED ERROR

class PHASE_MIXIN(object):
    """MIXIN CLASS FOR LAYERS WITH PHASES.

    ATTRIBUTES:
    -----------
    __TRAIN__: BOOL
        PHASE OF THE LAYER.

    METHODS:
    --------
    IS_TRAINING
        RETURNS PHASE OF THE LAYER.
    IS_TESTING
        RETURNS PHASE OF THE LAYER.
    """
    __TRAIN__ = False  # DEFAULT PHASE IS TESTING

    @property
    def IS_TRAINING(self):
        """PHASE OF THE LAYER.
        
        RETURNS:
        --------
        IS_TRAINING: BOOL
            PHASE OF THE LAYER.
        """
        return self.__TRAIN__  # RETURN PHASE

    @IS_TRAINING.setter
    def IS_TRAINING(self, IS_TRAIN=True):
        """SET PHASE OF THE LAYER.
        
        PARAMETERS:
        -----------
        IS_TRAIN: BOOL
            PHASE OF THE LAYER.
        """
        self.__TRAIN__ = IS_TRAIN  # SET PHASE

    @property
    def IS_TESTING(self):
        """PHASE OF THE LAYER.

        RETURNS:
        --------
        IS_TESTING: BOOL
            PHASE OF THE LAYER.
        """
        return not self.__TRAIN__  # RETURN PHASE

    @IS_TESTING.setter
    def IS_TESTING(self, IS_TEST=True):
        """SET PHASE OF THE LAYER.

        PARAMETERS:
        -----------
        IS_TEST: BOOL
            PHASE OF THE LAYER.
        """
        self.__TRAIN__ = not IS_TEST  # SET PHASE

class DENSE(LAYER, PARAM_MIXIN):
    """DENSE LAYER.

    ATTRIBUTES:
    -----------
    OUTPUT_DIM: INT
        OUTPUT DIMENSION OF THE LAYER.
    LAST_INPUT: NUMPY ARRAY
        LAST INPUT TO THE LAYER.
    PARAMETERS: PARAMETERS
        PARAMETERS OF THE LAYER.

    METHODS:
    --------
    SETUP(X_SHAPE)
        ALLOCATES INITIAL WEIGHTS.
    FORWARD_PASS(X)
        FORWARD PROPAGATION.
    BACKWARD_PASS(DELTA)
        BACKWARD PROPAGATION.
    SHAPE(X_SHAPE)
        RETURNS SHAPE OF THE CURRENT LAYER.
    """

    def __init__(self, OUTPUT_DIM, PARAMETERS=PARAMETER()):
        """INITIALIZE DENSE LAYER.
        
        PARAMETERS:
        -----------
        OUTPUT_DIM: INT
            OUTPUT DIMENSION OF THE LAYER.
        PARAMETERS: PARAMETERS
            PARAMETERS OF THE LAYER.
        """
        self.__PARAMETERS__ = PARAMETERS  # SET PARAMETERS
        self.OUTPUT_DIM = OUTPUT_DIM  # SET OUTPUT DIMENSION
        self.LAST_INPUT = None  # INITIALIZE LAST INPUT
        self.__PARAMETERS__ = PARAMETERS  # INITIALIZE PARAMETERS

    def SETUP(self, X_SHAPE):
        """ALLOCATES INITIAL WEIGHTS.

        PARAMETERS:
        ----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.
        
        RETURNS:
        --------
        NONE
        """
        self.__PARAMETERS__.SETUP_WEIGHTS((X_SHAPE[1], self.OUTPUT_DIM))  # INITIALIZE WEIGHTS

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION.

        PARAMETERS:
        ----------
        X: NUMPY ARRAY
            INPUT TO THE LAYER.
        
        RETURNS:
        --------
        FORWARD_PASS: NUMPY ARRAY
            OUTPUT OF THE LAYER.
        """
        self.LAST_INPUT = X  # SET LAST INPUT
        return self.SET_WEIGHT(X)  # RETURN OUTPUT

    def SET_WEIGHT(self, X):
        """SET WEIGHTS OF THE LAYER.

        PARAMETERS:
        ----------
        X: NUMPY ARRAY
            INPUT TO THE LAYER.
        
        RETURNS:
        --------
        SET_WEIGHT: NUMPY ARRAY
            OUTPUT OF THE LAYER.
        """
        W = np.dot(X, self.__PARAMETERS__["W"])  # COMPUTE WEIGHTS
        return W + self.__PARAMETERS__["b"]  # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION.

        PARAMETERS:
        ----------
        DELTA: NUMPY ARRAY
            DELTA FROM THE NEXT LAYER.
        
        RETURNS:
        --------
        BACKWARD_PASS: NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER.
        """
        assert self.LAST_INPUT is not None, "FORWARD PASS NOT CALLED"  # ASSERT FORWARD PASS CALLED
        DW = np.dot(self.LAST_INPUT.T, DELTA)  # COMPUTE GRADIENTS
        DB = np.sum(DELTA, axis=0)  # COMPUTE GRADIENTS
        self.__PARAMETERS__.UPDATE_GRAD("W", DW)  # UPDATE GRADIENTS
        self.__PARAMETERS__.UPDATE_GRAD("b", DB)  # UPDATE GRADIENTS
        return np.dot(DELTA, self.__PARAMETERS__["W"].T)  # RETURN DELTA

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE CURRENT LAYER.

        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.
        
        RETURNS:
        --------
        SHAPE: TUPLE
            SHAPE OF THE CURRENT LAYER.
        """
        return X_SHAPE[0], self.OUTPUT_DIM  # RETURN SHAPE

class ACTIVATION(LAYER):
    """ACTIVATION LAYER.

    ATTRIBUTES:
    -----------
    LAST_INPUT: NUMPY ARRAY
        LAST INPUT TO THE LAYER.
    ACTIVATION: FUNCTION
        ACTIVATION FUNCTION.

    METHODS:
    --------
    FORWARD_PASS(X): NUMPY ARRAY
        FORWARD PROPAGATION.
    BACKWARD_PASS(DELTA): NUMPY ARRAY
        BACKWARD PROPAGATION.
    SHAPE(X_SHAPE): TUPLE
        RETURNS SHAPE OF THE CURRENT LAYER.
    """

    def __init__(self, ACTIVATION):
        """INITIALIZE ACTIVATION LAYER.

        PARAMETERS:
        -----------
        ACTIVATION: FUNCTION
            ACTIVATION FUNCTION.
        """
        self.LAST_INPUT = None  # INITIALIZE LAST INPUT
        self.ACTIVATION = ACTIVATION  # SET ACTIVATION FUNCTION

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION.

        PARAMETERS:
        -----------
        X: NUMPY ARRAY
            INPUT TO THE LAYER.
        
        RETURNS:
        --------
        FORWARD_PASS: NUMPY ARRAY
            OUTPUT OF THE LAYER.
        """
        self.LAST_INPUT = X  # SET LAST INPUT
        return self.ACTIVATION(X)  # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION.

        PARAMETERS:
        -----------
        DELTA: NUMPY ARRAY
            DELTA FROM THE NEXT LAYER.
        
        RETURNS:
        --------
        BACKWARD_PASS: NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER.
        """
        return elementwise_grad(ACTIVATION, self.LAST_INPUT) * DELTA  # RETURN DELTA

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE CURRENT LAYER.

        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.
        
        RETURNS:
        --------
        SHAPE: TUPLE
            SHAPE OF THE CURRENT LAYER.
        """
        return X_SHAPE  # RETURN SHAPE

class DROP_OUT(LAYER, PHASE_MIXIN):
    """RANDOMLY SET A FRACTION OF INPUTS TO 0 AT EACH TRAINING UPDATE.

    ATTRIBUTES:
    -----------
    P: FLOAT
        FRACTION OF INPUTS TO SET TO 0.
    __MASK__: NUMPY ARRAY
        MASK TO SET INPUTS TO 0.

    METHODS:
    --------
    FORWARD_PASS(X):
        FORWARD PROPAGATION.
    BACKWARD_PASS(DELTA):
        BACKWARD PROPAGATION.
    """

    def __init__(self, P=0.1):
        """INITIALIZE DROPOUT LAYER.

        PARAMETERS:
        -----------
        P: FLOAT
            FRACTION OF INPUTS TO SET TO 0.
        """
        self.P = P  # SET FRACTION OF INPUTS TO SET TO 0
        self.__MASK__ = None  # INITIALIZE MASK

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION.

        PARAMETERS:
        -----------
        X: NUMPY ARRAY
            INPUT TO THE LAYER.
        
        RETURNS:
        --------
        FORWARD_PASS: NUMPY ARRAY
            OUTPUT OF THE LAYER.
        """
        assert self.P >= 0.0 and self.P <= 1.0, "P SHOULD BE BETWEEN 0 AND 1"  # CHECK IF P IS BETWEEN 0 AND 1
        if self.IS_TRAINING:  # CHECK IF TRAINING
            self.__MASK__ = np.random.uniform(size=X.shape) > self.P  # SET MASK
            return X * self.__MASK__  # RETURN OUTPUT
        else:  # CHECK IF NOT TRAINING
            return X * (1.0 - self.P)  # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION.

        PARAMETERS:
        -----------
        DELTA: NUMPY ARRAY
            DELTA FROM THE NEXT LAYER.

        RETURNS:
        --------
        BACKWARD_PASS: NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER.
        """
        return DELTA * self.__MASK__  # RETURN DELTA

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE CURRENT LAYER.

        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.

        RETURNS:
        --------
        SHAPE: TUPLE
            SHAPE OF THE CURRENT LAYER.
        """
        return X_SHAPE  # RETURN SHAPE

class TIME_STEP_SLICER(LAYER):
    """TAKE A SPECIFIC TIME STEP FROM 3D TENSOR.
    
    ATTRIBUTES:
    -----------
    STEP: INT
        TIME STEP TO TAKE.
    
    METHODS:
    --------
    FORWARD_PASS(X):
        TAKE A SPECIFIC TIME STEP FROM 3D TENSOR.
    BACKWARD_PASS(DELTA):
        TAKE A SPECIFIC TIME STEP FROM 3D TENSOR.
    SHAPE(X_SHAPE):
        TAKE A SPECIFIC TIME STEP FROM 3D TENSOR.
    """

    def __init__(self, STEP=-1):
        """INITIALIZE TIME STEP SLICER LAYER.

        PARAMETERS:
        -----------
        STEP: INT
            TIME STEP TO TAKE.
        """
        self.STEP = STEP  # SET TIME STEP TO TAKE

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION.
        
        PARAMETERS:
        -----------
        X: NUMPY ARRAY
            INPUT TO THE LAYER.
            
        RETURNS:
        --------
        FORWARD_PASS: NUMPY ARRAY
            OUTPUT OF THE LAYER.
        """
        return X[:, self.STEP, :]  # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION.

        PARAMETERS:
        -----------
        DELTA: NUMPY ARRAY
            DELTA FROM THE NEXT LAYER.

        RETURNS:
        --------
        BACKWARD_PASS: NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER.
        """
        return np.repeat(DELTA[:, np.newaxis, :], 2, 1)  # RETURN DELTA

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE CURRENT LAYER.

        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.

        RETURNS:
        --------
        SHAPE: TUPLE
            SHAPE OF THE CURRENT LAYER.
        """
        return X_SHAPE[0], X_SHAPE[2]  # RETURN SHAPE

class TIME_DISTRIBUTED_DENSE(LAYER):
    """APPLY DENSE LAYER TO EACH TIME STEP OF 3D TENSOR.

    ATTRIBUTES:
    -----------
    OUTPUT_DIM: INT
        OUTPUT DIMENSION OF THE DENSE LAYER.
    N_TIME_STEPS: INT
        NUMBER OF TIME STEPS.
    DENSE: DENSE
        DENSE LAYER.
    INPUT_DIM: INT
        INPUT DIMENSION OF THE DENSE LAYER.
    
    METHODS:
    --------
    SETUP(X_SHAPE):
        SETUP THE LAYER.
    FORWARD_PASS(X):
        FORWARD PROPAGATION.
    BACKWARD_PASS(DELTA):
        BACKWARD PROPAGATION.
    SHAPE(X_SHAPE):
        RETURNS SHAPE OF THE CURRENT LAYER.
    """

    def __init__(self, OUTPUT_DIM):
        """INITIALIZE TIME DISTRIBUTED DENSE LAYER.

        PARAMETERS:
        -----------
        OUTPUT_DIM: INT
            OUTPUT DIMENSION OF THE DENSE LAYER.
        """
        self.OUTPUT_DIM = OUTPUT_DIM  # SET OUTPUT DIMENSION OF THE DENSE LAYER
        self.N_TIME_STEPS = None  # INITIALIZE NUMBER OF TIME STEPS
        self.DENSE = None  # INITIALIZE DENSE LAYER
        self.INPUT_DIM = None  # INITIALIZE INPUT DIMENSION OF THE DENSE LAYER

    def SETUP(self, X_SHAPE):
        """SETUP THE LAYER.

        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.
        """
        self.DENSE = DENSE(self.OUTPUT_DIM)  # INITIALIZE DENSE LAYER
        self.DENSE.SETUP((X_SHAPE[0], X_SHAPE[2]))  # SETUP DENSE LAYER
        self.INPUT_DIM = X_SHAPE[2]  # SET INPUT DIMENSION OF THE DENSE LAYER

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION.

        PARAMETERS:
        -----------
        X: NUMPY ARRAY
            INPUT TO THE LAYER.

        RETURNS:
        --------
        FORWARD_PASS: NUMPY ARRAY
            OUTPUT OF THE LAYER.
        """
        assert self.DENSE is not None, "SETUP MUST BE CALLED BEFORE FORWARD PASS."  # CHECK IF SETUP IS CALLED
        N_TIME_STEPS = X.shape[1]  # GET NUMBER OF TIME STEPS
        X = X.reshape(-1, X.shape[-1])  # RESHAPE INPUT
        Y = self.DENSE.FORWARD_PASS(X)  # FORWARD PROPAGATE THROUGH DENSE LAYER
        Y = Y.reshape((-1, N_TIME_STEPS, self.OUTPUT_DIM))  # RESHAPE OUTPUT
        return Y  # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION.
        
        PARAMETERS:
        -----------
        DELTA: NUMPY ARRAY
            DELTA FROM THE NEXT LAYER.

        RETURNS:
        --------
        BACKWARD_PASS: NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER.
        """
        assert self.DENSE is not None, "SETUP MUST BE CALLED BEFORE BACKWARD PASS."  # CHECK IF SETUP IS CALLED
        N_TIME_STEPS = DELTA.shape[1]  # GET NUMBER OF TIME STEPS
        X = DELTA.reshape(-1, DELTA.shape[-1])  # RESHAPE DELTA
        Y = self.DENSE.BACKWARD_PASS(X)  # BACKWARD PROPAGATE THROUGH DENSE LAYER
        Y = Y.reshape((-1, N_TIME_STEPS, self.INPUT_DIM))  # RESHAPE DELTA
        return Y  # RETURN DELTA

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE CURRENT LAYER.

        PARAMETERS:
        -----------
        X_SHAPE: TUPLE
            SHAPE OF THE INPUT.

        RETURNS:
        --------
        SHAPE: TUPLE
            SHAPE OF THE CURRENT LAYER.
        """
        return X_SHAPE[0], X_SHAPE[1], self.OUTPUT_DIM  # RETURN SHAPE

    @property
    def PARAMETERS(self):
        """RETURNS PARAMETERS OF THE LAYER."""
        assert self.DENSE is not None, "SETUP MUST BE CALLED BEFORE PARAMETERS."  # CHECK IF SETUP IS CALLED
        return self.DENSE.__PARAMETERS__  # RETURN PARAMETERS