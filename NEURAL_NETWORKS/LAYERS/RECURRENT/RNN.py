from autograd import elementwise_grad
import numpy as np

from NEURAL_NETWORKS.ACTIVATIONS import *
from NEURAL_NETWORKS.INITIALIZATIONS import *
from NEURAL_NETWORKS.layers import LAYER, PARAM_MIXIN
from NEURAL_NETWORKS.PARAMETERS import PARAMETER

class RNN(LAYER, PARAM_MIXIN):
    """VANILLA RNN LAYER

    ATTRIBUTES
    ----------
    HIDDEN_DIM : INT
        NUMBER OF HIDDEN UNITS
    ACTIVATION : FUNCTION
        ACTIVATION FUNCTION
    INNER_INIT : FUNCTION
        INNER WEIGHT INITIALIZATION FUNCTION
    PARAMETERS : PARAMETER
        PARAMETER OBJECT
    RETURN_SEQUENCES : BOOL
        IF TRUE, RETURNS ALL OUTPUTS, ELSE RETURNS LAST OUTPUT

    METHODS
    -------
    SETUP(X_SHAPE)
        SETS UP PARAMETERS
    FORWARD_PASS(X)
        FORWARD PROPAGATION
    BACKWARD_PASS(DELTA)
        BACKWARD PROPAGATION
    SHAPE(X_SHAPE)
        RETURNS SHAPE OF OUTPUT
    """

    def __init__(self, HIDDEN_DIM, ACTIVATION=TANH, INNER_INIT=ORTHOGONAL, PARAMETERS=PARAMETER(), RETURN_SEQUENCES=True):
        """INITIALIZE VANILLA RNN LAYER
        
        PARAMETERS
        ----------
        HIDDEN_DIM : INT
            NUMBER OF HIDDEN UNITS
        ACTIVATION : FUNCTION
            ACTIVATION FUNCTION
        INNER_INIT : FUNCTION
            INNER WEIGHT INITIALIZATION FUNCTION
        PARAMETERS : PARAMETER
            PARAMETER OBJECT
        RETURN_SEQUENCES : BOOL
            IF TRUE, RETURNS ALL OUTPUTS, ELSE RETURNS LAST OUTPUT
        
        RETURNS
        -------
        NONE
        """
        self.RETURN_SEQUENCES = RETURN_SEQUENCES  # IF TRUE, RETURNS ALL OUTPUTS, ELSE RETURNS LAST OUTPUT
        self.HIDDEN_DIM = HIDDEN_DIM  # NUMBER OF HIDDEN UNITS
        self.INNER_INIT = INNER_INIT  # INNER WEIGHT INITIALIZATION FUNCTION
        self.ACTIVATION = ACTIVATION  # ACTIVATION FUNCTION
        self.__PARAMETERS__ = PARAMETERS  # PARAMETER OBJECT
        self.LAST_INPUT = None  # LAST INPUT
        self.STATES = None  # STATES: HIDDEN STATES
        self.H_PREV = None  # PREVIOUS HIDDEN STATE
        self.INPUT_DIM = None  # INPUT DIMENSION

    def SETUP(self, X_SHAPE):
        """SETUP PARAMETERS

        PARAMETERS
        ----------
        X_SHAPE : TUPLE
            INPUT SHAPE
        
        RETURNS
        -------
        NONE
        """
        self.INPUT_DIM = X_SHAPE[2]  # INPUT DIMENSION
        self.__PARAMETERS__["W"] = self.__PARAMETERS__.INIT((self.INPUT_DIM, self.HIDDEN_DIM))  # INITIALIZE WEIGHTS
        self.__PARAMETERS__["b"] = np.full((self.HIDDEN_DIM,), self.__PARAMETERS__.INITIAL_BIAS)  # INITIALIZE BIAS
        self.__PARAMETERS__["U"] = self.INNER_INIT((self.HIDDEN_DIM, self.HIDDEN_DIM))  # INNER WEIGHT INITIALIZATION
        self.__PARAMETERS__.INIT_GRAD()  # INITIALIZE GRADIENTS
        self.H_PREV = np.zeros((X_SHAPE[0], self.HIDDEN_DIM))  # INITIALIZE PREVIOUS HIDDEN STATE

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT ARRAY

        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT ARRAY
        """
        assert self.H_PREV is not None, "SETUP() MUST BE CALLED BEFORE FORWARD_PASS()"
        self.LAST_INPUT = X  # SAVE LAST INPUT
        N_SAMPLES, N_TIMESTEPS, INPUT_SHAPE = X.shape  # GET INPUT SHAPE
        STATES = np.zeros((N_SAMPLES, N_TIMESTEPS + 1, self.HIDDEN_DIM))  # INITIALIZE STATES
        STATES[:, -1, :] = self.H_PREV.copy()  # SET LAST STATE TO PREVIOUS HIDDEN STATE
        p = self.__PARAMETERS__  # GET PARAMETERS
        for STEP in range(N_TIMESTEPS):  # FORWARD PROPAGATION
            STATES[:, STEP, :] = np.tanh(np.dot(X[:, STEP, :], p["W"]) + np.dot(STATES[:, STEP - 1, :], p["U"]) + p["b"])  # HIDDEN STATE
        self.STATES = STATES  # SAVE STATES
        self.H_PREV = STATES[:, N_TIMESTEPS - 1, :].copy()  # SAVE PREVIOUS HIDDEN STATE
        if self.RETURN_SEQUENCES:  # RETURN OUTPUT
            return STATES[:, 0:-1, :]  # RETURN ALL OUTPUTS
        else:  # RETURN LAST OUTPUT
            return STATES[:, -2, :]  # RETURN LAST OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION

        PARAMETERS
        ----------
        DELTA : NUMPY ARRAY
            DELTA ARRAY
        
        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT GRADIENT
        """
        assert self.STATES is not None, "FORWARD_PASS() MUST BE CALLED BEFORE BACKWARD_PASS()"
        assert self.LAST_INPUT is not None, "FORWARD_PASS() MUST BE CALLED BEFORE BACKWARD_PASS()"
        assert self.H_PREV is not None, "SETUP() MUST BE CALLED BEFORE BACKWARD_PASS()"
        assert self.INPUT_DIM is not None, "SETUP() MUST BE CALLED BEFORE BACKWARD_PASS()"
        if len(DELTA.shape) == 2:  # IF DELTA IS 2D, CONVERT TO 3D
            DELTA = DELTA[:, np.newaxis, :]  # CONVERT TO 3D
        N_SAMPLES, N_TIMESTEPS, INPUT_SHAPE = DELTA.shape  # GET DELTA SHAPE
        P = self.__PARAMETERS__  # GET PARAMETERS
        GRAD = { KEY: np.zeros_like(P[KEY]) for KEY in P.KEYS() }  # INITIALIZE GRADIENTS
        DH_NEXT = np.zeros((N_SAMPLES, INPUT_SHAPE))  # INITIALIZE NEXT HIDDEN STATE GRADIENT
        OUTPUT = np.zeros((N_SAMPLES, N_TIMESTEPS, self.INPUT_DIM))  # INITIALIZE OUTPUT GRADIENT
        for STEP in reversed(range(N_TIMESTEPS)):  # BACKWARD PROPAGATION
            DHI = elementwise_grad(self.ACTIVATION, self.STATES[:, STEP, :]) * (DELTA[:, STEP, :] + DH_NEXT)  # HIDDEN STATE GRADIENT
            GRAD["W"] += np.dot(self.LAST_INPUT[:, STEP, :].T, DHI)  # UPDATE GRADIENTS
            GRAD["b"] += DELTA[:, STEP, :].sum(axis=0)  # UPDATE GRADIENTS
            GRAD["U"] += np.dot(self.STATES[:, STEP - 1, :].T, DHI)  # UPDATE GRADIENTS
            DH_NEXT = np.dot(DHI, P["U"].T)  # UPDATE NEXT HIDDEN STATE GRADIENT
            D = np.dot(DELTA[:, STEP, :], P["U"].T)  # OUTPUT GRADIENT
            OUTPUT[:, STEP, :] = np.dot(D, P["W"].T)  # UPDATE OUTPUT GRADIENT
        for KEY in GRAD.keys():  # UPDATE PARAMETERS
            self.__PARAMETERS__.UPDATE_GRAD(KEY, GRAD[KEY])  # UPDATE GRADIENTS
        return OUTPUT  # RETURN OUTPUT GRADIENT

    def SHAPE(self, X_SHAPE):
        """GET OUTPUT SHAPE

        PARAMETERS
        ----------
        X_SHAPE : TUPLE
            INPUT SHAPE

        RETURNS
        -------
        TUPLE
            OUTPUT SHAPE
        """
        if self.RETURN_SEQUENCES:  # RETURN OUTPUT SHAPE
            return X_SHAPE[0], X_SHAPE[1], self.HIDDEN_DIM  # RETURN OUTPUT SHAPE
        else:  # RETURN OUTPUT SHAPE
            return X_SHAPE[0], self.HIDDEN_DIM  # RETURN OUTPUT SHAPE