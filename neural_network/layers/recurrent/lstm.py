import numpy as np
from autograd import elementwise_grad
from neural_network.activations import *
from neural_network.initializations import *
from neural_network.layers import LAYER, PARAM_MIXIN
from neural_network.parameters import PARAMETER

class LSTM(LAYER, PARAM_MIXIN):
    """LONG SHORT-TERM MEMORY LAYER

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
        """INITIALIZE LSTM LAYER

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
        self.RETURN_SEQUENCES = RETURN_SEQUENCES # IF TRUE, RETURNS ALL OUTPUTS, ELSE RETURNS LAST OUTPUT
        self.HIDDEN_DIM = HIDDEN_DIM # NUMBER OF HIDDEN UNITS
        self.INNER_INIT = INNER_INIT # INNER WEIGHT INITIALIZATION FUNCTION
        self.ACTIVATION = ACTIVATION # ACTIVATION FUNCTION
        self.__PARAMETERS__ = PARAMETERS # PARAMETER OBJECT
        self.LAST_INPUT = None # LAST INPUT
        self.STATES = None # STATES
        self.OUTPUTS = None # OUTPUTS
        self.GATES = None # GATES
        self.H_PREV = None # PREVIOUS HIDDEN STATE
        self.INPUT_DIM = None # INPUT DIMENSION
        self.W = None # INPUT WEIGHTS
        self.U = None # HIDDEN WEIGHTS

    def SETUP(self, X_SHAPE):
        """SETUP PARAMETERS

        PARAMETERS
        ----------
        X_SHAPE : TUPLE
            SHAPE OF INPUT
        
        RETURNS
        -------
        NONE

        NAMING CONVENTION
        -----------------
        W_I : INPUT WEIGHTS
        W_F : FORGET WEIGHTS
        W_O : OUTPUT WEIGHTS
        W_C : CELL WEIGHTS
        U_I : INPUT WEIGHTS
        U_F : FORGET WEIGHTS
        U_O : OUTPUT WEIGHTS
        U_C : CELL WEIGHTS
        b_I : INPUT BIAS
        b_F : FORGET BIAS
        b_O : OUTPUT BIAS
        b_C : CELL BIAS
        """
        self.INPUT_DIM = X_SHAPE[2] # INPUT DIMENSION
        W_PARAMETERS = ["W_i", "W_f", "W_o", "W_c"] # INPUT -> HIDDEN
        U_PARAMETERS = ["U_i", "U_f", "U_o", "U_c"] # HIDDEN -> HIDDEN
        B_PARAMETERS = ["b_i", "b_f", "b_o", "b_c"] # BIAS
        for PARAM in W_PARAMETERS: # INITIALIZE PARAMETERS
            self.__PARAMETERS__[PARAM] = self.__PARAMETERS__.INIT((self.INPUT_DIM, self.HIDDEN_DIM)) # INITIALIZE WEIGHTS
        for PARAM in U_PARAMETERS: # INITIALIZE PARAMETERS
            self.__PARAMETERS__[PARAM] = self.INNER_INIT((self.HIDDEN_DIM, self.HIDDEN_DIM)) # INITIALIZE WEIGHTS
        for PARAM in B_PARAMETERS: # INITIALIZE PARAMETERS
            self.__PARAMETERS__[PARAM] = np.full((self.HIDDEN_DIM,), self.__PARAMETERS__.INITIAL_BIAS) # INITIALIZE BIAS
        self.W = [self.__PARAMETERS__[PARAM] for PARAM in W_PARAMETERS] # INPUT WEIGHTS
        self.U = [self.__PARAMETERS__[PARAM] for PARAM in U_PARAMETERS] # HIDDEN WEIGHTS
        self.__PARAMETERS__.INIT_GRAD() # INITIALIZE GRADIENTS
        self.H_PREV = np.zeros((X_SHAPE[0], self.HIDDEN_DIM)) # PREVIOUS HIDDEN STATE
        self.O_PREV = np.zeros((X_SHAPE[0], self.HIDDEN_DIM)) # PREVIOUS OUTPUT

    def FORWARD_PASS(self, X):
        """FORWARD PROPAGATION

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT
        
        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT
        """
        assert self.W is not None, "SETUP HAS NOT BEEN CALLED" # ENSURE SETUP HAS BEEN CALLED
        assert self.U is not None, "SETUP HAS NOT BEEN CALLED" # ENSURE SETUP HAS BEEN CALLED
        N_SAMPLES, N_TIMESTEPS, INPUT_SHAPE = X.shape # GET INPUT SHAPE
        P = self.__PARAMETERS__ # PARAMETERS
        self.LAST_INPUT = X # LAST INPUT
        self.STATES = np.zeros((N_SAMPLES, N_TIMESTEPS + 1, self.HIDDEN_DIM)) # STATES
        self.OUTPUTS = np.zeros((N_SAMPLES, N_TIMESTEPS + 1, self.HIDDEN_DIM)) # OUTPUTS
        self.GATES = {KEY: np.zeros((N_SAMPLES, N_TIMESTEPS, self.HIDDEN_DIM)) for KEY in ["i", "f", "o", "c"]} # GATES
        self.STATES[:, -1, :] = self.H_PREV # INITIALIZE STATES
        self.OUTPUTS[:, -1, :] = self.O_PREV # INITIALIZE OUTPUTS
        for STEP in range(N_TIMESTEPS): # LOOP OVER TIMESTEPS
            T_GATES = np.dot(X[:, STEP, :], self.W) + np.dot(self.OUTPUTS[:, STEP - 1, :], self.U) # INPUT * WEIGHTS + OUTPUT * WEIGHTS
            self.GATES["i"][:, STEP, :] = SIGMOID(T_GATES[:, 0, :] + P["b_i"]) # INPUT
            self.GATES["f"][:, STEP, :] = SIGMOID(T_GATES[:, 1, :] + P["b_f"]) # FORGET
            self.GATES["o"][:, STEP, :] = SIGMOID(T_GATES[:, 2, :] + P["b_o"]) # OUTPUT
            self.GATES["c"][:, STEP, :] = self.ACTIVATION(T_GATES[:, 3, :] + P["b_c"]) # CELL
            self.STATES[:, STEP, :] = (
                self.STATES[:, STEP - 1, :] * self.GATES["f"][:, STEP, :] # FORGET GATE
                + self.GATES["i"][:, STEP, :] * self.GATES["c"][:, STEP, :] # INPUT GATE
            ) # STATE
            self.OUTPUTS[:, STEP, :] = self.GATES["o"][:, STEP, :] * self.ACTIVATION(self.STATES[:, STEP, :]) # OUTPUT
        self.H_PREV = self.STATES[:, N_TIMESTEPS - 1, :].copy() # PREVIOUS HIDDEN STATE
        self.O_PREV = self.OUTPUTS[:, N_TIMESTEPS - 1, :].copy() # PREVIOUS OUTPUT
        if self.RETURN_SEQUENCES: # RETURN SEQUENCES
            return self.OUTPUTS[:, 0:-1, :] # RETURN OUTPUTS
        else: # RETURN LAST OUTPUT
            return self.OUTPUTS[:, -2, :] # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """BACKWARD PROPAGATION

        PARAMETERS
        ----------
        DELTA : NUMPY ARRAY
            DELTA
        
        RETURNS
        -------
        NUMPY ARRAY
            GRADIENT
        """
        assert self.W is not None, "SETUP HAS NOT BEEN CALLED" # ENSURE SETUP HAS BEEN CALLED
        assert self.U is not None, "SETUP HAS NOT BEEN CALLED" # ENSURE SETUP HAS BEEN CALLED
        assert self.STATES is not None, "FORWARD PASS HAS NOT BEEN CALLED" # ENSURE FORWARD PASS HAS BEEN CALLED
        assert self.OUTPUTS is not None, "FORWARD PASS HAS NOT BEEN CALLED" # ENSURE FORWARD PASS HAS BEEN CALLED
        assert self.GATES is not None, "FORWARD PASS HAS NOT BEEN CALLED" # ENSURE FORWARD PASS HAS BEEN CALLED
        assert self.LAST_INPUT is not None, "FORWARD PASS HAS NOT BEEN CALLED" # ENSURE FORWARD PASS HAS BEEN CALLED
        assert self.H_PREV is not None, "FORWARD PASS HAS NOT BEEN CALLED" # ENSURE FORWARD PASS HAS BEEN CALLED
        assert self.O_PREV is not None, "FORWARD PASS HAS NOT BEEN CALLED" # ENSURE FORWARD PASS HAS BEEN CALLED
        assert self.INPUT_DIM is not None, "SETUP HAS NOT BEEN CALLED" # ENSURE SETUP HAS BEEN CALLED
        if len(DELTA.shape) == 2: # IF DELTA IS 2D
            DELTA = DELTA[:, np.newaxis, :] # ADD DIMENSION
        N_SAMPLES, N_TIMESTEPS, INPUT_SHAPE = DELTA.shape # GET DELTA SHAPE
        GRAD = {KEY: np.zeros_like(self.__PARAMETERS__[KEY]) for KEY in self.__PARAMETERS__.KEYS()} # GRADIENT
        DH_NEXT = np.zeros((N_SAMPLES, INPUT_SHAPE)) # NEXT HIDDEN STATE
        OUTPUT = np.zeros((N_SAMPLES, N_TIMESTEPS, self.INPUT_DIM)) # OUTPUT
        for STEP in reversed(range(N_TIMESTEPS)): # LOOP OVER TIMESTEPS
            DHI = DELTA[:, STEP, :] * self.GATES["o"][:, STEP, :] * elementwise_grad(self.ACTIVATION, self.STATES[:, STEP, :]) + DH_NEXT # HIDDEN STATE
            OG = DELTA[:, STEP, :] * self.ACTIVATION(self.STATES[:, STEP, :]) # OUTPUT GATE
            DE_O = OG * elementwise_grad(SIGMOID, self.GATES["o"][:, STEP, :]) # OUTPUT
            GRAD["W_o"] += np.dot(self.LAST_INPUT[:, STEP, :].T, DE_O) # INPUT WEIGHTS
            GRAD["U_o"] += np.dot(self.OUTPUTS[:, STEP - 1, :].T, DE_O) # HIDDEN WEIGHTS
            GRAD["b_o"] += DE_O.sum(axis=0) # BIAS
            DE_F = (DHI * self.STATES[:, STEP - 1, :]) * elementwise_grad(SIGMOID, self.GATES["f"][:, STEP, :]) # FORGET
            GRAD["W_f"] += np.dot(self.LAST_INPUT[:, STEP, :].T, DE_F) # INPUT WEIGHTS
            GRAD["U_f"] += np.dot(self.OUTPUTS[:, STEP - 1, :].T, DE_F) # HIDDEN WEIGHTS
            GRAD["b_f"] += DE_F.sum(axis=0) # BIAS
            DE_I = (DHI * self.GATES["c"][:, STEP, :]) * elementwise_grad(SIGMOID, self.GATES["i"][:, STEP, :]) # INPUT
            GRAD["W_i"] += np.dot(self.LAST_INPUT[:, STEP, :].T, DE_I) # INPUT WEIGHTS
            GRAD["U_i"] += np.dot(self.OUTPUTS[:, STEP - 1, :].T, DE_I) # HIDDEN WEIGHTS
            GRAD["b_i"] += DE_I.sum(axis=0) # BIAS
            DE_C = (DHI * self.GATES["i"][:, STEP, :]) * elementwise_grad(self.ACTIVATION, self.GATES["c"][:, STEP, :]) # CELL
            GRAD["W_c"] += np.dot(self.LAST_INPUT[:, STEP, :].T, DE_C) # INPUT WEIGHTS
            GRAD["U_c"] += np.dot(self.OUTPUTS[:, STEP - 1, :].T, DE_C) # HIDDEN WEIGHTS
            GRAD["b_c"] += DE_C.sum(axis=0) # BIAS
            DH_NEXT = DHI * self.GATES["f"][:, STEP, :] # NEXT HIDDEN STATE
        for KEY in GRAD.keys(): # LOOP OVER GRADIENTS
            self.__PARAMETERS__.UPDATE_GRAD(KEY, GRAD[KEY]) # UPDATE GRADIENT
        return OUTPUT # RETURN OUTPUT

    def SHAPE(self, X_SHAPE):
        """SHAPE OF THE OUTPUT TENSOR
        
        PARAMETERS
        ----------
        X_SHAPE : TUPLE
            SHAPE OF THE INPUT TENSOR
        
        RETURNS
        -------
        TUPLE
            SHAPE OF THE OUTPUT TENSOR
        """
        if self.RETURN_SEQUENCES: # IF RETURN SEQUENCES
            return X_SHAPE[0], X_SHAPE[1], self.HIDDEN_DIM # RETURN 3D TENSOR
        else: # IF NOT RETURN SEQUENCES
            return X_SHAPE[0], self.HIDDEN_DIM # RETURN 2D TENSOR