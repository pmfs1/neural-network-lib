from neural_network.initializations import *

class PARAMETER(object):
    """CLASS FOR STORING PARAMETERS OF A LAYER."""

    def __init__(self, INIT=GLOROT_UNIFORM, SCALE=0.5, BIAS=1.0, REGULARIZERS=None, CONSTRAINTS=None):
        """INITIALIZE PARAMETERS OF A LAYER.
        
        PARAMETERS:
        ----------
        INIT: FUNCTION
            FUNCTION FOR INITIALIZING WEIGHTS.
        SCALE: FLOAT
            SCALE OF THE WEIGHTS.
        BIAS: FLOAT
            INITIAL BIAS.
        REGULARIZERS: DICT
            DICTIONARY OF REGULARIZERS.
        CONSTRAINTS: DICT
            DICTIONARY OF CONSTRAINTS.
        
        RETURNS:
        --------
        PARAMETERS: OBJECT
            OBJECT CONTAINING PARAMETERS OF A LAYER.
        """
        if CONSTRAINTS is None:  # IF NO CONSTRAINTS ARE PROVIDED, SET TO EMPTY DICT
            self.CONSTRAINTS = { }  # EMPTY DICT
        else:  # OTHERWISE
            self.CONSTRAINTS = CONSTRAINTS  # SET TO CONSTRAINTS
        if REGULARIZERS is None:  # IF NO REGULARIZERS ARE PROVIDED, SET TO EMPTY DICT
            self.REGULARIZERS = { }  # EMPTY DICT
        else:  # OTHERWISE
            self.REGULARIZERS = REGULARIZERS  # SET TO REGULARIZERS
        self.INITIAL_BIAS = BIAS  # SET INITIAL BIAS
        self.SCALE = SCALE  # SET SCALE
        self.INIT = INIT  # SET INITIALIZATION FUNCTION
        self.__PARAMETERS__ = { }  # EMPTY DICT FOR PARAMETERS
        self.__GRADS__ = { }  # EMPTY DICT FOR GRADIENTS

    def SETUP_WEIGHTS(self, W_SHAPE, B_SHAPE=None):
        """SETUP WEIGHTS OF A LAYER.

        PARAMETERS:
        ----------
        W_SHAPE: TUPLE
            SHAPE OF WEIGHTS.
        B_SHAPE: TUPLE
            SHAPE OF BIAS.
        
        RETURNS:
        --------
        NONE
        """
        if "W" not in self.__PARAMETERS__:  # IF WEIGHTS ARE NOT IN PARAMETERS
            self.__PARAMETERS__["W"] = self.INIT(W_SHAPE)  # SET WEIGHTS TO INITIALIZED WEIGHTS
            if B_SHAPE is None:  # IF NO BIAS SHAPE IS PROVIDED
                self.__PARAMETERS__["b"] = np.full(W_SHAPE[1], self.INITIAL_BIAS)  # SET BIAS TO INITIAL BIAS
            else:  # OTHERWISE
                self.__PARAMETERS__["b"] = np.full(B_SHAPE, self.INITIAL_BIAS)  # SET BIAS TO INITIAL BIAS
        self.INIT_GRAD()  # INITIALIZE GRADIENTS

    def INIT_GRAD(self):
        """INITIALIZE GRADIENTS OF A LAYER.

        PARAMETERS:
        ----------
        NONE

        RETURNS:
        --------
        NONE
        """
        for KEY in self.__PARAMETERS__.keys():  # LOOP OVER KEYS IN PARAMETERS
            if KEY not in self.__GRADS__:  # IF KEY IS NOT IN GRADIENTS
                self.__GRADS__[KEY] = np.zeros_like(self.__PARAMETERS__[KEY])  # SET GRADIENT TO ZERO

    def UPDATE_GRAD(self, NAME, VALUE):
        """UPDATE GRADIENTS OF A LAYER.

        PARAMETERS:
        ----------
        NAME: STRING
            NAME OF THE PARAMETER.
        VALUE: NUMPY ARRAY
            VALUE OF THE GRADIENT.
        
        RETURNS:
        --------
        NONE
        """
        self.__GRADS__[NAME] = VALUE  # SET GRADIENT TO VALUE
        if NAME in self.REGULARIZERS:  # IF NAME IS IN REGULARIZERS
            self.__GRADS__[NAME] += self.REGULARIZERS[NAME](self.__PARAMETERS__[NAME])  # ADD REGULARIZATION TO GRADIENT

    def STEP(self, NAME, STEP):
        """UPDATE PARAMETERS OF A LAYER.

        PARAMETERS:
        ----------
        NAME: STRING
            NAME OF THE PARAMETER.
        STEP: NUMPY ARRAY
            VALUE OF THE STEP.
        
        RETURNS:
        --------
        NONE
        """
        self.__PARAMETERS__[NAME] += STEP  # UPDATE PARAMETER
        if NAME in self.CONSTRAINTS:  # IF NAME IS IN CONSTRAINTS
            self.__PARAMETERS__[NAME] = self.CONSTRAINTS[NAME].clip(self.__PARAMETERS__[NAME])  # CLIP PARAMETER

    @property
    def __NUMBER_OF_PARAMETERS__(self):
        """RETURN NUMBER OF PARAMETERS IN A LAYER.
        
        PARAMETERS:
        ----------
        NONE
        
        RETURNS:
        --------
        NUMBER_OF_PARAMETERS: INT
            NUMBER OF PARAMETERS IN A LAYER.
        """
        return sum([np.prod(self.__PARAMETERS__[x].shape) for x in self.__PARAMETERS__.keys()])  # RETURN NUMBER OF PARAMETERS

    @property
    def GRAD(self):
        """RETURN GRADIENTS OF PARAMETERS.

        PARAMETERS:
        ----------
        NONE

        RETURNS:
        --------
        GRADS: DICT
            DICTIONARY OF GRADIENTS.
        """
        return self.__GRADS__  # RETURN GRADIENTS

    def KEYS(self):
        """RETURN KEYS OF PARAMETERS.

        PARAMETERS:
        ----------
        NONE

        RETURNS:
        --------
        KEYS: LIST
            LIST OF KEYS.
        """
        return self.__PARAMETERS__.keys()  # RETURN KEYS

    def __getitem__(self, ITEM):
        """RETURN PARAMETER.

        PARAMETERS:
        ----------
        ITEM: STRING
            NAME OF THE PARAMETER.
        
        RETURNS:
        --------
        PARAMETER: NUMPY ARRAY
            PARAMETER.
        """
        if ITEM in self.__PARAMETERS__:  # IF ITEM IS IN PARAMETERS
            return self.__PARAMETERS__[ITEM]  # RETURN PARAMETER
        else:  # OTHERWISE
            raise ValueError("PARAMETER NOT FOUND.")  # RAISE ERROR

    def __setitem__(self, KEY, VALUE):
        """SET PARAMETER.

        PARAMETERS:
        ----------
        KEY: STRING
            NAME OF THE PARAMETER.
        VALUE: NUMPY ARRAY
            VALUE OF THE PARAMETER.
        
        RETURNS:
        --------
        NONE
        """
        self.__PARAMETERS__[KEY] = VALUE  # SET PARAMETER