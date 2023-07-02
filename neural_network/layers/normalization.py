import numpy as np

from neural_network.layers import LAYER, PARAM_MIXIN, PHASE_MIXIN
from neural_network.parameters import PARAMETER

class BATCH_NORMALIZATION(LAYER, PARAM_MIXIN, PHASE_MIXIN):
    """BATCH NORMALIZATION LAYER

    ATTRIBUTES
    ----------
    MOMENTUM : FLOAT
        MOMENTUM FOR THE MOVING AVERAGE
    EPS : FLOAT
        EPSILON FOR NUMERICAL STABILITY
    EMA_MEAN : FLOAT
        EXPONENTIAL MOVING AVERAGE OF THE MEAN
    EMA_VAR : FLOAT
        EXPONENTIAL MOVING AVERAGE OF THE VARIANCE

    METHODS
    -------
    SETUP(X_SHAPE)
        SETS UP THE WEIGHTS OF THE LAYER
    FORWARD_PASS(X)
        PERFORMS A FORWARD PASS THROUGH THE LAYER
    BACKWARD_PASS(X)
        PERFORMS A BACKWARD PASS THROUGH THE LAYER
    SHAPE()
        RETURNS THE SHAPE OF THE LAYER
    """

    def __init__(self, MOMENTUM=0.9, EPS=1e-5, PARAMETERS=PARAMETER()):
        """INITIALIZE BATCH NORMALIZATION LAYER
        
        PARAMETERS
        ----------
        MOMENTUM : FLOAT
            MOMENTUM FOR THE MOVING AVERAGE
        EPS : FLOAT
            EPSILON FOR NUMERICAL STABILITY
        PARAMETERS : PARAMETER
            PARAMETERS OF THE LAYER
        EMA_MEAN : FLOAT
            EXPONENTIAL MOVING AVERAGE OF THE MEAN
        EMA_VAR : FLOAT
            EXPONENTIAL MOVING AVERAGE OF THE VARIANCE
        
        RETURNS
        -------
        NONE
        """
        super().__init__()  # INITIALIZE LAYER
        self.__PARAMETERS__ = PARAMETERS  # INITIALIZE PARAMETERS
        self.MOMENTUM = MOMENTUM  # INITIALIZE MOMENTUM
        self.EPS = EPS  # INITIALIZE EPSILON
        self.EMA_MEAN = None  # INITIALIZE EXPONENTIAL MOVING AVERAGE OF THE MEAN
        self.EMA_VAR = None  # INITIALIZE EXPONENTIAL MOVING AVERAGE OF THE VARIANCE

    def SETUP(self, X_SHAPE):
        """SETUP BATCH NORMALIZATION LAYER

        PARAMETERS
        ----------
        X_SHAPE : TUPLE
            SHAPE OF THE INPUT

        RETURNS
        -------
        NONE
        """
        self.__PARAMETERS__.SETUP_WEIGHTS((1, X_SHAPE[1]))  # INITIALIZE WEIGHTS

    def __FORWARD_PASS__(self, X):
        """PERFORMS A FORWARD PASS THROUGH THE LAYER

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT TO THE LAYER
        
        RETURNS
        -------
        OUT : NUMPY ARRAY
            OUTPUT OF THE LAYER
        """
        GAMMA = self.__PARAMETERS__["W"]  # GET GAMMA
        BETA = self.__PARAMETERS__["b"]  # GET BETA
        if self.IS_TESTING:  # IF TESTING
            assert self.EMA_MEAN is not None and self.EMA_VAR is not None, "EMA_MEAN AND EMA_VAR MUST BE SET FOR TESTING"  # ENSURE EMA_MEAN AND EMA_VAR ARE SET
            MU = self.EMA_MEAN  # GET MEAN
            XMU = X - MU  # GET X - MEAN
            VAR = self.EMA_VAR  # GET VARIANCE
            SQRTVAR = np.sqrt(VAR + self.EPS)  # GET SQRT(VARIANCE + EPSILON)
            IVAR = 1.0 / SQRTVAR  # GET INVERSE SQRT(VARIANCE + EPSILON)
            XHAT = XMU * IVAR  # GET NORMALIZED INPUT
            GAMMAX = GAMMA * XHAT  # GET GAMMA * NORMALIZED INPUT
            return GAMMAX + BETA  # RETURN GAMMA * NORMALIZED INPUT + BETA
        N, D = X.shape  # GET NUMBER OF SAMPLES AND FEATURES
        MU = 1.0 / N * np.sum(X, axis=0)  # GET MEAN
        XMU = X - MU  # GET X - MEAN
        SQ = XMU ** 2  # GET (X - MEAN) ** 2
        VAR = 1.0 / N * np.sum(SQ, axis=0)  # GET VARIANCE
        SQRTVAR = np.sqrt(VAR + self.EPS)  # GET SQRT(VARIANCE + EPSILON)
        IVAR = 1.0 / SQRTVAR  # GET INVERSE SQRT(VARIANCE + EPSILON)
        XHAT = XMU * IVAR  # GET NORMALIZED INPUT
        GAMMAX = GAMMA * XHAT  # GET GAMMA * NORMALIZED INPUT
        OUT = GAMMAX + BETA  # GET GAMMA * NORMALIZED INPUT + BETA
        if self.EMA_MEAN is None or self.EMA_VAR is None:  # IF EMA_MEAN OR EMA_VAR ARE NOT SET
            self.EMA_MEAN = MU  # SET EMA_MEAN
            self.EMA_VAR = VAR  # SET EMA_VAR
        else:  # IF EMA_MEAN AND EMA_VAR ARE SET
            self.EMA_MEAN = self.MOMENTUM * self.EMA_MEAN + (1 - self.MOMENTUM) * MU  # UPDATE EMA_MEAN
            self.EMA_VAR = self.MOMENTUM * self.EMA_VAR + (1 - self.MOMENTUM) * VAR  # UPDATE EMA_VAR
        self.CACHE = (XHAT, GAMMA, XMU, IVAR, SQRTVAR, VAR)  # STORE VARIABLES IN CACHE
        return OUT  # RETURN OUTPUT

    def FORWARD_PASS(self, X):
        """PERFORMS A FORWARD PASS THROUGH THE LAYER

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT TO THE LAYER
        
        RETURNS
        -------
        OUT : NUMPY ARRAY
            OUTPUT OF THE LAYER
        """
        if len(X.shape) == 2:  # IF INPUT IS A REGULAR LAYER
            return self.__FORWARD_PASS__(X)  # PERFORM FORWARD PASS
        elif len(X.shape) == 4:  # IF INPUT IS A CONVOLUTION LAYER
            N, C, H, W = X.shape  # GET NUMBER OF SAMPLES, CHANNELS, HEIGHT, AND WIDTH
            X_FLAT = X.transpose(0, 2, 3, 1).reshape(-1, C)  # RESHAPE INPUT
            OUT_FLAT = self.__FORWARD_PASS__(X_FLAT)  # PERFORM FORWARD PASS
            return OUT_FLAT.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # RESHAPE OUTPUT
        else:  # IF INPUT IS NEITHER A REGULAR LAYER OR A CONVOLUTION LAYER
            raise NotImplementedError("INPUT SHAPE NOT SUPPORTED")  # RAISE ERROR

    def __BACKWARD_PASS__(self, DELTA):
        """PERFORMS A BACKWARD PASS THROUGH THE LAYER

        PARAMETERS
        ----------
        DELTA : NUMPY ARRAY
            DELTA FROM THE NEXT LAYER
        
        RETURNS
        -------
        DELTA : NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER
        """
        XHAT, GAMMA, XMU, IVAR, SQRTVAR, VAR = self.CACHE  # GET VARIABLES FROM CACHE
        N, D = DELTA.shape  # GET NUMBER OF SAMPLES AND FEATURES
        D_BETA = np.sum(DELTA, axis=0)  # GET DERIVATIVE OF BETA
        D_GAMMAX = DELTA  # GET DERIVATIVE OF GAMMA * NORMALIZED INPUT
        D_GAMMA = np.sum(D_GAMMAX * XHAT, axis=0)  # GET DERIVATIVE OF GAMMA
        D_XHAT = D_GAMMAX * GAMMA  # GET DERIVATIVE OF NORMALIZED INPUT
        D_IVAR = np.sum(D_XHAT * XMU, axis=0)  # GET DERIVATIVE OF INVERSE SQRT(VARIANCE + EPSILON)
        D_XMU_1 = D_XHAT * IVAR  # GET DERIVATIVE OF X - MEAN
        D_SQRTVAR = -1.0 / (SQRTVAR ** 2) * D_IVAR  # GET DERIVATIVE OF SQRT(VARIANCE + EPSILON)
        D_VAR = 0.5 * 1.0 / np.sqrt(VAR + self.EPS) * D_SQRTVAR  # GET DERIVATIVE OF VARIANCE
        DSQ = 1.0 / N * np.ones((N, D)) * D_VAR  # GET DERIVATIVE OF (X - MEAN) ** 2
        D_XMU_2 = 2 * XMU * DSQ  # GET DERIVATIVE OF X - MEAN
        DX_1 = D_XMU_1 + D_XMU_2  # GET DERIVATIVE OF INPUT
        D_MU = -1 * np.sum(D_XMU_1 + D_XMU_2, axis=0)  # GET DERIVATIVE OF MEAN
        DX_2 = 1.0 / N * np.ones((N, D)) * D_MU  # GET DERIVATIVE OF INPUT
        DX = DX_1 + DX_2  # GET DERIVATIVE OF INPUT
        self.__PARAMETERS__.UPDATE_GRAD("W", D_GAMMA)  # UPDATE GRADIENT OF GAMMA
        self.__PARAMETERS__.UPDATE_GRAD("b", D_BETA)  # UPDATE GRADIENT OF BETA
        return DX  # RETURN DELTA TO THE PREVIOUS LAYER

    def BACKWARD_PASS(self, X):
        """PERFORMS A BACKWARD PASS THROUGH THE LAYER

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT TO THE LAYER
        
        RETURNS 
        -------
        DELTA : NUMPY ARRAY
            DELTA TO THE PREVIOUS LAYER
        """
        if len(X.shape) == 2:  # IF INPUT IS A REGULAR LAYER
            return self.__BACKWARD_PASS__(X)  # PERFORM BACKWARD PASS
        elif len(X.shape) == 4:  # IF INPUT IS A CONVOLUTION LAYER
            N, C, H, W = X.shape  # GET NUMBER OF SAMPLES, CHANNELS, HEIGHT, AND WIDTH
            X_FLAT = X.transpose(0, 2, 3, 1).reshape(-1, C)  # RESHAPE INPUT
            OUT_FLAT = self.__BACKWARD_PASS__(X_FLAT)  # PERFORM BACKWARD PASS
            return OUT_FLAT.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # RESHAPE OUTPUT
        else:  # IF INPUT IS NEITHER A REGULAR LAYER OR A CONVOLUTION LAYER
            raise NotImplementedError("INPUT SHAPE NOT SUPPORTED")  # RAISE ERROR

    def SHAPE(self, X_SHAPE):
        """RETURNS THE SHAPE OF THE OUTPUT GIVEN AN INPUT SHAPE

        PARAMETERS
        ----------
        X_SHAPE : TUPLE
            SHAPE OF THE INPUT

        RETURNS
        -------
        OUT_SHAPE : TUPLE
            SHAPE OF THE OUTPUT
        """
        return X_SHAPE  # RETURN INPUT SHAPE