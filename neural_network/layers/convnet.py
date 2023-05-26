import numpy as np
from neural_network.layers import LAYER, PARAM_MIXIN
from neural_network.parameters import PARAMETER

class CONVOLUTION(LAYER, PARAM_MIXIN):
    """2D CONVOLUTION LAYER.
    
    PARAMETERS
    ----------
    N_FILTERS : INT, DEFAULT 8
        NUMBER OF FILTERS
    FILTER_SHAPE : TUPLE(HEIGHT, WIDTH), DEFAULT (3, 3)
        SHAPE OF THE FILTER
    PADDING : TUPLE(HEIGHT, WIDTH), DEFAULT (0, 0)
        PADDING SIZE
    STRIDE : TUPLE(HEIGHT, WIDTH), DEFAULT (1, 1)
        STRIDE SIZE
    PARAMETERS : PARAMETERS, DEFAULT NONE
        PARAMETERS OF THE LAYER

    INPUT SHAPE
    -----------
    (N_IMAGES, N_CHANNELS, HEIGHT, WIDTH)

    OUTPUT SHAPE
    ------------
    (N_IMAGES, N_FILTERS, HEIGHT, WIDTH)

    METHODS
    -------
    SETUP(X_SHAPE)
        SETS UP PARAMETERS FOR THE LAYER
    FORWARD_PASS(X)
        RETURNS OUTPUT OF THE LAYER
    BACKWARD_PASS(DELTA)
        RETURNS DELTA FOR THE PREVIOUS LAYER
    SHAPE(X_SHAPE)
        RETURNS SHAPE OF THE OUTPUT
    """
    def __init__(self, N_FILTERS=8, FILTER_SHAPE=(3, 3), PADDING=(0, 0), STRIDE=(1, 1), PARAMETERS=PARAMETER()):
        """INITIALIZE CONVOLUTION LAYER.

        PARAMETERS
        ----------
        N_FILTERS : INT, DEFAULT 8
            NUMBER OF FILTERS
        FILTER_SHAPE : TUPLE(HEIGHT, WIDTH), DEFAULT (3, 3)
            SHAPE OF THE FILTER
        PADDING : TUPLE(HEIGHT, WIDTH), DEFAULT (0, 0)
            PADDING SIZE
        STRIDE : TUPLE(HEIGHT, WIDTH), DEFAULT (1, 1)
            STRIDE SIZE
        PARAMETERS : PARAMETERS, DEFAULT NONE
            PARAMETERS OF THE LAYER
        """
        self.PADDING = PADDING # SET PADDING
        self.__PARAMETERS__ = PARAMETERS # SET PARAMETERS
        self.STRIDE = STRIDE # SET STRIDE
        self.FILTER_SHAPE = FILTER_SHAPE # SET FILTER SHAPE
        self.N_FILTERS = N_FILTERS # SET NUMBER OF FILTERS
        self.__PARAMETERS__ = PARAMETERS # SET PARAMETERS

    def SETUP(self, X_SHAPE):
        """SETUP PARAMETERS FOR THE LAYER.

        PARAMETERS
        ----------
        X_SHAPE : TUPLE(INT, INT, INT, INT)
            SHAPE OF THE INPUT
        
        RETURNS
        -------
        NONE
        """
        N_CHANNELS, self.HEIGHT, self.WIDTH = X_SHAPE[1:] # GET HEIGHT AND WIDTH OF THE INPUT
        W_SHAPE = (self.N_FILTERS, N_CHANNELS) + self.FILTER_SHAPE # SET SHAPE OF THE WEIGHTS
        B_SHAPE = self.N_FILTERS # SET SHAPE OF THE BIASES
        self.__PARAMETERS__.SETUP_WEIGHTS(W_SHAPE, B_SHAPE) # SETUP WEIGHTS AND BIASES

    def FORWARD_PASS(self, X):
        """RETURNS OUTPUT OF THE LAYER.

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT TO THE LAYER
        
        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT OF THE LAYER
        """
        N_IMAGES, N_CHANNELS, HEIGHT, WIDTH = self.SHAPE(X.SHAPE) # GET SHAPE OF THE INPUT
        self.LAST_INPUT = X # SAVE INPUT FOR BACKWARD PASS
        self.COL = IMAGE_TO_COLUMN(X, self.FILTER_SHAPE, self.STRIDE, self.PADDING) # GET COLUMN FROM THE INPUT
        self.COL_W = self.__PARAMETERS__["W"].reshape(self.N_FILTERS, -1).T # GET COLUMN FROM THE WEIGHTS
        OUT = np.dot(self.COL, self.COL_W) + self.__PARAMETERS__["b"] # GET OUTPUT
        OUT = OUT.reshape(N_IMAGES, HEIGHT, WIDTH, -1).transpose(0, 3, 1, 2) # RESHAPE OUTPUT
        return OUT # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """RETURNS DELTA FOR THE PREVIOUS LAYER.

        PARAMETERS
        ----------
        DELTA : NUMPY ARRAY
            DELTA FROM THE NEXT LAYER
        
        RETURNS
        -------
        NUMPY ARRAY
            DELTA FOR THE PREVIOUS LAYER
        """
        DELTA = DELTA.transpose(0, 2, 3, 1).reshape(-1, self.N_FILTERS) # RESHAPE DELTA
        D_W = np.dot(self.COL.T, DELTA).transpose(1, 0).reshape(self.__PARAMETERS__["W"].shape) # GET DELTA FOR THE WEIGHTS
        D_B = np.sum(DELTA, axis=0) # GET DELTA FOR THE BIASES
        self.__PARAMETERS__.UPDATE_GRAD("b", D_B) # UPDATE GRADIENTS
        self.__PARAMETERS__.UPDATE_GRAD("W", D_W) # UPDATE GRADIENTS
        D_C = np.dot(DELTA, self.COL_W.T) # GET DELTA FOR THE INPUT
        return COLUMN_TO_IMAGE(D_C, self.LAST_INPUT.shape, self.FILTER_SHAPE, self.STRIDE, self.PADDING) # RETURN DELTA FOR THE PREVIOUS LAYER

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE OUTPUT.

        PARAMETERS
        ----------
        X_SHAPE : TUPLE(INT, INT, INT, INT)
            SHAPE OF THE INPUT
        
        RETURNS
        -------
        TUPLE(INT, INT, INT, INT)
            SHAPE OF THE OUTPUT
        """
        HEIGHT, WIDTH = CONVOLUTION_SHAPE(self.HEIGHT, self.WIDTH, self.FILTER_SHAPE, self.STRIDE, self.PADDING) # GET SHAPE OF THE OUTPUT
        return X_SHAPE[0], self.N_FILTERS, HEIGHT, WIDTH # RETURN SHAPE OF THE OUTPUT

class MAX_POOLING(LAYER):
    """MAX POOLING LAYER.

    OUTPUT SHAPE
    ------------
    (N_IMAGES, N_CHANNELS, HEIGHT, WIDTH)
    
    METHODS
    -------
    SETUP(X_SHAPE)
        SETS UP PARAMETERS FOR THE LAYER
    FORWARD_PASS(X)
        RETURNS OUTPUT OF THE LAYER
    BACKWARD_PASS(DELTA)
        RETURNS DELTA FOR THE PREVIOUS LAYER
    SHAPE(X_SHAPE)
        RETURNS SHAPE OF THE OUTPUT
    """
    def __init__(self, POOL_SHAPE=(2, 2), STRIDE=(1, 1), PADDING=(0, 0)):
        """INITIALIZE MAX POOLING LAYER.

        PARAMETERS
        ----------
        POOL_SHAPE : TUPLE(HEIGHT, WIDTH), DEFAULT (2, 2)
            SHAPE OF THE POOLING
        STRIDE : TUPLE(HEIGHT, WIDTH), DEFAULT (1, 1)
            STRIDE SIZE
        PADDING : TUPLE(HEIGHT, WIDTH), DEFAULT (0, 0)
            PADDING SIZE
        
        RETURNS
        -------
        NONE
        """
        self.POOL_SHAPE = POOL_SHAPE # SET POOL SHAPE
        self.STRIDE = STRIDE # SET STRIDE
        self.PADDING = PADDING # SET PADDING

    def FORWARD_PASS(self, X):
        """RETURNS OUTPUT OF THE LAYER.

        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT TO THE LAYER
        
        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT OF THE LAYER
        """
        self.LAST_INPUT = X # SAVE INPUT FOR BACKWARD PASS
        OUT_HEIGHT, OUT_WIDTH = POOLING_SHAPE(self.POOL_SHAPE, X.shape, self.STRIDE) # GET SHAPE OF THE OUTPUT
        N_IMAGES, N_CHANNELS, _, _ = X.shape # GET SHAPE OF THE INPUT
        COL = IMAGE_TO_COLUMN(X, self.POOL_SHAPE, self.STRIDE, self.PADDING) # GET COLUMN FROM THE INPUT
        COL = COL.reshape(-1, self.POOL_SHAPE[0] * self.POOL_SHAPE[1]) # RESHAPE COLUMN
        OUT = np.max(COL, axis=1) # GET MAX
        self.ARG_MAX = np.argmax(COL, axis=1) # SAVE ARGMAX FOR BACKWARD PASS
        return OUT.reshape(N_IMAGES, OUT_HEIGHT, OUT_WIDTH, N_CHANNELS).transpose(0, 3, 1, 2) # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """RETURNS DELTA FOR THE PREVIOUS LAYER.

        PARAMETERS
        ----------
        DELTA : NUMPY ARRAY
            DELTA FROM THE NEXT LAYER
        
        RETURNS
        -------
        NUMPY ARRAY
            DELTA FOR THE PREVIOUS LAYER
        """
        DELTA = DELTA.transpose(0, 2, 3, 1) # RESHAPE DELTA
        POOL_SIZE = self.POOL_SHAPE[0] * self.POOL_SHAPE[1] # GET POOL SIZE
        Y_MAX = np.zeros((DELTA.size, POOL_SIZE)) # CREATE ARRAY FOR DELTA
        Y_MAX[np.arange(self.ARG_MAX.size), self.ARG_MAX.flatten()] = DELTA.flatten() # GET DELTA
        Y_MAX = Y_MAX.reshape(DELTA.shape + (POOL_SIZE,)) # RESHAPE DELTA
        DCOL = Y_MAX.reshape(Y_MAX.shape[0] * Y_MAX.shape[1] * Y_MAX.shape[2], -1) # RESHAPE DELTA
        return COLUMN_TO_IMAGE(DCOL, self.LAST_INPUT.shape, self.POOL_SHAPE, self.STRIDE, self.PADDING) # RETURN DELTA FOR THE PREVIOUS LAYER

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE OUTPUT.

        PARAMETERS
        ----------
        X_SHAPE : TUPLE(INT, INT, INT, INT)
            SHAPE OF THE INPUT

        RETURNS
        -------
        TUPLE(INT, INT, INT, INT)
            SHAPE OF THE OUTPUT
        """
        HEIGHT, WIDTH = CONVOLUTION_SHAPE(X_SHAPE[2], X_SHAPE[3], self.POOL_SHAPE, self.STRIDE, self.PADDING) # GET SHAPE OF THE OUTPUT
        return X_SHAPE[0], X_SHAPE[1], HEIGHT, WIDTH # RETURN SHAPE OF THE OUTPUT

class FLATTEN(LAYER):
    """FLATTEN LAYER.

    OUTPUT SHAPE
    ------------
    (N_IMAGES, N_CHANNELS * HEIGHT * WIDTH)

    METHODS
    -------
    SETUP(X_SHAPE)
        SETS UP PARAMETERS FOR THE LAYER
    FORWARD_PASS(X)
        RETURNS OUTPUT OF THE LAYER
    BACKWARD_PASS(DELTA)
        RETURNS DELTA FOR THE PREVIOUS LAYER
    SHAPE(X_SHAPE)
        RETURNS SHAPE OF THE OUTPUT
    """
    def FORWARD_PASS(self, X):
        """RETURNS OUTPUT OF THE LAYER.
        
        PARAMETERS
        ----------
        X : NUMPY ARRAY
            INPUT TO THE LAYER
        
        RETURNS
        -------
        NUMPY ARRAY
            OUTPUT OF THE LAYER
        """
        self.LAST_INPUT_SHAPE = X.shape # SAVE SHAPE OF THE INPUT
        return X.reshape((X.shape[0], -1)) # RETURN OUTPUT

    def BACKWARD_PASS(self, DELTA):
        """RETURNS DELTA FOR THE PREVIOUS LAYER.

        PARAMETERS
        ----------
        DELTA : NUMPY ARRAY
            DELTA FROM THE NEXT LAYER
        
        RETURNS
        -------
        NUMPY ARRAY
            DELTA FOR THE PREVIOUS LAYER
        """
        return DELTA.reshape(self.LAST_INPUT_SHAPE) # RETURN DELTA FOR THE PREVIOUS LAYER

    def SHAPE(self, X_SHAPE):
        """RETURNS SHAPE OF THE OUTPUT.

        PARAMETERS
        ----------
        X_SHAPE : TUPLE(INT, INT, INT, INT)
            SHAPE OF THE INPUT

        RETURNS
        -------
        TUPLE(INT, INT)
            SHAPE OF THE OUTPUT
        """
        return X_SHAPE[0], np.prod(X_SHAPE[1:]) # RETURN SHAPE OF THE OUTPUT

##############################################################################################################

def IMAGE_TO_COLUMN(IMAGES, FILTER_SHAPE, STRIDE, PADDING):
    """RETURNS COLUMN FROM THE INPUT.

    PARAMETERS
    ----------
    IMAGES : NUMPY ARRAY
        INPUT TO THE LAYER
    FILTER_SHAPE : TUPLE(INT, INT)
        SHAPE OF THE FILTER
    STRIDE : TUPLE(INT, INT)
        STRIDE OF THE FILTER
    PADDING : INT
        PADDING OF THE INPUT
    
    RETURNS
    -------
    NUMPY ARRAY
        COLUMN FROM THE INPUT
    """
    N_IMAGES, N_CHANNELS, HEIGHT, WIDTH = IMAGES.shape # GET SHAPE OF THE INPUT
    F_HEIGHT, F_WIDTH = FILTER_SHAPE # GET SHAPE OF THE FILTER
    OUT_HEIGHT, OUT_WIDTH = CONVOLUTION_SHAPE(HEIGHT, WIDTH, (F_HEIGHT, F_WIDTH), STRIDE, PADDING) # GET SHAPE OF THE OUTPUT
    IMAGES = np.pad(IMAGES, ((0, 0), (0, 0), PADDING, PADDING), mode="constant") # PAD INPUT
    COL = np.zeros((N_IMAGES, N_CHANNELS, F_HEIGHT, F_WIDTH, OUT_HEIGHT, OUT_WIDTH)) # CREATE ARRAY FOR COLUMN
    for Y in range(F_HEIGHT): # LOOP OVER HEIGHT
        Y_BOUND = Y + STRIDE[0] * OUT_HEIGHT # GET BOUND
        for X in range(F_WIDTH): # LOOP OVER WIDTH
            X_BOUND = X + STRIDE[1] * OUT_WIDTH # GET BOUND
            COL[:, :, Y, X, :, :] = IMAGES[:, :, Y: Y_BOUND: STRIDE[0], X: X_BOUND: STRIDE[1]] # GET COLUMN
    COL = COL.transpose(0, 4, 5, 1, 2, 3).reshape(N_IMAGES * OUT_HEIGHT * OUT_WIDTH, -1) # RESHAPE COLUMN
    return COL # RETURN COLUMN

def COLUMN_TO_IMAGE(COLUMNS, IMAGES_SHAPE, FILTER_SHAPE, STRIDE, PADDING):
    """RETURNS IMAGE FROM THE COLUMN.

    PARAMETERS
    ----------
    COLUMNS : NUMPY ARRAY
        COLUMN FROM THE INPUT
    IMAGES_SHAPE : TUPLE(INT, INT, INT, INT)
        SHAPE OF THE INPUT
    FILTER_SHAPE : TUPLE(INT, INT)
        SHAPE OF THE FILTER
    STRIDE : TUPLE(INT, INT)
        STRIDE OF THE FILTER
    PADDING : INT
        PADDING OF THE INPUT

    RETURNS
    -------
    NUMPY ARRAY
        IMAGE FROM THE COLUMN
    """
    N_IMAGES, N_CHANNELS, HEIGHT, WIDTH = IMAGES_SHAPE # GET SHAPE OF THE INPUT
    F_HEIGHT, F_WIDTH = FILTER_SHAPE # GET SHAPE OF THE FILTER
    OUT_HEIGHT, OUT_WIDTH = CONVOLUTION_SHAPE(HEIGHT, WIDTH, (F_HEIGHT, F_WIDTH), STRIDE, PADDING) # GET SHAPE OF THE OUTPUT
    COLUMNS = COLUMNS.reshape(N_IMAGES, OUT_HEIGHT, OUT_WIDTH, N_CHANNELS, F_HEIGHT, F_WIDTH).transpose(
        0, 3, 4, 5, 1, 2
    ) # RESHAPE COLUMN
    IMG_H = HEIGHT + 2 * PADDING[0] + STRIDE[0] - 1 # GET THE HEIGHT OF THE IMAGE
    IMG_W = WIDTH + 2 * PADDING[1] + STRIDE[1] - 1 # GET THE WIDTH OF THE IMAGE
    IMG = np.zeros((N_IMAGES, N_CHANNELS, IMG_H, IMG_W)) # CREATE ARRAY FOR IMAGE
    for Y in range(F_HEIGHT): # LOOP OVER HEIGHT
        Y_BOUND = Y + STRIDE[0] * OUT_HEIGHT # GET BOUND
        for X in range(F_WIDTH): # LOOP OVER WIDTH
            X_BOUND = X + STRIDE[1] * OUT_WIDTH # GET BOUND
            IMG[:, :, Y: Y_BOUND: STRIDE[0], X: X_BOUND: STRIDE[1]] += COLUMNS[:, :, Y, X, :, :] # GET IMAGE
    return IMG[:, :, PADDING[0]: HEIGHT + PADDING[0], PADDING[1]: WIDTH + PADDING[1]] # RETURN IMAGE

def CONVOLUTION_SHAPE(IMG_HEIGHT, IMG_WIDTH, FILTER_SHAPE, STRIDE, PADDING):
    """CALCULATE OUTPUT SHAPE FOR CONVOLUTION LAYER.
    
    PARAMETERS
    ----------
    IMG_HEIGHT : INT
        HEIGHT OF THE INPUT
    IMG_WIDTH : INT
        WIDTH OF THE INPUT
    FILTER_SHAPE : TUPLE(INT, INT)
        SHAPE OF THE FILTER
    STRIDE : TUPLE(INT, INT)
        STRIDE OF THE FILTER
    PADDING : INT
        PADDING OF THE INPUT
    
    RETURNS
    -------
    TUPLE(INT, INT)
        OUTPUT SHAPE
    """
    HEIGHT = (IMG_HEIGHT + 2 * PADDING[0] - FILTER_SHAPE[0]) / float(STRIDE[0]) + 1 # CALCULATE HEIGHT
    WIDTH = (IMG_WIDTH + 2 * PADDING[1] - FILTER_SHAPE[1]) / float(STRIDE[1]) + 1 # CALCULATE WIDTH
    assert HEIGHT % 1 == 0 # CHECK IF HEIGHT IS INTEGER
    assert WIDTH % 1 == 0 # CHECK IF WIDTH IS INTEGER
    return int(HEIGHT), int(WIDTH) # RETURN OUTPUT SHAPE

def POOLING_SHAPE(POOL_SHAPE, IMAGE_SHAPE, STRIDE):
    """CALCULATE OUTPUT SHAPE FOR POOLING LAYER.

    PARAMETERS
    ----------
    POOL_SHAPE : TUPLE(INT, INT)
        SHAPE OF THE POOL
    IMAGE_SHAPE : TUPLE(INT, INT, INT, INT)
        SHAPE OF THE INPUT
    STRIDE : TUPLE(INT, INT)
        STRIDE OF THE POOL
    
    RETURNS
    -------
    TUPLE(INT, INT)
        OUTPUT SHAPE
    """
    N_IMAGES, N_CHANNELS, HEIGHT, WIDTH = IMAGE_SHAPE # GET SHAPE OF THE INPUT
    HEIGHT = (HEIGHT - POOL_SHAPE[0]) / float(STRIDE[0]) + 1 # CALCULATE HEIGHT
    WIDTH = (WIDTH - POOL_SHAPE[1]) / float(STRIDE[1]) + 1 # CALCULATE WIDTH
    assert HEIGHT % 1 == 0 # CHECK IF HEIGHT IS INTEGER
    assert WIDTH % 1 == 0 # CHECK IF WIDTH IS INTEGER
    return int(HEIGHT), int(WIDTH) # RETURN OUTPUT SHAPE