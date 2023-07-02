"""
LINEAR REGRESSION IS THE MOST BASIC TYPE OF REGRESSION COMMONLY USED FOR
PREDICTIVE ANALYSIS. THE IDEA IS PRETTY SIMPLE: WE HAVE A DATASET AND WE HAVE
FEATURES ASSOCIATED WITH IT. FEATURES SHOULD BE CHOSEN VERY CAUTIOUSLY
AS THEY DETERMINE HOW MUCH OUR MODEL WILL BE ABLE TO MAKE FUTURE PREDICTIONS.
WE TRY TO SET THE WEIGHT OF THESE FEATURES, OVER MANY ITERATIONS, SO THAT THEY BEST
FIT OUR DATASET. IN THIS PARTICULAR CODE, I HAD USED A CSGO DATASET (ADR VS
RATING). WE TRY TO BEST FIT A LINE THROUGH DATASET AND ESTIMATE THE PARAMETERS.
"""
import numpy as np

def RUN_STEEP_GRADIENT_DESCENT(DATA_X, DATA_Y, LEN_DATA, ALPHA, THETA):
    """
    RUN STEEP GRADIENT DESCENT ON THE DATASET.

    PARAMS
    ------
    DATA_X: THE DATA MATRIX
    DATA_Y: THE LABELS
    LEN_DATA: THE LENGTH OF THE DATA
    ALPHA: THE LEARNING RATE
    THETA: THE PARAMETERS

    RETURNS
    -------
    RETURNS THE UPDATED PARAMETERS
    """
    N = LEN_DATA # NUMBER OF DATA-POINTS
    PROD = np.dot(THETA, DATA_X.transpose()) # DOT PRODUCT OF THETA AND DATA_X
    PROD -= DATA_Y.transpose() # SUBTRACT DATA_Y FROM THE ABOVE DOT PRODUCT
    SUM_GRAD = np.dot(PROD, DATA_X) # DOT PRODUCT OF THE ABOVE RESULT AND DATA_X
    THETA = THETA - (ALPHA / N) * SUM_GRAD # UPDATE THETA
    return THETA # RETURN UPDATED THETA

def SUM_OF_SQUARE_ERROR(DATA_X, DATA_Y, LEN_DATA, THETA):
    """
    COMPUTE THE SUM OF SQUARE ERROR.

    PARAMS
    ------
    DATA_X: THE DATA MATRIX
    DATA_Y: THE LABELS
    LEN_DATA: THE LENGTH OF THE DATA
    THETA: THE PARAMETERS

    RETURNS
    -------
    RETURNS THE SUM OF SQUARE ERROR
    """
    PROD = np.dot(THETA, DATA_X.transpose()) # DOT PRODUCT OF THETA AND DATA_X
    PROD -= DATA_Y.transpose() # SUBTRACT DATA_Y FROM THE ABOVE DOT PRODUCT
    SUM_ELEM = np.sum(np.square(PROD)) # SUM OF SQUARE OF EACH ELEMENT
    ERROR = SUM_ELEM / (2 * LEN_DATA) # ERROR COMPUTED
    return ERROR # RETURN ERROR

def RUN_LINEAR_REGRESSION(DATA_X, DATA_Y, ITERATIONS = 100000, ALPHA = 0.0001550):
    """
    RUN LINEAR REGRESSION ON THE DATASET.

    PARAMS
    ------
    DATA_X: THE DATA MATRIX
    DATA_Y: THE LABELS
    ITERATIONS: THE NUMBER OF ITERATIONS
    ALPHA: THE LEARNING RATE
    
    RETURNS
    -------
    RETURNS THE PARAMETERS
    """
    NO_FEATURES = DATA_X.shape[1] # NUMBER OF FEATURES
    LEN_DATA = DATA_X.shape[0] - 1 # LENGTH OF THE DATA
    THETA = np.zeros((1, NO_FEATURES)) # INITIALIZE THETA TO ZERO
    for ITERATION in range(0, ITERATIONS): # ITERATE OVER THE DATASET
        THETA = RUN_STEEP_GRADIENT_DESCENT(DATA_X, DATA_Y, LEN_DATA, ALPHA, THETA) # UPDATE THETA
        ERROR = SUM_OF_SQUARE_ERROR(DATA_X, DATA_Y, LEN_DATA, THETA) # COMPUTE ERROR
        print(f"ITERATION: {ITERATION} ERROR: {ERROR}") # PRINT ERROR
    return THETA # RETURN THETA

def MEAN_ABSOLUTE_ERROR(PREDICTED_Y, ORIGINAL_Y):
    """
    COMPUTE THE MEAN ABSOLUTE ERROR.

    PARAMS
    ------
    PREDICTED_Y: THE PREDICTED LABELS
    ORIGINAL_Y: THE ORIGINAL LABELS
    
    RETURNS
    -------
    RETURNS THE MEAN ABSOLUTE ERROR
    """
    return sum(abs(Y - PREDICTED_Y[I]) for I, Y in enumerate(ORIGINAL_Y)) / len(ORIGINAL_Y) # RETURN MEAN ABSOLUTE ERROR