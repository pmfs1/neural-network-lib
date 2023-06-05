import numpy as np

def SIGMOID(Z):
    """SIGMOID ACTIVATION FUNCTION

    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION

    RETURNS
    -------
    RETURN SIGMOID ACTIVATION
    """
    return 1.0 / (1.0 + np.exp(-Z))  # RETURN SIGMOID ACTIVATION

def SOFTMAX(Z):
    """SOFTMAX ACTIVATION FUNCTION

    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION
    
    RETURNS
    -------
    RETURN SOFTMAX ACTIVATION
    """
    E = np.exp(Z - np.amax(Z, axis=1, keepdims=True))  # COMPUTE EXPONENTIALS
    return E / np.sum(E, axis=1, keepdims=True)  # RETURN SOFTMAX ACTIVATION

def LINEAR(Z):
    """LINEAR ACTIVATION FUNCTION
    
    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION
    
    RETURNS
    -------
    RETURN LINEAR ACTIVATION
    """
    return Z  # RETURN LINEAR ACTIVATION

def SOFT_PLUS(Z):
    """SOFT PLUS ACTIVATION FUNCTION

    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION

    RETURNS
    -------
    RETURN SOFT PLUS ACTIVATION
    """
    return np.logaddexp(0.0, Z)  # RETURN SOFT PLUS ACTIVATION

def SOFT_SIGN(Z):
    """SOFT SIGN ACTIVATION FUNCTION
    
    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION
        
    RETURNS
    -------
    RETURN SOFT SIGN ACTIVATION
    """
    return Z / (1 + np.abs(Z))  # RETURN SOFT SIGN ACTIVATION

def TANH(Z):
    """TANH ACTIVATION FUNCTION
    
    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION
        
    RETURNS
    -------
    RETURN TANH ACTIVATION
    """
    return np.tanh(Z)  # RETURN TANH ACTIVATION

def RELU(Z):
    """RELU ACTIVATION FUNCTION

    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION

    RETURNS
    -------
    RETURN RELU ACTIVATION
    """
    return np.maximum(0, Z)  # RETURN RELU ACTIVATION

def LEAKY_RELU(Z, A=0.01):
    """LEAKY RELU ACTIVATION FUNCTION

    PARAMETERS
    ----------
    Z : ARRAY
        LINEAR TRANSFORMATION
    A : FLOAT, OPTIONAL
        LEAKY RELU PARAMETER, BY DEFAULT 0.01
    
    RETURNS
    -------
    RETURN LEAKY RELU ACTIVATION
    """
    return np.maximum(Z * A, Z)  # RETURN LEAKY RELU ACTIVATION