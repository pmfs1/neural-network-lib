import numpy as np

class CONSTRAINT(object):
    """BASE CLASS FOR CONSTRAINTS"""

    def CLIP(self, P):
        """CLIP WEIGHTS

        PARAMETERS
        ----------
        P : ARRAY
            WEIGHTS

        RETURNS
        -------
        RETURN CLIPPED WEIGHTS
        """
        return P  # RETURN UNCHANGED WEIGHTS


class MAX_NORM(object):
    """MAX NORM CONSTRAINT"""

    def __init__(self, M=2, AXIS=0):
        """INITIALIZE MAX NORM CONSTRAINT

        PARAMETERS
        ----------
        M : FLOAT
            MAXIMUM NORM
        AXIS : INT
            AXIS TO COMPUTE NORM ALONG
        """
        self.AXIS = AXIS  # AXIS TO COMPUTE NORM ALONG
        self.M = M  # MAXIMUM NORM

    def CLIP(self, P):
        """CLIP WEIGHTS

        PARAMETERS
        ----------
        P : ARRAY
            WEIGHTS

        RETURNS
        -------
        RETURN CLIPPED WEIGHTS
        """
        NORMS = np.sqrt(np.sum(P ** 2, axis=self.AXIS))  # COMPUTE NORMS
        DESIRED = np.clip(NORMS, 0, self.M)  # CLIP NORMS
        P = P * (DESIRED / (10e-8 + NORMS))  # SCALE WEIGHTS
        return P  # RETURN CLIPPED WEIGHTS


class NON_NEG(object):
    """NON NEGATIVITY CONSTRAINT"""

    def CLIP(self, P):
        """CLIP WEIGHTS

        PARAMETERS
        ----------
        P : ARRAY
            WEIGHTS

        RETURNS
        -------
        RETURN CLIPPED WEIGHTS
        """
        P[P < 0.0] = 0.0  # CLIP WEIGHTS
        return P  # RETURN CLIPPED WEIGHTS


class SMALL_NORM(object):
    """SMALL NORM CONSTRAINT"""

    def CLIP(self, P):
        """CLIP WEIGHTS

        PARAMETERS
        ----------
        P : ARRAY
            WEIGHTS

        RETURNS
        -------
        RETURN CLIPPED WEIGHTS
        """
        return np.clip(P, -5, 5)  # CLIP WEIGHTS


class UNIT_NORM(CONSTRAINT):
    """UNIT NORM CONSTRAINT"""

    def __init__(self, AXIS=0):
        """INITIALIZE UNIT NORM CONSTRAINT"""
        self.AXIS = AXIS  # AXIS TO COMPUTE NORM ALONG

    def CLIP(self, P):
        """CLIP WEIGHTS

        PARAMETERS
        ----------
        P : ARRAY
            WEIGHTS

        RETURNS
        -------
        RETURN CLIPPED WEIGHTS
        """
        return P / (10e-8 + np.sqrt(np.sum(P ** 2, axis=self.AXIS)))  # RETURN CLIPPED WEIGHTS
