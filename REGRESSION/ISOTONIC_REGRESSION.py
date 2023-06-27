import numpy as np

# ISOTONIC_REGRESSION: IMPLEMENTATION OF ISOTONIC REGRESSION. IN STATISTICS AND NUMERICAL ANALYSIS, ISOTONIC REGRESSION OR MONOTONIC REGRESSION IS THE TECHNIQUE OF FITTING A FREE-FORM LINE TO A SEQUENCE OF OBSERVATIONS SUCH THAT THE FITTED LINE IS NON-DECREASING (OR NON-INCREASING) EVERYWHERE, AND LIES AS CLOSE TO THE OBSERVATIONS AS POSSIBLE.


class ISOTONIC_REGRESSION:
    """IMPLEMENTATION OF ISOTONIC REGRESSION. IN STATISTICS AND NUMERICAL ANALYSIS, ISOTONIC REGRESSION OR MONOTONIC REGRESSION IS THE TECHNIQUE OF FITTING A FREE-FORM LINE TO A SEQUENCE OF OBSERVATIONS SUCH THAT THE FITTED LINE IS NON-DECREASING (OR NON-INCREASING) EVERYWHERE, AND LIES AS CLOSE TO THE OBSERVATIONS AS POSSIBLE.

    ATTRIBUTES
    ----------
    X_MIN : FLOAT (DEFAULT: -NP.INF)
        IT'S THE MINIMUM VALUE OF X.
    X_MAX : FLOAT (DEFAULT: NP.INF)
        IT'S THE MAXIMUM VALUE OF X.
    W : NUMPY ARRAY (DEFAULT: NONE)
        IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ISOTONIC REGRESSION MODEL.
    J : FLOAT (DEFAULT: NONE)
        IT'S THE LIST OF INDICES OF THE WEIGHTS.

    METHODS
    -------
    FIT(X, Y)
        FITS THE ISOTONIC REGRESSION MODEL TO THE TRAINING DATA.
    PREDICT(X)
        PREDICTS THE OUTPUT OF THE GIVEN INPUT DATA.
    """

    # INITIALIZES THE ISOTONIC REGRESSION MODEL
    def __init__(self, X_MIN=None, X_MAX=None):
        """INITIALIZES THE ISOTONIC REGRESSION MODEL

        PARAMETERS
        ----------
        X_MIN : FLOAT (DEFAULT: -NP.INF)
            IT'S THE MINIMUM VALUE OF X.
        X_MAX : FLOAT (DEFAULT: NP.INF)
            IT'S THE MAXIMUM VALUE OF X.

        ATTRIBUTES
        ----------
        X_MIN : FLOAT (DEFAULT: -NP.INF)
            IT'S THE MINIMUM VALUE OF X.
        X_MAX : FLOAT (DEFAULT: NP.INF)
            IT'S THE MAXIMUM VALUE OF X.
        W : NUMPY ARRAY (DEFAULT: NONE)
            IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ISOTONIC REGRESSION MODEL.
        J : LIST (DEFAULT: NONE)
            IT'S THE LIST OF THE INDICES OF THE WEIGHTS.
        """
        self.X_MIN = X_MIN if X_MIN is not None else - \
            np.inf  # X_MIN: IT'S THE MINIMUM VALUE OF X
        # X_MAX: IT'S THE MAXIMUM VALUE OF X
        self.X_MAX = X_MAX if X_MAX is not None else np.inf
        # W: IT'S THE PARAMETER THAT CORRESPONDS TO THE WEIGHTS OF THE ISOTONIC REGRESSION MODEL.
        self.W = None
        self.J = None  # J: IT'S THE LIST OF THE INDICES OF THE WEIGHTS

    # FIT(): FITS THE ISOTONIC REGRESSION MODEL TO THE TRAINING DATA
    def FIT(self, X, Y):
        """FITS THE ISOTONIC REGRESSION MODEL TO THE TRAINING DATA

        PARAMETERS
        ----------
        X : NUMPY ARRAY (N_SAMPLES, N_FEATURES)
            IT'S THE INPUT DATA.
        Y : NUMPY ARRAY (N_SAMPLES, 1)
            IT'S THE OUTPUT DATA.

        RETURNS
        -------
        NONE
        """
        self.W = np.ones(len(X))  # SETS THE WEIGHTS TO A VECTOR OF ONES
        if self.X_MIN is not None or self.X_MAX is not None:  # IF X_MIN OR X_MAX IS NOT NONE
            Y = np.copy(Y)  # COPIES THE OUTPUT DATA
            self.W = np.copy(self.W)  # COPIES THE WEIGHTS
            if self.X_MIN is not None:  # IF X_MIN IS NOT NONE
                Y[0] = self.X_MIN  # SETS THE FIRST ELEMENT OF Y TO X_MIN
                # SETS THE FIRST ELEMENT OF THE WEIGHTS TO 1e32
                self.W[0] = 1e32
            if self.X_MAX is not None:  # IF X_MAX IS NOT NONE
                Y[-1] = self.X_MAX  # SETS THE LAST ELEMENT OF Y TO X_MAX
                # SETS THE LAST ELEMENT OF THE WEIGHTS TO 1e32
                self.W[-1] = 1e32
        # J: IT'S THE LIST OF THE INDICES OF THE WEIGHTS; SETS THE LIST OF THE INDICES OF THE WEIGHTS TO A LIST OF LISTS
        self.J = [[_Y,] for _Y in range(len(Y))]
        IDX = 0  # IDX: IT'S THE CURRENT INDEX
        while IDX < len(self.J) - 1:  # WHILE IDX IS SMALLER THAN THE LENGTH OF J - 1
            AV0, AV1, AV2 = 0, 0, np.inf  # AV0: IT'S THE AVERAGE OF THE WEIGHTS OF THE CURRENT INDEX; AV1: IT'S THE AVERAGE OF THE WEIGHTS OF THE NEXT INDEX; AV2: IT'S THE AVERAGE OF THE WEIGHTS OF THE PREVIOUS INDEX; SETS THE AVERAGES TO 0 AND INFINITY
            # WHILE AV0 IS SMALLER THAN OR EQUAL TO AV1 AND IDX IS SMALLER THAN THE LENGTH OF J - 1
            while AV0 <= AV1 and IDX < len(self.J) - 1:
                IDX0 = self.J[IDX]  # SETS IDX0 TO THE CURRENT INDEX
                IDX1 = self.J[IDX + 1]  # SETS IDX1 TO THE NEXT INDEX
                # SETS AV0 TO THE AVERAGE OF THE WEIGHTS OF THE CURRENT INDEX
                AV0 = np.dot(self.W[IDX0], Y[IDX0]) / np.sum(self.W[IDX0])
                # SETS AV1 TO THE AVERAGE OF THE WEIGHTS OF THE NEXT INDEX
                AV1 = np.dot(self.W[IDX1], Y[IDX1]) / np.sum(self.W[IDX1])
                # INCREASES IDX BY 1 IF AV0 IS SMALLER THAN OR EQUAL TO AV1
                IDX += 1 if AV0 <= AV1 else 0
            if IDX == len(self.J) - 1:  # IF IDX IS EQUAL TO THE LENGTH OF J - 1
                break  # BREAKS THE LOOP
            # SETS A TO THE CURRENT INDEX AND REMOVES IT FROM J
            A = self.J.pop(IDX)
            # SETS B TO THE NEXT INDEX AND REMOVES IT FROM J
            B = self.J.pop(IDX)
            # INSERTS THE SUM OF A AND B AT THE CURRENT INDEX
            self.J.insert(IDX, A + B)
            while AV2 > AV0 and IDX > 0:  # WHILE AV2 IS BIGGER THAN AV0 AND IDX IS BIGGER THAN 0
                IDX0 = self.J[IDX]  # SETS IDX0 TO THE CURRENT INDEX
                IDX2 = self.J[IDX - 1]  # SETS IDX2 TO THE PREVIOUS INDEX
                # SETS AV0 TO THE AVERAGE OF THE WEIGHTS OF THE CURRENT INDEX
                AV0 = np.dot(self.W[IDX0], Y[IDX0]) / np.sum(self.W[IDX0])
                # SETS AV2 TO THE AVERAGE OF THE WEIGHTS OF THE PREVIOUS INDEX
                AV2 = np.dot(self.W[IDX2], Y[IDX2]) / np.sum(self.W[IDX2])
                if AV2 >= AV0:  # IF AV2 IS BIGGER THAN OR EQUAL TO AV0
                    # SETS A TO THE PREVIOUS INDEX AND REMOVES IT FROM J
                    A = self.J.pop(IDX - 1)
                    # SETS B TO THE CURRENT INDEX AND REMOVES IT FROM J
                    B = self.J.pop(IDX - 1)
                    # INSERTS THE SUM OF A AND B AT THE PREVIOUS INDEX
                    self.J.insert(IDX - 1, A + B)
                    IDX -= 1  # DECREASES IDX BY 1
        # SETS THE WEIGHTS TO THE SUM OF THE WEIGHTS OF THE INDICES IN J
        self.W = np.array([np.sum(self.W[_IDX]) for _IDX in self.J])
        self.W = self.W / np.sum(self.W)  # NORMALIZES THE WEIGHTS

    # PREDICT(): PREDICTS THE OUTPUT DATA
    def PREDICT(self, X):
        """PREDICTS THE OUTPUT DATA

        PARAMETERS
        ----------
        X : NUMPY ARRAY (N_SAMPLES, N_FEATURES)
            IT'S THE INPUT DATA.

        RETURNS
        -------
        Y : NUMPY ARRAY (N_SAMPLES, 1)
            IT'S THE OUTPUT DATA.

        EXCEPTIONS
        ----------
        EXCEPTION : FIT THE MODEL BEFORE USING IT
            THIS EXCEPTION IS RAISED WHEN THE MODEL IS USED BEFORE BEING FITTED.
        """
        assert self.J is not None and self.W is not None, 'FIT THE MODEL BEFORE USING IT'  # ASSERTS THAT J AND THE WEIGHTS ARE NOT NONE
        Y = np.zeros(len(X))  # SETS Y TO A VECTOR OF ZEROS
        for IDX in range(len(X)):  # FOR EACH INDEX IN THE LENGTH OF X
            IDX0 = self.J[IDX]  # SETS IDX0 TO THE CURRENT INDEX
            # SETS THE CURRENT ELEMENT OF Y TO THE AVERAGE OF THE WEIGHTS OF THE CURRENT INDEX
            Y[IDX] = np.dot(self.W[IDX0], X[IDX0]) / np.sum(self.W[IDX0])
        return Y  # RETURNS Y
