import numpy as np


class BASE_ESTIMATOR:
    """BASE CLASS FOR ESTIMATORS IN THE NEURAL NETWORKS MODULE.

    THIS CLASS DEFINES THE BASIC METHODS AND ATTRIBUTES THAT ALL ESTIMATORS
    SHOULD HAVE. IT ALSO PROVIDES A __SETUP_INPUT__ METHOD THAT ENSURES INPUTS
    TO AN ESTIMATOR ARE IN THE EXPECTED FORMAT.

    PARAMETERS
    ----------
    NONE

    ATTRIBUTES
    ----------
    N_SAMPLES : INT
        NUMBER OF SAMPLES.
    N_FEATURES : INT
        NUMBER OF FEATURES.
    X : NUMPY NDARRAY
        FEATURE DATASET.
    Y : NUMPY NDARRAY
        TARGET VALUES.

    METHODS
    -------
    __SETUP_INPUT__(X, Y=None)
        ENSURE INPUTS TO AN ESTIMATOR ARE IN THE EXPECTED FORMAT.
    FIT(X, Y)
        FIT THE ESTIMATOR TO THE GIVEN DATASET.
    PREDICT(X)
        PREDICT THE TARGET VALUES OF THE GIVEN DATASET.
    """
    # Y_REQUIRED DEFINES WHETHER THE ESTIMATOR REQUIRES A SET OF Y TARGET VALUES OR NOT. BY DEFAULT, IT IS SET TO TRUE.
    Y_REQUIRED = True
    # FIT_REQUIRED DEFINES WHETHER THE ESTIMATOR REQUIRES A FIT METHOD OR NOT. BY DEFAULT, IT IS SET TO TRUE.
    FIT_REQUIRED = True

    def __init__(self):
        """INITIALIZES THE BASE ESTIMATOR CLASS

        PARAMETERS
        ----------
        NONE

        ATTRIBUTES
        ----------
        N_SAMPLES : INT
            NUMBER OF SAMPLES.
        N_FEATURES : INT
            NUMBER OF FEATURES.
        X : NUMPY NDARRAY
            FEATURE DATASET.
        Y : NUMPY NDARRAY
            TARGET VALUES.    
        """
        # N_SAMPLES: IT'S THE NUMBER OF SAMPLES.
        self.N_SAMPLES = None
        # N_FEATURES: IT'S THE NUMBER OF FEATURES.
        self.N_FEATURES = None
        # X: IT'S THE FEATURE DATASET.
        self.X = None
        # Y: IT'S THE TARGET VALUES.
        self.Y = None

    def __SETUP_INPUT__(self, X, Y=None):
        """ENSURE INPUTS TO AN ESTIMATOR ARE IN THE EXPECTED FORMAT.

        ENSURES X AND Y ARE STORED AS NUMPY NDARRAYS BY CONVERTING FROM AN
        ARRAY-LIKE OBJECT IF NECESSARY. ENABLES ESTIMATORS TO DEFINE WHETHER
        THEY REQUIRE A SET OF Y TARGET VALUES OR NOT WITH Y_REQUIRED, E.G.
        KMEANS CLUSTERING REQUIRES NO TARGET LABELS AND IS FIT AGAINST ONLY X.

        PARAMETERS
        ----------
        X : ARRAY-LIKE
            FEATURE DATASET.
        Y : ARRAY-LIKE
            TARGET VALUES. BY DEFAULT IS REQUIRED, BUT IF Y_REQUIRED = FALSE
            THEN MAY BE OMITTED.
        """
        if not isinstance(X, np.ndarray):  # IF X IS NOT A NUMPY ARRAY
            X = np.array(X)  # CONVERT IT TO ONE
        if X.size == 0:  # IF X (OR THE CONVERTED X) IS EMPTY
            raise ValueError("GOT AN EMPTY MATRIX")  # RAISE A VALUE ERROR
        if X.ndim == 1:  # IF X IS 1-DIMENSIONAL
            # SET THE NUMBER OF SAMPLES TO 1 AND THE NUMBER OF FEATURES TO THE SHAPE OF X
            self.N_SAMPLES, self.N_FEATURES = 1, X.shape
        else:  # OTHERWISE, IF X IS 2-DIMENSIONAL OR GREATER
            # SET THE NUMBER OF SAMPLES TO THE FIRST ELEMENT OF THE SHAPE OF X AND THE NUMBER OF FEATURES TO THE PRODUCT OF THE REMAINING ELEMENTS OF THE SHAPE OF X
            self.N_SAMPLES, self.N_FEATURES = X.shape[0], np.prod(X.shape[1:])
        self.X = X  # SET THE X ATTRIBUTE TO X
        if self.Y_REQUIRED:  # IF Y_REQUIRED IS TRUE
            if Y is None:  # IF Y IS NONE
                # RAISE A VALUE ERROR
                raise ValueError("MISSED REQUIRED ARGUMENT Y")
            if not isinstance(Y, np.ndarray):  # IF Y IS NOT A NUMPY ARRAY
                Y = np.array(Y)  # CONVERT IT TO ONE
            if Y.size == 0:  # IF Y (OR THE CONVERTED Y) IS EMPTY
                # RAISE A VALUE ERROR
                raise ValueError("THE TARGETS ARRAY MUST NOT BE EMPTY")
        self.Y = Y  # SET THE Y ATTRIBUTE TO Y

    def FIT(self, X, Y=None):
        """FIT AN ESTIMATOR TO A DATASET.

        PARAMETERS
        ----------
        X : ARRAY-LIKE
            FEATURE DATASET.
        Y : ARRAY-LIKE
            TARGET VALUES. BY DEFAULT IS REQUIRED, BUT IF Y_REQUIRED = FALSE
            THEN MAY BE OMITTED.
        """
        self.__SETUP_INPUT__(
            X, Y)  # CALL THE __SETUP_INPUT__ FUNCTION WITH ARGUMENTS X AND Y

    def PREDICT(self, X=None):
        """PREDICT TARGET VALUES FOR A DATASET.

        PARAMETERS
        ----------
        X : ARRAY-LIKE
            FEATURE DATASET.
        """
        if not isinstance(X, np.ndarray):  # IF X IS NOT A NUMPY ARRAY
            X = np.array(X)  # CONVERT IT TO ONE
        # RETURN THE RESULT OF CALLING THE __PREDICT__ FUNCTION WITH ARGUMENT X
        return self.__PREDICT__(X)

    def __PREDICT__(self, X=None):
        """PREDICT TARGET VALUES FOR A DATASET.

        PARAMETERS
        ----------
        X : ARRAY-LIKE
            FEATURE DATASET.
        """
        raise NotImplementedError()  # RAISE A NOT IMPLEMENTED ERROR
