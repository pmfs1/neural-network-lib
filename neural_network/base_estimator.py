import numpy as np

class BASE_ESTIMATOR:
    Y_REQUIRED = True # Y_REQUIRED DEFINES WHETHER THE ESTIMATOR REQUIRES A SET OF Y TARGET VALUES OR NOT. BY DEFAULT, IT IS SET TO TRUE.
    FIT_REQUIRED = True # FIT_REQUIRED DEFINES WHETHER THE ESTIMATOR REQUIRES A FIT METHOD OR NOT. BY DEFAULT, IT IS SET TO TRUE.

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
        if not isinstance(X, np.ndarray): # IF X IS NOT A NUMPY ARRAY
            X = np.array(X) # CONVERT IT TO ONE
        if X.size == 0: # IF X (OR THE CONVERTED X) IS EMPTY
            raise ValueError("GOT AN EMPTY MATRIX") # RAISE A VALUE ERROR
        if X.ndim == 1: # IF X IS 1-DIMENSIONAL
            self.N_SAMPLES, self.N_FEATURES = 1, X.shape # SET THE NUMBER OF SAMPLES TO 1 AND THE NUMBER OF FEATURES TO THE SHAPE OF X
        else: # OTHERWISE, IF X IS 2-DIMENSIONAL OR GREATER
            self.N_SAMPLES, self.N_FEATURES = X.shape[0], np.prod(X.shape[1:]) # SET THE NUMBER OF SAMPLES TO THE FIRST ELEMENT OF THE SHAPE OF X AND THE NUMBER OF FEATURES TO THE PRODUCT OF THE REMAINING ELEMENTS OF THE SHAPE OF X
        self.X = X # SET THE X ATTRIBUTE TO X
        if self.Y_REQUIRED: # IF Y_REQUIRED IS TRUE
            if Y is None: # IF Y IS NONE
                raise ValueError("MISSED REQUIRED ARGUMENT Y") # RAISE A VALUE ERROR
            if not isinstance(Y, np.ndarray): # IF Y IS NOT A NUMPY ARRAY
                Y = np.array(Y) # CONVERT IT TO ONE
            if Y.size == 0: # IF Y (OR THE CONVERTED Y) IS EMPTY
                raise ValueError("THE TARGETS ARRAY MUST NOT BE EMPTY") # RAISE A VALUE ERROR
        self.Y = Y # SET THE Y ATTRIBUTE TO Y

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
        self.__SETUP_INPUT__(X, Y) # CALL THE __SETUP_INPUT__ FUNCTION WITH ARGUMENTS X AND Y

    def PREDICT(self, X=None):
        """PREDICT TARGET VALUES FOR A DATASET.

        PARAMETERS
        ----------
        X : ARRAY-LIKE
            FEATURE DATASET.
        """
        if not isinstance(X, np.ndarray): # IF X IS NOT A NUMPY ARRAY
            X = np.array(X) # CONVERT IT TO ONE
        return self.__PREDICT__(X) # RETURN THE RESULT OF CALLING THE __PREDICT__ FUNCTION WITH ARGUMENT X
    
    def __PREDICT__(self, X=None):
        """PREDICT TARGET VALUES FOR A DATASET.

        PARAMETERS
        ----------
        X : ARRAY-LIKE
            FEATURE DATASET.
        """
        raise NotImplementedError() # RAISE A NOT IMPLEMENTED ERROR