import numpy as np

# BERNOULLI_NAIVE_BAYES: CLASS THAT IMPLEMENTS BERNOULLI NAIVE BAYES MODEL
class BERNOULLI_NAIVE_BAYES:
    # INITIALIZES THE BERNOULLI NAIVE BAYES MODEL
    def __init__(self, ALPHA=1):
        # ALPHA: HYPERPARAMETER THAT CONTROLS THE SMOOTHING OF THE MODEL.
        self.ALPHA = ALPHA
        # NUMBER OF CLASSES: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF CLASSES IN THE DATASET.
        self.N_CLASSES = None
        # NUMBER OF FEATURES: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF FEATURES IN THE DATASET.
        self.N_FEATURES = None
        self.LOG_CLASS_PRIOR = None  # LOG OF THE PRIOR PROBABILITY OF EACH CLASS
        # LOG OF THE CONDITIONAL PROBABILITY OF EACH FEATURE GIVEN THE POSITIVE CLASS
        self.LOG_CLASS_CONDITIONAL_POSITIVE = None
        # LOG OF THE CONDITIONAL PROBABILITY OF EACH FEATURE GIVEN THE NEGATIVE CLASS
        self.LOG_CLASS_CONDITIONAL_NEGATIVE = None
        self.LOG_CLASS_PRIOR = None  # LOG OF THE PRIOR PROBABILITY OF EACH CLASS

    # FIT(): TRAINS THE BERNOULLI NAIVE BAYES MODEL
    def fit(self, X, Y):
        # GETTING THE COUNTS OF EACH CLASS
        Y_COUNTS = np.unique(Y, return_counts=True)[1]
        self.N_CLASSES = len(np.unique(Y))  # GETTING THE NUMBER OF CLASSES
        self.N_FEATURES = X.shape[1]  # GETTING THE NUMBER OF FEATURES
        # CALCULATING THE PRIOR PROBABILITY OF EACH CLASS
        CLASS_PRIOR = Y_COUNTS / Y_COUNTS.sum()
        self.LOG_CLASS_PRIOR = np.expand_dims(np.log(CLASS_PRIOR), axis=1)
        # CALCULATING THE LIKELIHOOD OF EACH FEATURE GIVEN EACH CLASS
        PROB_X_GIVEN_Y = np.zeros([self.N_CLASSES, self.N_FEATURES])
        for CLASS_INDEX in range(self.N_CLASSES):  # FOR EACH CLASS
            # GETTING THE ROWS THAT BELONG TO THE CLASS
            ROW_MASK = (Y == CLASS_INDEX)
            # GETTING THE FEATURES THAT BELONG TO THE CLASS
            X_FILTERED = X[ROW_MASK, :]
            # CALCULATING THE NUMERATOR OF THE LIKELIHOOD
            NUMERATOR = (X_FILTERED.sum(axis=0) + self.ALPHA)
            # CALCULATING THE DENOMINATOR OF THE LIKELIHOOD
            DENOMINATOR = (X_FILTERED.shape[0] + 2 * self.ALPHA)
            # CALCULATING THE LIKELIHOOD OF EACH FEATURE GIVEN THE CLASS
            PROB_X_GIVEN_Y[CLASS_INDEX, :] = NUMERATOR / DENOMINATOR
        # CALCULATING THE LOG LIKELIHOOD OF EACH FEATURE GIVEN THE POSITIVE CLASS
        self.LOG_CLASS_CONDITIONAL_POSITIVE = np.log(PROB_X_GIVEN_Y)
        # CALCULATING THE LOG LIKELIHOOD OF EACH FEATURE GIVEN THE NEGATIVE CLASS
        self.LOG_CLASS_CONDITIONAL_NEGATIVE = np.log(1 - PROB_X_GIVEN_Y)

    # TRANSFORM(): PREDICTS THE CLASS OF EACH SAMPLE IN THE DATASET
    def TRANSFORM(self, X):
        X = X.todense()  # CONVERTING THE SPARSE MATRIX TO A DENSE MATRIX
        # CALCULATING THE LOG LIKELIHOOD OF EACH FEATURE GIVEN THE POSITIVE CLASS
        LOG_PROBS_POSITIVE = self.LOG_CLASS_CONDITIONAL_POSITIVE.dot(X.T)
        # CALCULATING THE LOG LIKELIHOOD OF EACH FEATURE GIVEN THE NEGATIVE CLASS
        LOG_PROBS_NEGATIVE = self.LOG_CLASS_CONDITIONAL_NEGATIVE.dot(1 - X.T)
        # CALCULATING THE LOG LIKELIHOOD OF EACH SAMPLE GIVEN EACH CLASS
        LOG_LIKELIHOODS = LOG_PROBS_POSITIVE + LOG_PROBS_NEGATIVE
        # CALCULATING THE LOG JOINT LIKELIHOOD OF EACH SAMPLE GIVEN EACH CLASS
        LOG_JOINT_LIKEHOODS = LOG_LIKELIHOODS + self.LOG_CLASS_PRIOR
        # CALCULATING THE HYPOTHESIS OF EACH SAMPLE
        HYPOTHESIS = np.argmax(LOG_JOINT_LIKEHOODS, axis=0)
        # CONVERTING THE HYPOTHESIS TO A NUMPY ARRAY
        HYPOTHESIS = np.array(HYPOTHESIS).squeeze()
        return HYPOTHESIS  # RETURNING THE HYPOTHESIS