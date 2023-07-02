import numpy as np
from collections import Counter
from DECISION_TREE.NODE import NODE

# DECISION_TREE: CLASS THAT IMPLEMENTS DECISION TREE MODEL
class DECISION_TREE:
    # INITIALIZES THE DECISION TREE MODEL
    def __init__(self, MIN_SAMPLES_SPLIT=2, MAX_DEPTH=100, N_FEATURES=None):
        self.MIN_SAMPLES_SPLIT = MIN_SAMPLES_SPLIT # MIN_SAMPLES_SPLIT: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE MINIMUM NUMBER OF SAMPLES REQUIRED TO SPLIT AN INTERNAL NODE.
        self.MAX_DEPTH = MAX_DEPTH # MAX_DEPTH: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE MAXIMUM DEPTH OF THE TREE.
        self.N_FEATURES = N_FEATURES # N_FEATURES: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF FEATURES.
        self.ROOT = None # ROOT: IT'S THE ROOT OF THE TREE.

    # FIT(): METHOD THAT TRAINS THE DECISION TREE MODEL
    def FIT(self, X, Y):
        self.N_FEATURES = X.shape[1] if not self.N_FEATURES else min(self.N_FEATURES, X.shape[1]) # N_FEATURES: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF FEATURES. IT'S EQUAL TO THE NUMBER OF FEATURES OF THE DATASET IF N_FEATURES IS NONE, OTHERWISE IT'S EQUAL TO THE MINIMUM BETWEEN N_FEATURES AND THE NUMBER OF FEATURES OF THE DATASET.
        self.ROOT = self.__GROW_TREE__(X, Y) # ROOT: IT'S THE ROOT OF THE TREE. IT'S THE OUTPUT OF THE __GROW_TREE__() FUNCTION.

    # TRANSFORM(): METHOD THAT TRANSFORMS THE DATASET
    def TRANSFORM(self, X):
        return np.array([self.__TRAVERSE_TREE__(X_I, self.ROOT) for X_I in X]) # RETURNS THE TRANSFORMED DATASET: [TRAVERSE_TREE(X[I], ROOT) FOR X[I] IN X]. IT'S THE OUTPUT OF THE __TRAVERSE_TREE__() FUNCTION.

    # __GROW_TREE__() [PRIVATE METHOD]: METHOD THAT GROWS THE TREE
    def __GROW_TREE__(self, X, Y, DEPTH=0):
        NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES = X.shape # NUMBER OF TRAINING EXAMPLES: IT'S THE NUMBER OF ROWS OF THE DATASET; NUMBER OF FEATURES: IT'S THE NUMBER OF COLUMNS OF THE DATASET.
        NUMBER_OF_LABELS = len(np.unique(Y)) # NUMBER OF LABELS: IT'S THE NUMBER OF UNIQUE LABELS IN THE DATASET.
        # STOPPING CRITERIA
        if (DEPTH >= self.MAX_DEPTH) or (NUMBER_OF_LABELS == 1) or (NUMBER_OF_SAMPLES < self.MIN_SAMPLES_SPLIT): # IF DEPTH >= MAX_DEPTH OR NUMBER_OF_LABELS == 1 OR NUMBER_OF_SAMPLES < MIN_SAMPLES_SPLIT, THEN
            LEAF_VALUE = self.__MOST_COMMON_LABEL__(Y) # LEAF_VALUE: IT'S THE MOST COMMON LABEL IN THE DATASET. IT'S THE OUTPUT OF THE __MOST_COMMON_LABEL__() FUNCTION.
            return NODE(LEAF_VALUE) # RETURNS A NODE WITH LEAF_VALUE AS VALUE.
        FEATURE_INDEXES = np.random.choice(NUMBER_OF_FEATURES, self.N_FEATURES, replace=False) # FEATURE_INDEXES: IT'S A LIST OF N_FEATURES RANDOMLY SELECTED FEATURES.
        # GREEDILY SELECT THE BEST SPLIT ACCORDING TO INFORMATION GAIN
        BEST_FEATURE_INDEX, BEST_THRESHOLD = self.__BEST_CRITERIA__(X, Y, FEATURE_INDEXES) # BEST_FEATURE_INDEX: IT'S THE INDEX OF THE FEATURE THAT MAXIMIZES THE INFORMATION GAIN; BEST_THRESHOLD: IT'S THE THRESHOLD THAT MAXIMIZES THE INFORMATION GAIN. THEY ARE THE OUTPUTS OF THE __BEST_CRITERIA__() FUNCTION.
        # GROW THE CHILDREN THAT RESULT FROM THE SPLIT
        LEFT_INDEXES, RIGHT_INDEXES = self.__SPLIT__(X[:, BEST_FEATURE_INDEX], BEST_THRESHOLD) # LEFT_INDEXES: IT'S THE LIST OF INDEXES OF THE LEFT CHILDREN; RIGHT_INDEXES: IT'S THE LIST OF INDEXES OF THE RIGHT CHILDREN. THEY ARE THE OUTPUTS OF THE __SPLIT__() FUNCTION.
        LEFT = self.__GROW_TREE__(X[LEFT_INDEXES, :], Y[LEFT_INDEXES], DEPTH + 1) # LEFT: IT'S THE LEFT CHILD. IT'S THE OUTPUT OF THE __GROW_TREE__() FUNCTION.
        RIGHT = self.__GROW_TREE__(X[RIGHT_INDEXES, :], Y[RIGHT_INDEXES], DEPTH + 1) # RIGHT: IT'S THE RIGHT CHILD. IT'S THE OUTPUT OF THE __GROW_TREE__() FUNCTION.
        return NODE(BEST_FEATURE_INDEX, BEST_THRESHOLD, LEFT, RIGHT) # RETURNS A NODE WITH BEST_FEATURE_INDEX AS FEATURE_INDEX, BEST_THRESHOLD AS THRESHOLD, LEFT AS LEFT CHILD AND RIGHT AS RIGHT CHILD.
    
    # __TRAVERSE_TREE__() [PRIVATE METHOD]: METHOD THAT TRAVERSES THE TREE
    def __TRAVERSE_TREE__(self, X, NODE):
        if NODE.LEAF_VALUE is not None: # IF NODE.LEAF_VALUE IS NOT NONE, THEN
            return NODE.LEAF_VALUE # RETURNS NODE.LEAF_VALUE
        FEATURE_VALUE = X[NODE.FEATURE_INDEX] # FEATURE_VALUE: IT'S THE VALUE OF THE FEATURE WITH INDEX NODE.FEATURE_INDEX IN THE INPUT X.
        CHILD = NODE.LEFT if FEATURE_VALUE <= NODE.THRESHOLD else NODE.RIGHT # CHILD: IT'S THE LEFT CHILD IF FEATURE_VALUE <= NODE.THRESHOLD, OTHERWISE IT'S THE RIGHT CHILD.
        return self.__TRAVERSE_TREE__(X, CHILD) # RETURNS THE OUTPUT OF THE __TRAVERSE_TREE__() FUNCTION.

    # __TRAVERSE_TREE__() [PRIVATE METHOD]: METHOD THAT TRAVERSES THE TREE
    # def __TRAVERSE_TREE__(self, X, NODE):
    #     if NODE.LEAF_VALUE is not None: # IF NODE.LEAF_VALUE IS NOT NONE, THEN
    #         return NODE.LEAF_VALUE # RETURNS NODE.LEAF_VALUE
    #     if X[NODE.FEATURE] <= NODE.THRESHOLD: # IF X[NODE.feature] <= NODE.THRESHOLD, THEN
    #         return self.__TRAVERSE_TREE__(X, NODE.LEFT) # RETURNS THE OUTPUT OF THE _TRAVERSE_TREE() FUNCTION.
    #     return self.__TRAVERSE_TREE__(X, NODE.RIGHT) # RETURNS THE OUTPUT OF THE _TRAVERSE_TREE() FUNCTION.

    # __MOST_COMMON_LABEL__() [PRIVATE METHOD]: METHOD THAT RETURNS THE MOST COMMON LABEL IN THE DATASET~
    def __MOST_COMMON_LABEL__(self, Y):
        return Counter(Y).most_common(1)[0][0] # RETURNS THE MOST COMMON LABEL IN THE DATASET.

    # __BEST_CRITERIA__() [PRIVATE METHOD]: METHOD THAT RETURNS THE BEST CRITERIA
    def __BEST_CRITERIA__(self, X, Y, FEATURE_INDEXES):
        BEST_GAIN = -1 # BEST_GAIN: IT'S THE BEST GAIN. IT'S INITIALIZED TO -1.
        BEST_FEATURE_INDEX, BEST_THRESHOLD = None, None # BEST_FEATURE_INDEX: IT'S THE INDEX OF THE FEATURE THAT MAXIMIZES THE INFORMATION GAIN; BEST_THRESHOLD: IT'S THE THRESHOLD THAT MAXIMIZES THE INFORMATION GAIN. THEY ARE INITIALIZED TO NONE.
        for FEATURE_INDEX in FEATURE_INDEXES: # FOR EACH FEATURE_INDEX IN FEATURE_INDEXES, THEN
            X_COLUMN = X[:, FEATURE_INDEX] # X_COLUMN: IT'S THE COLUMN OF X WITH INDEX FEATURE_INDEX.
            THRESHOLDS = np.unique(X_COLUMN) # THRESHOLDS: IT'S THE LIST OF UNIQUE VALUES IN X_COLUMN.
            for THRESHOLD in THRESHOLDS: # FOR EACH THRESHOLD IN THRESHOLDS, THEN
                GAIN = self.__INFORMATION_GAIN__(Y, X_COLUMN, THRESHOLD) # GAIN: IT'S THE INFORMATION GAIN. IT'S THE OUTPUT OF THE __INFORMATION_GAIN__() FUNCTION.
                if GAIN > BEST_GAIN: # IF GAIN > BEST_GAIN, THEN
                    BEST_GAIN = GAIN # BEST_GAIN = GAIN
                    BEST_FEATURE_INDEX = FEATURE_INDEX # BEST_FEATURE_INDEX = FEATURE_INDEX
                    BEST_THRESHOLD = THRESHOLD # BEST_THRESHOLD = THRESHOLD
        return BEST_FEATURE_INDEX, BEST_THRESHOLD # RETURNS BEST_FEATURE_INDEX AND BEST_THRESHOLD.
    
    # __INFORMATION_GAIN__() [PRIVATE METHOD]: METHOD THAT RETURNS THE INFORMATION GAIN
    def __INFORMATION_GAIN__(self, Y, X_COLUMN, SPLIT_THRESHOLD):
        PARENT_ENTROPY = self.ENTROPY(Y) # PARENT_ENTROPY: IT'S THE ENTROPY OF THE PARENT. IT'S THE OUTPUT OF THE ENTROPY() FUNCTION.
        LEFT_INDEXES, RIGHT_INDEXES = self.__SPLIT__(X_COLUMN, SPLIT_THRESHOLD) # LEFT_INDEXES: IT'S THE LIST OF INDEXES OF THE LEFT CHILDREN; RIGHT_INDEXES: IT'S THE LIST OF INDEXES OF THE RIGHT CHILDREN. THEY ARE THE OUTPUTS OF THE __SPLIT__() FUNCTION.
        if len(LEFT_INDEXES) == 0 or len(RIGHT_INDEXES) == 0: # IF THE LENGTH OF LEFT_INDEXES IS 0 OR THE LENGTH OF RIGHT_INDEXES IS 0, THEN
            return 0 # RETURNS 0.
        N = len(Y) # N: IT'S THE LENGTH OF Y.
        N_L, N_R = len(LEFT_INDEXES), len(RIGHT_INDEXES) # N_L: IT'S THE LENGTH OF LEFT_INDEXES; N_R: IT'S THE LENGTH OF RIGHT_INDEXES.
        CHILDREN_ENTROPY = (N_L / N) * self.ENTROPY(Y[LEFT_INDEXES]) + (N_R / N) * self.ENTROPY(Y[RIGHT_INDEXES]) # CHILDREN_ENTROPY: IT'S THE ENTROPY OF THE CHILDREN. IT'S THE SUM OF THE PRODUCT OF THE PROBABILITY OF THE LEFT CHILDREN AND THE ENTROPY OF THE LEFT CHILDREN AND THE PRODUCT OF THE PROBABILITY OF THE RIGHT CHILDREN AND THE ENTROPY OF THE RIGHT CHILDREN.
        return PARENT_ENTROPY - CHILDREN_ENTROPY # RETURNS THE INFORMATION GAIN.
    
    # __SPLIT__() [PRIVATE METHOD]: METHOD THAT SPLITS THE DATASET
    def __SPLIT__(self, X_COLUMN, SPLIT_THRESHOLD):
        LEFT_INDEXES = np.argwhere(X_COLUMN <= SPLIT_THRESHOLD).flatten() # LEFT_INDEXES: IT'S THE LIST OF INDEXES OF THE LEFT CHILDREN. IT'S THE INDEXES OF THE ELEMENTS IN X_COLUMN THAT ARE LESS THAN OR EQUAL TO SPLIT_THRESHOLD.
        RIGHT_INDEXES = np.argwhere(X_COLUMN > SPLIT_THRESHOLD).flatten() # RIGHT_INDEXES: IT'S THE LIST OF INDEXES OF THE RIGHT CHILDREN. IT'S THE INDEXES OF THE ELEMENTS IN X_COLUMN THAT ARE GREATER THAN SPLIT_THRESHOLD.
        return LEFT_INDEXES, RIGHT_INDEXES # RETURNS LEFT_INDEXES AND RIGHT_INDEXES.


    # ENTROPY() [STATIC METHOD]: METHOD THAT RETURNS THE ENTROPY
    @staticmethod
    def ENTROPY(Y):
        return -np.sum([P * np.log2(P) for P in (np.bincount(Y) / len(Y)) if P > 0]) # RETURNS THE ENTROPY. IT'S THE SUM OF THE NEGATIVE OF THE PRODUCT OF P AND LOG2(P) FOR EACH P IN THE LIST OF THE PROBABILITIES OF THE LABELS IN Y.