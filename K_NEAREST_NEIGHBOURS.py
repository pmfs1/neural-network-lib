from collections import Counter
import numpy as np

# K_NEAREST_NEIGHBOURS: CLASS THAT IMPLEMENTS K NEAREST NEIGHBOURS MODEL


class K_NEAREST_NEIGHBOURS:
    # INITIALIZES THE K NEAREST NEIGHBOURS MODEL
    def __init__(self, K=3):
        # K: IT'S THE HYPERPARAMETER THAT CORRESPONDS TO THE NUMBER OF NEAREST NEIGHBOURS.
        self.K = K
        # X: IT'S THE DATASET.
        self.X = None
        # Y: IT'S THE ARRAY THAT CONTAINS THE CLASS LABELS OF THE DATASET.
        self.Y = None

    # FIT(): METHOD THAT TRAINS THE K NEAREST NEIGHBOURS MODEL
    def FIT(self, X, Y):
        self.X = X  # X: IT'S THE DATASET.
        # Y: IT'S THE ARRAY THAT CONTAINS THE CLASS LABELS OF THE DATASET.
        self.Y = Y

    # PREDICT(): METHOD THAT PREDICTS THE DATASET
    def PREDICT(self, X):
        # PREDICTION: IT'S THE ARRAY THAT CONTAINS THE PREDICTED CLASS LABELS OF THE DATASET.
        PREDICTION = [self.__PREDICT__(_X) for _X in X]
        return np.array(PREDICTION)  # RETURNS THE PREDICTED DATASET

    # __PREDICT__() [PRIVATE METHOD]: METHOD THAT PREDICTS A SAMPLE
    def __PREDICT__(self, X):
        # DISTANCES: IT'S THE ARRAY THAT CONTAINS THE EUCLIDEAN DISTANCES BETWEEN THE SAMPLE X AND THE SAMPLES IN THE DATASET.
        DISTANCES = [self.EUCLIDEAN_DISTANCE(X, _X) for _X in self.X]
        # K_INDEX_LENGTH: IT'S THE ARRAY THAT CONTAINS THE INDEXES OF THE K NEAREST NEIGHBOURS.
        K_INDEX_LENGTH = np.argsort(DISTANCES)[:self.K]
        # K_NEIGHBOR_LABELS: IT'S THE ARRAY THAT CONTAINS THE CLASS LABELS OF THE K NEAREST NEIGHBOURS.
        K_NEIGHBOR_LABELS = [self.Y[INDEX] for INDEX in K_INDEX_LENGTH]
        # MOST_COMMON: IT'S THE MOST COMMON CLASS LABEL OF THE K NEAREST NEIGHBOURS.
        MOST_COMMON = Counter(K_NEIGHBOR_LABELS).most_common(1)
        # RETURNS THE MOST COMMON CLASS LABEL OF THE K NEAREST NEIGHBOURS
        return MOST_COMMON[0][0]

    # EUCLIDEAN_DISTANCE() [STATIC METHOD]: METHOD THAT CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO SAMPLES
    @staticmethod
    def EUCLIDEAN_DISTANCE(X_1, X_2):
        # RETURNS THE EUCLIDEAN DISTANCE BETWEEN TWO SAMPLES: SQRT(SUM((X_1 - X_2) ** 2))
        return np.sqrt(np.sum((X_1 - X_2) ** 2))
