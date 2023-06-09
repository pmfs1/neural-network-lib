import numpy as np
import pandas as pd

# GAUSSIAN_NAIVE_BAYES: CLASS THAT IMPLEMENTS GAUSSIAN NAIVE BAYES MODEL


class GAUSSIAN_NAIVE_BAYES:
    # INITIALIZES THE GAUSSIAN NAIVE BAYES MODEL
    def __init__(self, VAR_SMOOTHING=1e-9):
        # VAR_SMOOTHING: PORTION OF THE LARGEST VARIANCE OF ALL FEATURES THAT IS ADDED TO VARIANCES FOR CALCULATION STABILITY.
        self.VAR_SMOOTHING = VAR_SMOOTHING
        # SIGMAS: DICTIONARY THAT CONTAINS THE VARIANCES OF EACH FEATURE FOR EACH TARGET
        self.SIGMAS = dict()
        # DISPERSIONS: DICTIONARY THAT CONTAINS THE MEANS OF EACH FEATURE FOR EACH TARGET
        self.DISPERSIONS = dict()
        # PROBABILITIES: DICTIONARY THAT CONTAINS THE PROBABILITIES OF EACH TARGET
        self.PROBABILITIES = dict()
        self.TARGETS = list()  # TARGETS: LIST THAT CONTAINS THE TARGETS OF THE DATASET
        self.COLUMNS = list()  # COLUMNS: LIST THAT CONTAINS THE COLUMNS OF THE DATASET

    # FIT(): TRAINS THE GUASSIAN NAIVE BAYES MODEL
    def FIT(self, X, Y, COLUMN_NAMES):
        # DATASET: DATAFRAME THAT CONTAINS THE FEATURES OF THE DATASET
        DATASET = pd.DataFrame(data=X, index=None, columns=COLUMN_NAMES[:-1])
        # TARGET_COLUMN_NAME: STRING THAT CONTAINS THE NAME OF THE TARGET COLUMN
        TARGET_COLUMN_NAME = COLUMN_NAMES[-1]
        # ADDS THE TARGET COLUMN TO THE DATASET
        DATASET[TARGET_COLUMN_NAME] = Y
        # TARGETS: LIST THAT CONTAINS THE TARGETS OF THE DATASET
        self.TARGETS = list(DATASET[TARGET_COLUMN_NAME].unique())
        # COLUMNS: LIST THAT CONTAINS THE COLUMNS OF THE DATASET
        self.COLUMNS = list(DATASET.columns)
        for TARGET in self.TARGETS:  # ITERATES OVER THE TARGETS OF THE DATASET
            # TARGET_DATASET: DATAFRAME THAT CONTAINS THE ROWS OF THE DATASET THAT HAVE THE CURRENT TARGET
            TARGET_DATASET = DATASET[DATASET[TARGET_COLUMN_NAME] == TARGET]
            # PROBABILITY: FLOAT THAT CONTAINS THE PROBABILITY OF THE CURRENT TARGET
            PROBABILITY = len(TARGET_DATASET) / len(DATASET)
            # UPDATES THE PROBABILITIES DICTIONARY
            self.PROBABILITIES[TARGET] = PROBABILITY
            # UPDATES THE SIGMAS DICTIONARY
            self.SIGMAS[TARGET] = dict()
            # UPDATES THE DISPERSIONS DICTIONARY
            self.DISPERSIONS[TARGET] = dict()
            # ITERATES OVER THE FEATURES OF THE DATASET
            for COLUMN in self.COLUMNS[:-1]:
                # SIGMA: FLOAT THAT CONTAINS THE VARIANCE OF THE CURRENT FEATURE FOR THE CURRENT TARGET
                SIGMA = TARGET_DATASET[COLUMN].var(
                ) + self.VAR_SMOOTHING * DATASET[COLUMN].var()
                # DISPERSION: FLOAT THAT CONTAINS THE MEAN OF THE CURRENT FEATURE FOR THE CURRENT TARGET
                DISPERSION = TARGET_DATASET[COLUMN].mean()
                # UPDATES THE SIGMAS DICTIONARY
                self.SIGMAS[TARGET][COLUMN] = SIGMA
                # UPDATES THE DISPERSIONS DICTIONARY
                self.DISPERSIONS[TARGET][COLUMN] = DISPERSION

    # TRANSFORM(): PREDICTS THE TARGETS OF THE GIVEN DATASET
    def TRANSFORM(self, X):
        # PREDICTION: LIST THAT CONTAINS THE PREDICTED TARGETS OF THE GIVEN DATASET
        PREDICTION = list()
        for ROW in X:  # ITERATES OVER THE ROWS OF THE GIVEN DATASET
            # TARGET_PROBABILITIES_DICT: DICTIONARY THAT CONTAINS THE PROBABILITIES OF EACH TARGET FOR THE CURRENT ROW
            TARGET_PROBABILITIES_DICT = dict()
            for TARGET in self.TARGETS:  # ITERATES OVER THE TARGETS OF THE DATASET
                # ITERATES OVER THE FEATURES OF THE CURRENT ROW
                for X, COLUMN in zip(ROW, self.COLUMNS[:-1]):
                    # SIGMA: FLOAT THAT CONTAINS THE VARIANCE OF THE CURRENT FEATURE FOR THE CURRENT TARGET
                    SIGMA = self.SIGMAS[COLUMN][TARGET]
                    # DISPERSION: FLOAT THAT CONTAINS THE MEAN OF THE CURRENT FEATURE FOR THE CURRENT TARGET
                    DISPERSION = self.DISPERSIONS[COLUMN][TARGET]
                    # PROBABILITY: FLOAT THAT CONTAINS THE PROBABILITY OF THE CURRENT TARGET
                    PROBABILITY = self.PROBABILITIES[COLUMN]
                    # TARGET_PROBABILITY: FLOAT THAT CONTAINS THE PROBABILITY OF THE CURRENT TARGET FOR THE CURRENT FEATURE
                    TARGET_PROBABILITY = ((1 / np.sqrt(np.pi * SIGMA)) * np.exp(-(
                        (X - DISPERSION) ** 2 / (2 * SIGMA ** 2)))) * PROBABILITY
                    # UPDATES THE PROBABILITY OF THE CURRENT TARGET FOR THE CURRENT ROW
                    TARGET_PROBABILITIES_DICT[TARGET] = TARGET_PROBABILITIES_DICT.get(
                        TARGET, 1) * TARGET_PROBABILITY
            # APPENDS THE TARGET WITH THE HIGHEST PROBABILITY FOR THE CURRENT ROW
            PREDICTION.append(max(TARGET_PROBABILITIES_DICT,
                              key=TARGET_PROBABILITIES_DICT.get))
        return PREDICTION  # RETURNS THE PREDICTED TARGETS OF THE GIVEN DATASET
