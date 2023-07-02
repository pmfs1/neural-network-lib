import numpy as np
import pandas as pd

# CATEGORICAL_NAIVE_BAYES: CLASS THAT IMPLEMENTS CATEGORICAL NAIVE BAYES MODEL


class CATEGORICAL_NAIVE_BAYES:
    # INITIALIZES THE CATEGORICAL NAIVE BAYES MODEL
    def __init__(self, ALPHA=1):
        # ALPHA: HYPERPARAMETER THAT CONTROLS THE SMOOTHING OF THE MODEL.
        self.ALPHA = ALPHA
        # PROBABILITIES: DICTIONARY THAT STORES THE PROBABILITIES OF EACH FEATURE.
        self.PROBABILITIES = {}
        # CONDITIONAL PROBABILITIES: DICTIONARY THAT STORES THE CONDITIONAL PROBABILITIES OF EACH FEATURE.
        self.CONDITIONAL_PROBABILITIES = {}
        self.TARGETS = []  # TARGETS: LIST THAT STORES THE UNIQUE TARGETS.
        self.COLUMNS = []  # COLUMNS: LIST THAT STORES THE COLUMN NAMES.

    # FIT(): TRAINS THE CATEGORICAL NAIVE BAYES MODEL
    def fit(self, X, Y, COLUMN_NAMES):
        # DATASET: DATAFRAME THAT STORES THE DATASET.
        DATASET = pd.DataFrame(data=X, index=None, columns=COLUMN_NAMES[:-1])
        # TARGET_COLUMN_NAME: STRING THAT STORES THE TARGET COLUMN NAME.
        TARGET_COLUMN_NAME = COLUMN_NAMES[-1]
        # ADDS THE TARGET COLUMN TO THE DATASET.
        DATASET[TARGET_COLUMN_NAME] = Y
        # COLUMNS: LIST THAT STORES THE COLUMN NAMES.
        self.COLUMNS = COLUMN_NAMES
        # TARGETS: LIST THAT STORES THE UNIQUE TARGETS.
        self.TARGETS = list(set(Y))
        # PROBABILITIES: DICTIONARY THAT STORES THE PROBABILITIES OF EACH FEATURE.
        self.PROBABILITIES = {COLUMN: {} for COLUMN in self.COLUMNS}
        # CONDITIONAL PROBABILITIES: DICTIONARY THAT STORES THE CONDITIONAL PROBABILITIES OF EACH FEATURE.
        self.CONDITIONAL_PROBABILITIES = {
            COLUMN: {} for COLUMN in self.COLUMNS}
        for COLUMN in self.COLUMNS:  # LOOP THAT ITERATES OVER THE COLUMNS.
            # UNIQUE_VALUES: LIST THAT STORES THE UNIQUE VALUES OF THE COLUMN.
            UNIQUE_VALUES = list(set(DATASET[COLUMN]))
            for VALUE in UNIQUE_VALUES:  # LOOP THAT ITERATES OVER THE UNIQUE VALUES.
                # PROBABILITY: FLOAT THAT STORES THE PROBABILITY OF THE VALUE.
                PROBABILITY = len(
                    DATASET[DATASET[COLUMN] == VALUE]) / len(DATASET)
                # PROBABILITIES[COLUMN][VALUE]: STORES THE PROBABILITY OF THE VALUE.
                self.PROBABILITIES[COLUMN][VALUE] = PROBABILITY
                for TARGET in self.TARGETS:  # LOOP THAT ITERATES OVER THE TARGETS.
                    # CONDITIONAL_PROBABILITY: FLOAT THAT STORES THE CONDITIONAL PROBABILITY OF THE VALUE.
                    CONDITIONAL_PROBABILITY = len(
                        DATASET[(DATASET[COLUMN] == VALUE) & (DATASET[TARGET_COLUMN_NAME] == TARGET)]) / len(DATASET[DATASET[TARGET_COLUMN_NAME] == TARGET])
                    # CONDITIONAL_PROBABILITIES[COLUMN][f'{VALUE}-{TARGET}']: STORES THE CONDITIONAL PROBABILITY OF THE VALUE.
                    self.CONDITIONAL_PROBABILITIES[COLUMN][f'{VALUE}-{TARGET}'] = CONDITIONAL_PROBABILITY

    # TRANSFORM(): PREDICTS THE TARGET OF THE GIVEN DATA.
    def TRANSFORM(self, X):
        # PREDICTION: LIST THAT STORES THE PREDICTION OF THE TARGETS.
        PREDICTION = []
        for ROW in X:  # LOOP THAT ITERATES OVER THE ROWS OF THE DATA.
            # TARGET_PROBABILITY_DICT: DICTIONARY THAT STORES THE PROBABILITIES OF EACH TARGET.
            TARGET_PROBABILITY_DICT = {}
            for TARGET in self.TARGETS:  # LOOP THAT ITERATES OVER THE TARGETS.
                # ROW_CONDITIONAL_PROBABILITIES: LIST THAT STORES THE CONDITIONAL PROBABILITIES OF EACH FEATURE.
                ROW_CONDITIONAL_PROBABILITIES = [
                    self.CONDITIONAL_PROBABILITIES[COLUMN][f'{VALUE}-{TARGET}'] for COLUMN, VALUE in zip(self.COLUMNS, ROW)]
                # CALCULATES THE PROBABILITY OF THE TARGET.
                TARGET_PROBABILITY_DICT[TARGET] = (np.prod(
                    ROW_CONDITIONAL_PROBABILITIES) * self.PROBABILITIES[self.COLUMNS[-1]][TARGET])
            # APPENDS THE TARGET WITH THE HIGHEST PROBABILITY.
            PREDICTION.append(max(TARGET_PROBABILITY_DICT,
                              key=TARGET_PROBABILITY_DICT.get))
        return PREDICTION  # RETURNS THE PREDICTION OF THE TARGETS.
