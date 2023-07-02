import numpy as np

# BAYES: CLASS THAT IMPLEMENTS THE NAIVE BAYES ALGORITHM
class BAYES:
    # INITIALIZES THE NAIVE BAYES ALGORITHM
    def __init__(self, C=1):
        # REJECTION THRESHOLD: C. ABOVE THIS THRESHOLD, THE SAMPLE IS CLASSIFIED AS POSITIVE, OTHERWISE IT'S CLASSIFIED AS NEGATIVE.
        self.C = C
        # THRESHOLD DESIGNED TO OFFSET THE THRESHOLD TO A 0 VALUE (COMPARISSON VALUE), B, AND PROBABILITIES, P, IN R^{2 x N}.
        self.B, self.P = None, None

    # FIT(): TRAINS THE MODEL; X REPRESENTS A VECTOR OF SAMPLES, Y REPRESENTS A VECTOR OF CORRESPONDING LABELS (1 FOR POSITIVE, 0 FOR NEGATIVE). EACH SAMPLE IS A DICIONARY WHERE THE KEY IS THE WORD AND THE VALUE IS THE ABSOLUTE FREQUENCY (COUNT) OF THE WORD IN THE SAMPLE.
    def FIT(self, X, Y):
        # COMPUTE M, M_NEGATIVE, M_POSITIVE, N:
        M_POSITIVE = np.sum(Y == 1)  # NUMBER OF POSITIVE SAMPLES: M_POSITIVE
        M_NEGATIVE = np.sum(Y == -1)  # NUMBER OF NEGATIVE SAMPLES: M_NEGATIVE
        M = M_POSITIVE + M_NEGATIVE  # TOTAL NUMBER OF SAMPLES: M
        N = len(X[0])  # NUMBER OF WORDS IN THE VOCABULARY: N
        # INITIALIZE THE B = LOG C + LOG M_NEGATIVE - LOG M_POSITIVE TO OFFSET THE REJECTION THRESHOLD.
        self.B = np.log(self.C) + np.log(M_NEGATIVE) - np.log(M_POSITIVE)
        # INITIALIZE THE PROBABILITIES, P IN R^{2 x N} WITH P_I_J = 1, W_POSITIVE = N, W_NEGATIVE = N:
        self.P = np.ones((2, N))  # PROBABILITIES, P IN R^{2 x N}
        W_POSITIVE = N  # NUMBER OF WORDS IN THE POSITIVE VOCABULARY: W_POSITIVE
        W_NEGATIVE = N  # NUMBER OF WORDS IN THE NEGATIVE VOCABULARY: W_NEGATIVE
        for I in range(M):  # FOR EACH SAMPLE: X[I]
            if Y[I] == 1:  # IF THE SAMPLE IS POSITIVE
                for J in range(N):  # FOR EACH WORD IN THE VOCABULARY
                    # INCREMENT THE NUMBER OF TIMES THE WORD APPEARS IN THE POSITIVE VOCABULARY: P_POSITIVE_J; HERE X_I_J DENOTES THE NUMBER OF TIMES THE WORD J APPEARS IN THE SAMPLE X[I].
                    self.P[0, J] += X[I][J]
                    # INCREMENT THE NUMBER OF WORDS IN THE POSITIVE VOCABULARY: W_POSITIVE
                    W_POSITIVE += X[I][J]
            else:  # IF THE SAMPLE IS NEGATIVE
                for J in range(N):  # FOR EACH WORD IN THE VOCABULARY
                    # INCREMENT THE NUMBER OF TIMES THE WORD APPEARS IN THE NEGATIVE VOCABULARY: P_NEGATIVE_J; HERE X_I_J DENOTES THE NUMBER OF TIMES THE WORD J APPEARS IN THE SAMPLE X[I].
                    self.P[1, J] += X[I][J]
                    # INCREMENT THE NUMBER OF WORDS IN THE NEGATIVE VOCABULARY: W_NEGATIVE
                    W_NEGATIVE += X[I][J]
        # COMPUTE THE PROBABILITIES, P IN R^{2 x N}:
        self.P[0, :] /= W_POSITIVE  # P_POSITIVE_J = P_POSITIVE_J / W_POSITIVE
        self.P[1, :] /= W_NEGATIVE  # P_NEGATIVE_J = P_NEGATIVE_J / W_NEGATIVE

    # TRANSFORM(): PREDICTS THE OUTPUT OF A SINGLE SAMPLE: X; SAMPLE IS A DICIONARY WHERE THE KEY IS THE WORD AND THE VALUE IS THE ABSOLUTE FREQUENCY (COUNT) OF THE WORD IN THE SAMPLE. RETURNS 1 IF POSITIVE, 0 IF NEGATIVE.
    def TRANSFORM(self, X):
        assert self.B is not None and self.P is not None, 'ERROR: THE MODEL HAS NOT BEEN TRAINED.'
        T = -self.B  # INITIALIZE THE SCORE THRESHOLD, T = -B.
        for J in range(len(X)):  # FOR EACH WORD IN THE VOCABULARY
            # INCREMENT THE SCORE THRESHOLD, T, BY THE LOG RATIO OF THE PROBABILITIES OF THE WORD J IN THE POSITIVE AND NEGATIVE VOCABULARIES.
            T += X[J] * np.log(self.P[0, J] / self.P[1, J])
        # IF THE SCORE THRESHOLD, T, IS GREATER THAN 0, THEN THE SAMPLE IS POSITIVE, OTHERWISE IT'S NEGATIVE.
        return 1 if T > 0 else -1