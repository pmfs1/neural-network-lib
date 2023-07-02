from collections import defaultdict
import numpy as np
from .BATCH_ITERATOR import BATCH_ITERATOR


class OPTIMIZER():
    """BASE CLASS FOR OPTIMIZERS"""

    def OPTIMIZE(self, NETWORK):
        """OPTIMIZATION PROCESS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        RETURN LOSS HISTORY
        """
        LOSS_HISTORY = []  # LOSS HISTORY LIST: STORES LOSS HISTORY
        for _ in range(NETWORK.MAX_EPOCHS):  # ITERATE OVER MAX_EPOCHS
            if NETWORK.SHUFFLE:  # IF SHUFFLE IS TRUE
                NETWORK.SUFFLE_DATASET()  # SHUFFLE DATASET
            LOSS = self.TRAIN_EPOCH(NETWORK)  # TRAIN EPOCH
            LOSS_HISTORY.append(LOSS)  # APPEND LOSS TO LOSS HISTORY
        return LOSS_HISTORY  # RETURN LOSS HISTORY

    def UPDATE(self, NETWORK):
        """UPDATE PARAMETERS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        raise NotImplementedError  # RAISE NOT IMPLEMENTED ERROR

    def TRAIN_EPOCH(self, NETWORK):
        """TRAIN EPOCH

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        RETURN EPOCH LOSS
        """
        LOSSES = []  # LOSS LIST: STORES LOSS
        # CREATE BATCH ITERATOR FOR X
        X_BATCH = BATCH_ITERATOR(NETWORK.X, NETWORK.BATCH_SIZE)
        # CREATE BATCH ITERATOR FOR Y
        Y_BATCH = BATCH_ITERATOR(NETWORK.Y, NETWORK.BATCH_SIZE)
        BATCH = zip(X_BATCH, Y_BATCH)  # ZIP X_BATCH AND Y_BATCH
        for X, Y in BATCH:  # FOR EACH X, Y IN BATCH # type: ignore
            LOSS = np.mean(NETWORK.UPDATE(X, Y))  # CALCULATE LOSS
            self.UPDATE(NETWORK)  # UPDATE NETWORK
            LOSSES.append(LOSS)  # APPEND LOSS TO LOSSES
        EPOCH_LOSS = np.mean(LOSSES)  # CALCULATE EPOCH LOSS
        return EPOCH_LOSS  # RETURN EPOCH LOSS

    def TRAIN_BATCH(self, NETWORK, X, Y):
        """TRAIN BATCH

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT
        X : ARRAY-LIKE
            FEATURE DATASET
        Y : ARRAY-LIKE
            TARGET DATASET

        RETURNS
        -------
        RETURN BATCH LOSS
        """
        LOSS = np.mean(NETWORK.UPDATE(X, Y))  # CALCULATE LOSS
        self.UPDATE(NETWORK)  # UPDATE NETWORK
        return LOSS  # RETURN LOSS

    def SETUP(self, NETWORK):
        """SETUP OPTIMIZER

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        raise NotImplementedError  # RAISE NOT IMPLEMENTED ERROR


class STOCHASTIC_GRADIENT_DESCENT(OPTIMIZER):
    """STOCHASTIC GRADIENT DESCENT OPTIMIZER"""

    def __init__(self, LEARNING_RATE=0.01, MOMENTUM=0.9, DECAY=0.0, NESTEROV=False):
        """INITIALIZE SGD OPTIMIZER

        PARAMETERS
        ----------
        LEARNING_RATE : FLOAT, OPTIONAL (DEFAULT=0.01)
            LEARNING RATE
        MOMENTUM : FLOAT, OPTIONAL (DEFAULT=0.9)
            MOMENTUM
        DECAY : FLOAT, OPTIONAL (DEFAULT=0.0)
            DECAY
        NESTEROV : BOOL, OPTIONAL (DEFAULT=False)
            NESTEROV

        RETURNS
        -------
        NONE
        """
        self.NESTEROV = NESTEROV  # SET NESTEROV
        self.DECAY = DECAY  # SET DECAY
        self.MOMENTUM = MOMENTUM  # SET MOMENTUM
        self.LEARNING_RATE = LEARNING_RATE  # SET LEARNING RATE
        self.ITERATION = 0  # SET ITERATION TO 0
        self.VELOCITY = None  # SET VELOCITY TO NONE

    def SETUP(self, NETWORK):
        """SETUP OPTIMIZER

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        self.VELOCITY = defaultdict(dict)  # SET VELOCITY TO DEFAULTDICT
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                self.VELOCITY[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET VELOCITY TO ZERO

    def UPDATE(self, NETWORK):
        """UPDATE PARAMETERS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        assert self.VELOCITY is not None, "CALL SETUP() BEFORE UPDATE()"  # ASSERT VELOCITY IS NOT NONE
        LEARNING_RATE = self.LEARNING_RATE * \
            (1.0 / (1.0 + self.DECAY * self.ITERATION))  # CALCULATE LEARNING RATE
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                GRAD = LAYER.PARAMETERS.GRAD[n]  # GET GRADIENT
                UPDATE = self.MOMENTUM * \
                    self.VELOCITY[i][n] - LEARNING_RATE * \
                    GRAD  # CALCULATE UPDATE
                self.VELOCITY[i][n] = UPDATE  # UPDATE VELOCITY
                if self.NESTEROV:  # IF NESTEROV IS TRUE
                    UPDATE = self.MOMENTUM * \
                        self.VELOCITY[i][n] - LEARNING_RATE * \
                        GRAD  # CALCULATE UPDATE
                LAYER.PARAMETERS.STEP(n, UPDATE)  # UPDATE PARAMETER
        self.ITERATION += 1  # INCREMENT ITERATION


class ADA_GRAD(OPTIMIZER):
    """ADA_GRAD OPTIMIZER"""

    def __init__(self, LEARNING_RATE=0.01, EPSILON=1e-8):
        """INITIALIZE ADA_GRAD OPTIMIZER

        PARAMETERS
        ----------
        LEARNING_RATE : FLOAT, OPTIONAL (DEFAULT=0.01)
            LEARNING RATE
        EPSILON : FLOAT, OPTIONAL (DEFAULT=1e-8)
            EPSILON

        RETURNS
        -------
        NONE
        """
        self.EPSILON = EPSILON  # SET EPSILON
        self.LEARNING_RATE = LEARNING_RATE  # SET LEARNING RATE
        self.ACCUMULATOR = None  # SET ACCUMULATOR TO NONE
        self.DELTA_ACCUMULATOR = None  # SET DELTA_ACCUMULATOR TO NONE
        self.MS = None  # SET MS TO NONE
        self.VS = None  # SET VS TO NONE
        self.US = None  # SET US TO NONE

    def UPDATE(self, NETWORK):
        """UPDATE PARAMETERS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        assert self.ACCUMULATOR is not None, "CALL SETUP() BEFORE UPDATE()"  # ASSERT ACCUMULATOR IS NOT NONE
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                GRAD = LAYER.PARAMETERS.GRAD[n]  # GET GRADIENT
                self.ACCUMULATOR[i][n] += GRAD ** 2  # UPDATE ACCUMULATOR
                STEP = self.LEARNING_RATE * GRAD / \
                    (np.sqrt(self.ACCUMULATOR[i][n]) +
                     self.EPSILON)  # CALCULATE STEP
                LAYER.PARAMETERS.STEP(n, -STEP)  # UPDATE PARAMETER

    def SETUP(self, NETWORK):
        """SETUP OPTIMIZER

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        self.ACCUMULATOR = defaultdict(dict)  # SET ACCUMULATOR TO DEFAULTDICT
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                self.ACCUMULATOR[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET ACCUMULATOR TO ZERO


class ADA_DELTA(OPTIMIZER):
    """ADA_DELTA OPTIMIZER"""

    def __init__(self, LEARNING_RATE=1.0, RHO=0.95, EPSILON=1e-8):
        """INITIALIZE ADA_DELTA OPTIMIZER

        PARAMETERS
        ----------
        LEARNING_RATE : FLOAT, OPTIONAL (DEFAULT=1.0)
            LEARNING RATE
        RHO : FLOAT, OPTIONAL (DEFAULT=0.95)
            RHO
        EPSILON : FLOAT, OPTIONAL (DEFAULT=1e-8)
            EPSILON

        RETURNS
        -------
        NONE
        """
        self.RHO = RHO  # SET RHO
        self.EPSILON = EPSILON  # SET EPSILON
        self.LEARNING_RATE = LEARNING_RATE  # SET LEARNING RATE
        self.ACCUMULATOR = None  # SET ACCUMULATOR TO NONE
        self.DELTA_ACCUMULATOR = None  # SET DELTA_ACCUMULATOR TO NONE
        self.MS = None  # SET MS TO NONE
        self.VS = None  # SET VS TO NONE
        self.US = None  # SET US TO NONE

    def UPDATE(self, NETWORK):
        """UPDATE PARAMETERS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        assert self.ACCUMULATOR is not None, "CALL SETUP() BEFORE UPDATE()"  # ASSERT ACCUMULATOR IS NOT NONE
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                GRAD = LAYER.PARAMETERS.GRAD[n]  # GET GRADIENT
                self.ACCUMULATOR[i][n] = self.RHO * self.ACCUMULATOR[i][n] + \
                    (1.0 - self.RHO) * GRAD ** 2  # UPDATE ACCUMULATOR
                STEP = GRAD * np.sqrt(self.DELTA_ACCUMULATOR[i][n] + self.EPSILON) / np.sqrt(
                    self.ACCUMULATOR[i][n] + self.EPSILON)  # CALCULATE STEP
                LAYER.PARAMETERS.STEP(
                    n, -STEP * self.LEARNING_RATE)  # UPDATE PARAMETER
                self.DELTA_ACCUMULATOR[i][n] = self.RHO * \
                    self.DELTA_ACCUMULATOR[i][n] + (1.0 - self.RHO) * STEP ** 2

    def SETUP(self, NETWORK):
        """SETUP OPTIMIZER

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        self.ACCUMULATOR = defaultdict(dict)  # SET ACCUMULATOR TO DEFAULTDICT
        # SET DELTA_ACCUMULATOR TO DEFAULTDICT
        self.DELTA_ACCUMULATOR = defaultdict(dict)
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                self.ACCUMULATOR[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET ACCUMULATOR TO ZERO
                self.DELTA_ACCUMULATOR[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET DELTA_ACCUMULATOR TO ZERO


class RMS_PROP(OPTIMIZER):
    """RMS_PROP OPTIMIZER"""

    def __init__(self, LEARNING_RATE=0.001, RHO=0.9, EPSILON=1e-8):
        """INITIALIZE RMS_PROP OPTIMIZER

        PARAMETERS
        ----------
        LEARNING_RATE : FLOAT, OPTIONAL (DEFAULT=0.001)
            LEARNING RATE
        RHO : FLOAT, OPTIONAL (DEFAULT=0.9)
            RHO
        EPSILON : FLOAT, OPTIONAL (DEFAULT=1e-8)
            EPSILON

        RETURNS
        -------
        NONE
        """
        self.EPSILON = EPSILON  # SET EPSILON
        self.RHO = RHO  # SET RHO
        self.LEARNING_RATE = LEARNING_RATE  # SET LEARNING RATE
        self.ACCUMULATOR = None  # SET ACCUMULATOR TO NONE
        self.MS = None  # SET MS TO NONE
        self.VS = None  # SET VS TO NONE
        self.US = None  # SET US TO NONE

    def UPDATE(self, NETWORK):
        """UPDATE PARAMETERS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        assert hasattr(
            self, 'ACCUMULATOR'), "ERROR: SETUP HAS NOT BEEN CALLED"  # ENSURE SETUP HAS BEEN CALLED
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                GRAD = LAYER.PARAMETERS.GRAD[n]  # GET GRADIENT
                self.ACCUMULATOR[i][n] = (
                    self.RHO * self.ACCUMULATOR[i][n]) + (1.0 - self.RHO) * (GRAD ** 2)  # UPDATE ACCUMULATOR
                STEP = self.LEARNING_RATE * GRAD / \
                    (np.sqrt(self.ACCUMULATOR[i][n]) +
                     self.EPSILON)  # CALCULATE STEP
                LAYER.PARAMETERS.STEP(n, -STEP)  # UPDATE PARAMETER

    def SETUP(self, NETWORK):
        """SETUP OPTIMIZER

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        self.ACCUMULATOR = defaultdict(dict)  # SET ACCUMULATOR TO DEFAULTDICT
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                self.ACCUMULATOR[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET ACCUMULATOR TO ZERO


class ADMA(OPTIMIZER):
    def __init__(self, LEARNING_RATE=0.001, FIRST_BETA=0.9, SECOND_BETA=0.999, EPSILON=1e-8):
        """INITIALIZE ADAM OPTIMIZER

        PARAMETERS
        ----------
        LEARNING_RATE : FLOAT, OPTIONAL (DEFAULT=0.001)
            LEARNING RATE
        FIRST_BETA : FLOAT, OPTIONAL (DEFAULT=0.9)
            FIRST BETA
        SECOND_BETA : FLOAT, OPTIONAL (DEFAULT=0.999)
            SECOND BETA
        EPSILON : FLOAT, OPTIONAL (DEFAULT=1e-8)
            EPSILON

        RETURNS
        -------
        NONE
        """
        self.EPSILON = EPSILON  # SET EPSILON
        self.SECOND_BETA = SECOND_BETA  # SET SECOND BETA
        self.FIRST_BETA = FIRST_BETA  # SET FIRST BETA
        self.LEARNING_RATE = LEARNING_RATE  # SET LEARNING RATE
        self.ITERATIONs = 0  # SET ITERATION TO 0
        self.T = 1  # SET T TO 1
        self.MS = None  # SET MS TO NONE
        self.VS = None  # SET VS TO NONE

    def UPDATE(self, NETWORK):
        """UPDATE PARAMETERS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        assert hasattr(
            self, "MS"), "SETUP HAS NOT BEEN CALLED"  # ASSERT SETUP HAS BEEN CALLED
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                GRAD = LAYER.PARAMETERS.GRAD[n]  # GET GRADIENT
                # UPDATE FIRST MOMENT ESTIMATE
                self.MS[i][n] = (self.FIRST_BETA * self.MS[i]
                                 [n]) + (1.0 - self.FIRST_BETA) * GRAD
                # UPDATE SECOND MOMENT ESTIMATE
                self.VS[i][n] = (self.SECOND_BETA * self.VS[i]
                                 [n]) + (1.0 - self.SECOND_BETA) * GRAD ** 2
                LEARNING_RATE = self.LEARNING_RATE * np.sqrt(1.0 - self.SECOND_BETA ** self.T) / (
                    1.0 - self.FIRST_BETA ** self.T)  # CALCULATE LEARNING RATE
                STEP = LEARNING_RATE * \
                    self.MS[i][n] / (np.sqrt(self.VS[i][n]) +
                                     self.EPSILON)  # CALCULATE STEP
                LAYER.PARAMETERS.STEP(n, -STEP)  # UPDATE PARAMETER
        self.T += 1  # INCREMENT T

    def SETUP(self, NETWORK):
        """SETUP OPTIMIZER

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        self.MS = defaultdict(dict)  # SET MS TO DEFAULTDICT
        self.VS = defaultdict(dict)  # SET VS TO DEFAULTDICT
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                self.MS[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET MS TO ZERO
                self.VS[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET VS TO ZERO


class ADA_MAX(OPTIMIZER):
    """ADA_MAX OPTIMIZER"""

    def __init__(self, LEARNING_RATE=0.002, FIRST_BETA=0.9, SECOND_BETA=0.999, EPSILON=1e-8):
        """INITIALIZE ADA_MAX OPTIMIZER

        PARAMETERS
        ----------
        LEARNING_RATE : FLOAT, OPTIONAL (DEFAULT=0.002)
            LEARNING RATE
        FIRST_BETA : FLOAT, OPTIONAL (DEFAULT=0.9)
            FIRST BETA
        SECOND_BETA : FLOAT, OPTIONAL (DEFAULT=0.999)
            SECOND BETA
        EPSILON : FLOAT, OPTIONAL (DEFAULT=1e-8)
            EPSILON

        RETURNS
        -------
        NONE
        """
        self.EPSILON = EPSILON  # SET EPSILON
        self.SECOND_BETA = SECOND_BETA  # SET SECOND BETA
        self.FIRST_BETA = FIRST_BETA  # SET FIRST BETA
        self.LEARNING_RATE = LEARNING_RATE  # SET LEARNING RATE
        self.T = 1  # SET T TO 1
        self.MS = None  # SET MS TO NONE
        self.US = None  # SET US TO NONE

    def UPDATE(self, NETWORK):
        """UPDATE PARAMETERS

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        assert hasattr(
            self, 'MS'), "ERROR: SETUP HAS NOT BEEN CALLED"  # ENSURE SETUP HAS BEEN CALLED
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                GRAD = LAYER.PARAMETERS.GRAD[n]  # GET GRADIENT
                self.MS[i][n] = self.FIRST_BETA * self.MS[i][n] + \
                    (1.0 - self.FIRST_BETA) * \
                    GRAD  # UPDATE FIRST MOMENT ESTIMATE
                # UPDATE SECOND MOMENT ESTIMATE
                self.US[i][n] = np.maximum(
                    self.SECOND_BETA * self.US[i][n], np.abs(GRAD))
                STEP = self.LEARNING_RATE / (1 - self.FIRST_BETA ** self.T) * self.MS[i][n] / (
                    self.US[i][n] + self.EPSILON)  # CALCULATE STEP
                LAYER.PARAMETERS.STEP(n, -STEP)  # UPDATE PARAMETER
        self.T += 1  # INCREMENT T

    def SETUP(self, NETWORK):
        """SETUP OPTIMIZER

        PARAMETERS
        ----------
        NETWORK : OBJECT
            NEURAL NETWORK OBJECT

        RETURNS
        -------
        NONE
        """
        self.MS = defaultdict(dict)  # SET MS TO DEFAULTDICT
        self.US = defaultdict(dict)  # SET US TO DEFAULTDICT
        # FOR EACH LAYER IN NETWORK.PARAMETRIC_LAYERS
        for i, LAYER in enumerate(NETWORK.PARAMETRIC_LAYERS):
            for n in LAYER.PARAMETERS.KEYS():  # FOR EACH PARAMETER IN LAYER.PARAMETERS
                self.MS[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET MS TO ZERO
                self.US[i][n] = np.zeros_like(
                    LAYER.PARAMETERS[n])  # SET US TO ZERO
