import math
import numpy as np
from .CSP import CSP

def SIMULATED_ANNEALING(CSP_PROBLEM: CSP, TEMPERATURE = 100, COOLING_RATE = 0.99) -> dict:

    def __COST_FUNCTION__(STATE: dict) -> int:
        COST = 0
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if not __CONSTRAINT_SATISFIED__(CONSTRAINT, STATE):
                COST += 1
        return COST

    def __CONSTRAINT_SATISFIED__(CONSTRAINT: tuple, STATE: dict) -> bool:
        return STATE[CONSTRAINT[0]] != STATE[CONSTRAINT[1]]

    CURRENT_STATE = {}
    for VARIABLE in CSP_PROBLEM.VARIABLES:
        CURRENT_STATE[VARIABLE] = np.random.choice(
            CSP_PROBLEM.DOMAINS[VARIABLE])
    while TEMPERATURE > 1:
        NEXT_STATE = {}
        for VARIABLE in CSP_PROBLEM.VARIABLES:
            NEXT_STATE[VARIABLE] = np.random.choice(
                CSP_PROBLEM.DOMAINS[VARIABLE])
        if __COST_FUNCTION__(NEXT_STATE) < __COST_FUNCTION__(CURRENT_STATE):
            CURRENT_STATE = NEXT_STATE
        else:
            PROBABILITY = math.exp(- (__COST_FUNCTION__(NEXT_STATE) - __COST_FUNCTION__(
                CURRENT_STATE)) / TEMPERATURE)
            if PROBABILITY > np.random.uniform(0, 1):
                CURRENT_STATE = NEXT_STATE
        TEMPERATURE *= COOLING_RATE
    return CURRENT_STATE