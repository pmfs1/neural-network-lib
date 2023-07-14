import numpy as np
from .CSP import CSP

def LATE_ACCEPTANCE_HILL_CLIMBING(CSP_PROBLEM: CSP, MAX_ITERATIONS: int, LAHC_MEMORY_LENGTH: int = 100) -> dict:
    
    def __GENERATE_RANDOM_STATE__() -> dict:
        STATE = {}
        for VARIABLE in CSP_PROBLEM.VARIABLES:
            VALUES = CSP_PROBLEM.DOMAINS[VARIABLE]
            RANDOM_VALUE = np.random.choice(VALUES)
            STATE[VARIABLE] = RANDOM_VALUE
        return STATE

    def __GET_RANDOM_INDEX__(ITERATION: int) -> int:
        MAX_INDEX = len(CSP_PROBLEM.VARIABLES) - 1
        if ITERATION < MAX_INDEX:
            return ITERATION
        return np.random.randint(0, MAX_INDEX)

    def __SELECT_VALUE__(STATE: dict, VARIABLE: str) -> str:
        VALUES = CSP_PROBLEM.DOMAINS[VARIABLE]
        RANDOM_VALUE = np.random.choice(VALUES)
        if RANDOM_VALUE != STATE[VARIABLE]:
            return RANDOM_VALUE
        return __SELECT_VALUE__(STATE, VARIABLE)

    def __COMPLETE__(STATE: dict) -> bool:
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if not __SATISFIED__(STATE, CONSTRAINT):
                return False
        return True

    def __SATISFIED__(STATE: dict, CONSTRAINT: tuple) -> bool:
        VARIABLE_1 = CONSTRAINT[0]
        VARIABLE_2 = CONSTRAINT[1]
        if STATE[VARIABLE_1] != STATE[VARIABLE_2]:
            return True
        return False

    def __BETTER__(CURRENT_STATE: dict, BEST_STATE: dict) -> bool:
        CURRENT_COST = __CALCULATE_COST__(CURRENT_STATE)
        BEST_COST = __CALCULATE_COST__(BEST_STATE)
        if CURRENT_COST < BEST_COST:
            return True
        return False

    def __CALCULATE_COST__(STATE: dict) -> int:
        COST = 0
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if not __SATISFIED__(STATE, CONSTRAINT):
                COST += 1
        return COST
    
    CURRENT_STATE = __GENERATE_RANDOM_STATE__()
    BEST_STATE = CURRENT_STATE
    LAHC_LIST = [BEST_STATE]
    for ITERATION in range(MAX_ITERATIONS):
        RANDOM_INDEX = __GET_RANDOM_INDEX__(ITERATION)
        RANDOM_VARIABLE = CSP_PROBLEM.VARIABLES[RANDOM_INDEX]
        RANDOM_VALUE = __SELECT_VALUE__(CURRENT_STATE, RANDOM_VARIABLE)
        CURRENT_STATE[RANDOM_VARIABLE] = RANDOM_VALUE
        if __COMPLETE__(CURRENT_STATE):
            return CURRENT_STATE
        if __BETTER__(CURRENT_STATE, BEST_STATE):
            BEST_STATE = CURRENT_STATE
        LAHC_LIST.append(BEST_STATE)
        if ITERATION >= LAHC_MEMORY_LENGTH:
            CURRENT_STATE = LAHC_LIST[ITERATION - LAHC_MEMORY_LENGTH]
    return BEST_STATE