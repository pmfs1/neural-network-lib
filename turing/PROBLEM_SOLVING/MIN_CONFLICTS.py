import numpy as np
from .CSP import CSP

def MIN_CONFLICTS(CSP_PROBLEM: CSP, MAX_STEPS: int) -> dict | bool:

    def __SELECT_VARIABLE__(ASSIGNMENT: dict) -> str | None:
        VARIABLE = None
        MINIMUM = float('inf')
        for _VARIABLE in CSP_PROBLEM.VARIABLES:
            if __CONFLICTS__(ASSIGNMENT, _VARIABLE) < MINIMUM:
                MINIMUM = __CONFLICTS__(ASSIGNMENT, _VARIABLE)
                VARIABLE = _VARIABLE
        return VARIABLE

    def __SELECT_VALUE__(ASSIGNMENT: dict, VARIABLE: str) -> str | None:
        VALUE = None
        MINIMUM = float('inf')
        for _VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
            if __CONFLICTS__(ASSIGNMENT, VARIABLE, _VALUE) < MINIMUM:
                MINIMUM = __CONFLICTS__(ASSIGNMENT, VARIABLE, _VALUE)
                VALUE = _VALUE
        return VALUE

    def __CONFLICTS__(ASSIGNMENT: dict, VARIABLE: str, VALUE: str | None = None) -> int:
        CONFLICTS = 0
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if VARIABLE in CONSTRAINT:
                VALUE_1 = None
                VALUE_2 = None
                VARIABLE_1 = CONSTRAINT[0]
                VARIABLE_2 = CONSTRAINT[1]
                if VARIABLE == VARIABLE_1:
                    VALUE_1 = VALUE
                    VALUE_2 = ASSIGNMENT[VARIABLE_2]
                elif VARIABLE == VARIABLE_2:
                    VALUE_1 = ASSIGNMENT[VARIABLE_1]
                    VALUE_2 = VALUE
                if VALUE_1 == VALUE_2:
                    CONFLICTS += 1
        return CONFLICTS

    def __COMPLETE__(ASSIGNMENT: dict) -> bool:
        for VARIABLE in CSP_PROBLEM.VARIABLES:
            if VARIABLE not in ASSIGNMENT:
                return False
        return True

    ASSIGNMENT = {}
    for VARIABLE in CSP_PROBLEM.VARIABLES:
        ASSIGNMENT[VARIABLE] = np.random.choice(CSP_PROBLEM.DOMAINS[VARIABLE])
    for _ in range(MAX_STEPS):
        if __COMPLETE__(ASSIGNMENT):
            return ASSIGNMENT
        VARIABLE = __SELECT_VARIABLE__(ASSIGNMENT)
        if VARIABLE is None:
            return False
        VALUE = __SELECT_VALUE__(ASSIGNMENT, VARIABLE)
        ASSIGNMENT[VARIABLE] = VALUE
    return False