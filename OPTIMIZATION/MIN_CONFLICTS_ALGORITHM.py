import numpy as np

from .CSP_PROBLEM import CSP

# MIN_CONFLICTS(): RUNS THE MIN-CONFLICTS ALGORITHM. RETURNS THE ASSIGNMENT.
def MIN_CONFLICTS(CSP: CSP, MAX_STEPS):
    # ASSIGNMENT, DICTIONARY, INITIALLY AN EMPTY DICTIONARY.
    ASSIGNMENT = {}
    # FOR EACH VARIABLE IN THE PROBLEM, DO THE FOLLOWING:
    for VARIABLE in CSP.VARIABLES:
        # ASSIGN A RANDOM VALUE TO THE VARIABLE.
        ASSIGNMENT[VARIABLE] = np.random.choice(CSP.DOMAINS[VARIABLE])
    # FOR EACH STEP IN THE MAXIMUM NUMBER OF STEPS, DO THE FOLLOWING:
    for _ in range(MAX_STEPS):
        # IF THE ASSIGNMENT IS COMPLETE, RETURN THE ASSIGNMENT.
        if __COMPLETE__(CSP, ASSIGNMENT):
            return ASSIGNMENT
        # VARIABLE, VARIABLE, THE VARIABLE TO BE ASSIGNED.
        VARIABLE = __SELECT_VARIABLE__(CSP, ASSIGNMENT)
        # VALUE, VALUE, THE VALUE TO BE ASSIGNED TO THE VARIABLE.
        VALUE = __SELECT_VALUE__(CSP, ASSIGNMENT, VARIABLE)
        # ASSIGN THE VALUE TO THE VARIABLE.
        ASSIGNMENT[VARIABLE] = VALUE
    return False  # RETURN FALSE.

# __SELECT_VARIABLE__(): [PRIVATE FUNCTION] RETURNS THE VARIABLE TO BE ASSIGNED.
def __SELECT_VARIABLE__(CSP: CSP, ASSIGNMENT):
    # VARIABLE, VARIABLE, THE VARIABLE TO BE ASSIGNED.
    VARIABLE = None
    # MINIMUM, INTEGER, THE MINIMUM NUMBER OF CONFLICTS.
    MINIMUM = float('inf')
    # FOR EACH VARIABLE IN THE PROBLEM, DO THE FOLLOWING:
    for _VARIABLE in CSP.VARIABLES:
        # IF THE NUMBER OF CONFLICTS IS LESS THAN THE MINIMUM, DO THE FOLLOWING:
        if __CONFLICTS__(CSP, ASSIGNMENT, _VARIABLE) < MINIMUM:
            # MINIMUM, INTEGER, THE MINIMUM NUMBER OF CONFLICTS.
            MINIMUM = __CONFLICTS__(CSP, ASSIGNMENT, _VARIABLE)
            # VARIABLE, VARIABLE, THE VARIABLE TO BE ASSIGNED.
            VARIABLE = _VARIABLE
    return VARIABLE  # RETURN THE VARIABLE.

# __SELECT_VALUE__(): [PRIVATE FUNCTION] RETURNS THE VALUE TO BE ASSIGNED TO THE VARIABLE.
def __SELECT_VALUE__(CSP: CSP, ASSIGNMENT, VARIABLE):
    # VALUE, VALUE, THE VALUE TO BE ASSIGNED TO THE VARIABLE.
    VALUE = None
    # MINIMUM, INTEGER, THE MINIMUM NUMBER OF CONFLICTS.
    MINIMUM = float('inf')
    # FOR EACH VALUE IN THE DOMAIN OF THE VARIABLE, DO THE FOLLOWING:
    for _VALUE in CSP.DOMAINS[VARIABLE]:
        # IF THE NUMBER OF CONFLICTS IS LESS THAN THE MINIMUM, DO THE FOLLOWING:
        if __CONFLICTS__(CSP, ASSIGNMENT, VARIABLE, _VALUE) < MINIMUM:
            # MINIMUM, INTEGER, THE MINIMUM NUMBER OF CONFLICTS.
            MINIMUM = __CONFLICTS__(CSP, ASSIGNMENT, VARIABLE, _VALUE)
            # VALUE, VALUE, THE VALUE TO BE ASSIGNED TO THE VARIABLE.
            VALUE = _VALUE
    return VALUE  # RETURN THE VALUE.

# __CONFLICTS__(): [PRIVATE FUNCTION] RETURNS THE NUMBER OF CONFLICTS OF THE VARIABLE.
def __CONFLICTS__(CSP: CSP, ASSIGNMENT, VARIABLE, VALUE=None):
    # CONFLICTS, INTEGER, THE NUMBER OF CONFLICTS OF THE VARIABLE.
    CONFLICTS = 0
    # FOR EACH CONSTRAINT IN THE PROBLEM, DO THE FOLLOWING:
    for CONSTRAINT in CSP.CONSTRAINTS:
        # IF THE VARIABLE IS INVOLVED IN THE CONSTRAINT, DO THE FOLLOWING:
        if VARIABLE in CONSTRAINT:
            VALUE_1 = None  # VALUE_1, VALUE, THE VALUE OF THE FIRST VARIABLE INVOLVED IN THE CONSTRAINT.
            VALUE_2 = None  # VALUE_2, VALUE, THE VALUE OF THE SECOND VARIABLE INVOLVED IN THE CONSTRAINT.
            # VARIABLE_1, VARIABLE, THE FIRST VARIABLE INVOLVED IN THE CONSTRAINT.
            VARIABLE_1 = CONSTRAINT[0]
            # VARIABLE_2, VARIABLE, THE SECOND VARIABLE INVOLVED IN THE CONSTRAINT.
            VARIABLE_2 = CONSTRAINT[1]
            # IF THE VARIABLE IS THE FIRST VARIABLE INVOLVED IN THE CONSTRAINT, DO THE FOLLOWING:
            if VARIABLE == VARIABLE_1:
                # VALUE_1, VALUE, THE VALUE OF THE FIRST VARIABLE INVOLVED IN THE CONSTRAINT.
                VALUE_1 = VALUE
                # VALUE_2, VALUE, THE VALUE OF THE SECOND VARIABLE INVOLVED IN THE CONSTRAINT.
                VALUE_2 = ASSIGNMENT[VARIABLE_2]
            # IF THE VARIABLE IS THE SECOND VARIABLE INVOLVED IN THE CONSTRAINT, DO THE FOLLOWING:
            elif VARIABLE == VARIABLE_2:
                # VALUE_1, VALUE, THE VALUE OF THE FIRST VARIABLE INVOLVED IN THE CONSTRAINT.
                VALUE_1 = ASSIGNMENT[VARIABLE_1]
                # VALUE_2, VALUE, THE VALUE OF THE SECOND VARIABLE INVOLVED IN THE CONSTRAINT.
                VALUE_2 = VALUE
            # IF THE VALUES ARE THE SAME, INCREMENT THE NUMBER OF CONFLICTS.
            if VALUE_1 == VALUE_2:
                CONFLICTS += 1
    return CONFLICTS  # RETURN THE NUMBER OF CONFLICTS.

# __COMPLETE__(): [PRIVATE FUNCTION] RETURNS TRUE IF THE ASSIGNMENT IS COMPLETE, OTHERWISE RETURNS FALSE.
def __COMPLETE__(CSP: CSP, ASSIGNMENT):
    # FOR EACH VARIABLE IN THE PROBLEM, DO THE FOLLOWING:
    for VARIABLE in CSP.VARIABLES:
        # IF THE VARIABLE IS NOT IN THE ASSIGNMENT, RETURN FALSE.
        if VARIABLE not in ASSIGNMENT:
            return False
    return True  # RETURN TRUE.