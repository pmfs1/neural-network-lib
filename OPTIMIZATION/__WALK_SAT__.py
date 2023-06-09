import numpy as np

from .CSP import CSP

# WALK_SAT(): RUNS THE WALK-SAT ALGORITHM.
def WALK_SAT(CSP_PROBLEM: CSP, MAX_FLIPS: int, MAX_RESTARTS: int):
    # ASSIGNMENT, DICTIONARY, INITIALLY AN EMPTY DICTIONARY.
    ASSIGNMENT = {}
    # INITIALIZES THE CLAUSES.
    CLAUSES = []
    # FOR EACH CONSTRAINT, DO THE FOLLOWING:
    for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
        # CLAUSE, LIST, INITIALLY AN EMPTY LIST.
        CLAUSE = []
        # FOR EACH VARIABLE IN THE CONSTRAINT, DO THE FOLLOWING:
        for VARIABLE in CONSTRAINT:
            # APPEND THE VARIABLE TO THE CLAUSE.
            CLAUSE.append(VARIABLE)
        # APPEND THE CLAUSE TO THE CLAUSES.
        CLAUSES.append(CLAUSE)
    # RETURN __WALK_SAT__(CSP_PROBLEM, ASSIGNMENT, MAX_FLIPS, MAX_RESTARTS).
    return __WALK_SAT__(CSP_PROBLEM, ASSIGNMENT, MAX_FLIPS, MAX_RESTARTS, CLAUSES)

# __WALK_SAT__() [PRIVATE FUNCTION]: RETURNS THE ASSIGNMENT. 
def __WALK_SAT__(CSP_PROBLEM: CSP, ASSIGNMENT, MAX_FLIPS: int, MAX_RESTARTS: int, CLAUSES: list):
    # FOR EACH VARIABLE IN THE PROBLEM, DO THE FOLLOWING:
    for VARIABLE in CSP_PROBLEM.VARIABLES:
        # ASSIGN THE VARIABLE TO A RANDOM VALUE.
        ASSIGNMENT[VARIABLE] = CSP_PROBLEM.DOMAINS[VARIABLE][np.random.randint(0, len(CSP_PROBLEM.DOMAINS[VARIABLE]) - 1)]
    # FOR EACH RESTART, DO THE FOLLOWING:
    for _ in range(MAX_RESTARTS):
        # FOR EACH FLIP, DO THE FOLLOWING:
        for _ in range(MAX_FLIPS):
            # IF THE ASSIGNMENT IS COMPLETE, RETURN THE ASSIGNMENT.
            if __COMPLETE__(CSP_PROBLEM, ASSIGNMENT):
                return ASSIGNMENT
            # CLAUSE, LIST, THE CLAUSE TO BE FLIPPED.
            CLAUSE = __SELECT_CLAUSE__(CSP_PROBLEM, ASSIGNMENT, CLAUSES)
            # VARIABLE, VARIABLE, THE VARIABLE TO BE FLIPPED.
            VARIABLE = __SELECT_VARIABLE__(ASSIGNMENT, CLAUSE)
            # VALUE, VALUE, THE VALUE TO BE ASSIGNED.
            VALUE = __SELECT_VALUE__(CSP_PROBLEM, ASSIGNMENT, VARIABLE, CLAUSE)
            # ASSIGN THE VALUE TO THE VARIABLE.
            ASSIGNMENT[VARIABLE] = VALUE
    return False  # RETURN FALSE.

# __SELECT_CLAUSE__(): [PRIVATE FUNCTION] RETURNS THE CLAUSE TO BE FLIPPED.
def __SELECT_CLAUSE__(CSP_PROBLEM: CSP, ASSIGNMENT, CLAUSES):
    # FOR EACH CLAUSE IN THE PROBLEM, DO THE FOLLOWING:
    for CLAUSE in CLAUSES:
        # IF THE CLAUSE IS NOT SATISFIED, RETURN THE CLAUSE.
        if not __SATISFIED__(CSP_PROBLEM, ASSIGNMENT, CLAUSE):
            return CLAUSE  # RETURN THE CLAUSE.
        
# __SELECT_VARIABLE__(): [PRIVATE FUNCTION] RETURNS THE VARIABLE TO BE FLIPPED.
def __SELECT_VARIABLE__(ASSIGNMENT, CLAUSE):
    # FOR EACH VARIABLE IN THE CLAUSE, DO THE FOLLOWING:
    for VARIABLE in CLAUSE:
        # IF THE VARIABLE IS NOT ASSIGNED, RETURN THE VARIABLE.
        if VARIABLE not in ASSIGNMENT:
            return VARIABLE  # RETURN THE VARIABLE.
    
# __SELECT_VALUE__(): [PRIVATE FUNCTION] RETURNS THE VALUE TO BE ASSIGNED.
def __SELECT_VALUE__(CSP_PROBLEM: CSP, ASSIGNMENT, VARIABLE, CLAUSE):
    # FOR EACH VALUE IN THE VARIABLE'S DOMAIN, DO THE FOLLOWING:
    for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
        # ASSIGN THE VALUE TO THE VARIABLE.
        ASSIGNMENT[VARIABLE] = VALUE
        # IF THE CLAUSE IS SATISFIED, RETURN THE VALUE.
        if __SATISFIED__(CSP_PROBLEM, ASSIGNMENT, CLAUSE):
            return VALUE  # RETURN THE VALUE.
    return ASSIGNMENT[VARIABLE]  # RETURN THE VARIABLE'S CURRENT VALUE.

# __SATISFIED__(): [PRIVATE FUNCTION] RETURNS TRUE IF THE CLAUSE IS SATISFIED, OTHERWISE RETURNS FALSE.
def __SATISFIED__(CSP_PROBLEM: CSP, ASSIGNMENT, CLAUSE):
    # FOR EACH VARIABLE IN THE CLAUSE, DO THE FOLLOWING:
    for VARIABLE in CLAUSE:
        # IF THE VARIABLE IS NOT ASSIGNED, RETURN FALSE.
        if VARIABLE not in ASSIGNMENT:
            return False
        # IF THE VARIABLE IS ASSIGNED AND THE ASSIGNMENT IS VALID, RETURN TRUE.
        if VARIABLE in ASSIGNMENT and ASSIGNMENT[VARIABLE] in CSP_PROBLEM.DOMAINS[VARIABLE]:
            return True
    return False  # RETURN FALSE.

# __COMPLETE__(): [PRIVATE FUNCTION] RETURNS TRUE IF THE ASSIGNMENT IS COMPLETE, OTHERWISE RETURNS FALSE.
def __COMPLETE__(CSP_PROBLEM: CSP, ASSIGNMENT):
    # FOR EACH VARIABLE IN THE PROBLEM, DO THE FOLLOWING:
    for VARIABLE in CSP_PROBLEM.VARIABLES:
        # IF THE VARIABLE IS NOT ASSIGNED, RETURN FALSE.
        if VARIABLE not in ASSIGNMENT:
            return False
    return True  # RETURN TRUE.