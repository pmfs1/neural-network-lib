from .CSP import CSP

# TABU_SEARCH(): RUNS THE TABU SEARCH ALGORITHM IN ORDER TO FIND A SOLUTION TO THE CONSTRAINT SATISFACTION PROBLEM. RETURNS THE CURRENT STATE IF A SOLUTION IS FOUND, OTHERWISE RETURNS FALSE.
def TABU_SEARCH(CSP_PROBLEM: CSP):
    # INITIALIZE THE CURRENT STATE TO THE INITIAL STATE.
    CURRENT_STATE = __INITIAL_STATE__(CSP_PROBLEM)
    # INITIALIZE THE TABU LIST.
    TABU_LIST = []
    # INITIALIZE THE TABU LIST SIZE, THE NUMBER OF STATES TO BE STORED IN THE TABU LIST, TO THE NUMBER OF VARIABLES IN THE PROBLEM. (BASED ON THE GIVEN CSP, THIS IS THE BEST VALUE FOR THE TABU LIST SIZE.)
    TABU_LIST_SIZE = len(CSP_PROBLEM.VARIABLES)
    # LOOP UNTIL A SOLUTION IS FOUND.
    while True:
        # IF THE CURRENT STATE IS A SOLUTION, THEN RETURN THE CURRENT STATE.
        if __SOLUTION__(CURRENT_STATE, CSP_PROBLEM):
            return CURRENT_STATE
        # FIND THE BEST SUCCESSOR OF THE CURRENT STATE.
        CURRENT_STATE = __BEST_SUCCESSOR__(CURRENT_STATE, CSP_PROBLEM, TABU_LIST)
        # ADD THE CURRENT STATE TO THE TABU LIST.
        TABU_LIST.append(CURRENT_STATE)
        # IF THE TABU LIST IS FULL, THEN REMOVE THE FIRST STATE FROM THE TABU LIST.
        if len(TABU_LIST) > TABU_LIST_SIZE:
            TABU_LIST.pop(0)
        
# __INITIAL_STATE__() [PRIVATE FUNCTION]: RETURNS THE INITIAL STATE OF THE CONSTRAINT SATISFACTION PROBLEM.
def __INITIAL_STATE__(CSP_PROBLEM: CSP):
    # INITIALIZE THE INITIAL STATE.
    INITIAL_STATE = {}
    # FOR EACH VARIABLE, FIND THE INITIAL VALUE OF THE VARIABLE.
    for VARIABLE in CSP_PROBLEM.VARIABLES:
        # SET THE INITIAL VALUE OF THE VARIABLE TO THE FIRST VALUE IN THE DOMAIN OF THE VARIABLE.
        INITIAL_STATE[VARIABLE] = CSP_PROBLEM.DOMAINS[VARIABLE][0]
    # RETURN THE INITIAL STATE.
    return INITIAL_STATE

# __SOLUTION__() [PRIVATE FUNCTION]: RETURNS TRUE IF THE STATE IS A SOLUTION TO THE CONSTRAINT SATISFACTION PROBLEM, OTHERWISE RETURNS FALSE.
def __SOLUTION__(STATE, CSP_PROBLEM: CSP):
    # FOR EACH CONSTRAINT, CHECK IF THE CONSTRAINT IS SATISFIED.
    for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
        # IF THE CONSTRAINT IS NOT SATISFIED, THEN RETURN FALSE.
        if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
            return False # THE CONSTRAINT IS NOT SATISFIED.
    # RETURN TRUE.
    return True

# __CONSTRAINT_SATISFIED__() [PRIVATE FUNCTION]: RETURNS TRUE IF THE CONSTRAINT IS SATISFIED, OTHERWISE RETURNS FALSE.
def __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
    # FOR EACH VARIABLE INVOLVED IN THE CONSTRAINT, CHECK IF THE PAIR OF VARIABLES HAVE THE SAME VALUE.
    for VARIABLE_1, VARIABLE_2 in CONSTRAINT:
        # IF THE PAIR OF VARIABLES HAVE THE SAME VALUE, THEN RETURN FALSE.
        if STATE[VARIABLE_1] == STATE[VARIABLE_2]:
            return False # THE CONSTRAINT IS NOT SATISFIED.
    # RETURN TRUE.
    return True

# __BEST_SUCCESSOR__() [PRIVATE FUNCTION]: RETURNS THE BEST SUCCESSOR OF THE STATE, OTHERWISE RETURNS FALSE.
def __BEST_SUCCESSOR__(STATE, CSP_PROBLEM: CSP, TABU_LIST):
    # INITIALIZE THE BEST SUCCESSOR.
    BEST_SUCCESSOR = False
    # FOR EACH SUCCESSOR OF THE STATE, FIND THE BEST SUCCESSOR.
    for SUCCESSOR in __SUCCESSORS__(STATE, CSP_PROBLEM):
        # IF THE SUCCESSOR IS NOT IN THE TABU LIST, THEN FIND THE BEST SUCCESSOR.
        if SUCCESSOR not in TABU_LIST:
            # IF THERE IS NO BEST SUCCESSOR, THEN SET THE SUCCESSOR AS THE BEST SUCCESSOR.
            if BEST_SUCCESSOR is False:
                BEST_SUCCESSOR = SUCCESSOR
            # IF THE SUCCESSOR IS BETTER THAN THE BEST SUCCESSOR, THEN SET THE SUCCESSOR AS THE BEST SUCCESSOR.
            elif __BETTER__(SUCCESSOR, BEST_SUCCESSOR, CSP_PROBLEM):
                BEST_SUCCESSOR = SUCCESSOR
    # RETURN THE BEST SUCCESSOR.
    return BEST_SUCCESSOR

# __SUCCESSORS__() [PRIVATE FUNCTION]: RETURNS THE SUCCESSORS OF THE STATE.
def __SUCCESSORS__(STATE, CSP_PROBLEM: CSP):
    # INITIALIZE THE SUCCESSORS.
    SUCCESSORS = []
    # FOR EACH VARIABLE, FIND THE SUCCESSORS OF THE STATE.
    for VARIABLE in CSP_PROBLEM.VARIABLES:
        # FOR EACH VALUE OF THE VARIABLE, FIND THE SUCCESSORS OF THE STATE.
        for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
            # IF THE VALUE IS NOT THE VALUE OF THE VARIABLE, THEN FIND THE SUCCESSORS OF THE STATE.
            if STATE[VARIABLE] != VALUE:
                # INITIALIZE THE SUCCESSOR.
                SUCCESSOR = STATE.copy()
                # SET THE VALUE OF THE VARIABLE TO THE VALUE.
                SUCCESSOR[VARIABLE] = VALUE
                # ADD THE SUCCESSOR TO THE LIST OF SUCCESSORS.
                SUCCESSORS.append(SUCCESSOR)
    # RETURN THE SUCCESSORS.
    return SUCCESSORS

# __BETTER__() [PRIVATE FUNCTION]: RETURNS TRUE IF THE SUCCESSOR IS BETTER THAN THE BEST SUCCESSOR, OTHERWISE RETURNS FALSE.
def __BETTER__(SUCCESSOR, BEST_SUCCESSOR, CSP_PROBLEM: CSP):
    # IF THE SUCCESSOR IS BETTER THAN THE BEST SUCCESSOR, THEN RETURN TRUE.
    if __COST__(SUCCESSOR, CSP_PROBLEM) < __COST__(BEST_SUCCESSOR, CSP_PROBLEM):
        return True # THE SUCCESSOR IS BETTER THAN THE BEST SUCCESSOR.
    # RETURN FALSE.
    return False

# __COST__() [PRIVATE FUNCTION]: RETURNS THE COST OF THE STATE.
def __COST__(STATE, CSP_PROBLEM: CSP):
    # INITIALIZE THE COST.
    COST = 0
    # FOR EACH CONSTRAINT, FIND THE COST OF THE STATE.
    for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
        # IF THE CONSTRAINT IS NOT SATISFIED, THEN INCREASE THE COST.
        if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
            COST += 1
    # RETURN THE COST.
    return COST