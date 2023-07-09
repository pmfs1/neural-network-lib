from .CSP import CSP

def HILL_CLIMBING(CSP_PROBLEM: CSP) -> dict | bool:

    def __INITIAL_STATE__() -> dict:
        INITIAL_STATE = {}
        for VARIABLE in CSP_PROBLEM.VARIABLES:
            INITIAL_STATE[VARIABLE] = CSP_PROBLEM.DOMAINS[VARIABLE][0]
        return INITIAL_STATE

    def __SOLUTION__(STATE: dict) -> bool:
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
                return False
        return True

    def __CONSTRAINT_SATISFIED__(STATE: dict, CONSTRAINT: tuple) -> bool:
        for VARIABLE_1, VARIABLE_2 in CONSTRAINT:
            if STATE[VARIABLE_1] == STATE[VARIABLE_2]:
                return False
        return True

    def __BEST_SUCCESSOR__(STATE: dict) -> dict | None:
        BEST_SUCCESSOR = None
        for SUCCESSOR in __SUCCESSORS__(STATE):
            if __BETTER__(SUCCESSOR, BEST_SUCCESSOR):
                BEST_SUCCESSOR = SUCCESSOR
        return BEST_SUCCESSOR

    def __SUCCESSORS__(STATE: dict) -> list:
        SUCCESSORS = []
        for VARIABLE in CSP_PROBLEM.VARIABLES:
            for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
                if VALUE != STATE[VARIABLE]:
                    SUCCESSOR = {}
                    for VARIABLE_2 in CSP_PROBLEM.VARIABLES:
                        SUCCESSOR[VARIABLE_2] = STATE[VARIABLE_2]
                    SUCCESSOR[VARIABLE] = VALUE
                    if __CONSISTENT__(SUCCESSOR):
                        SUCCESSORS.append(SUCCESSOR)
        return SUCCESSORS

    def __CONSISTENT__(SUCCESSOR: dict) -> bool:
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if not __CONSTRAINT_SATISFIED__(SUCCESSOR, CONSTRAINT):
                return False
        return True

    def __BETTER__(SUCCESSOR: dict, BEST_SUCCESSOR: dict | None) -> bool:
        if BEST_SUCCESSOR is None:
            return True
        if __CONSISTENT__(SUCCESSOR):
            return True
        if not __CONSISTENT__(BEST_SUCCESSOR):
            return False
        if __NUMBER_OF_VIOLATIONS__(SUCCESSOR) < __NUMBER_OF_VIOLATIONS__(BEST_SUCCESSOR):
            return True
        return False

    def __NUMBER_OF_VIOLATIONS__(STATE: dict) -> int:
        NUMBER_OF_VIOLATIONS = 0
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
                NUMBER_OF_VIOLATIONS += 1
        return NUMBER_OF_VIOLATIONS
    CURRENT_STATE = __INITIAL_STATE__()
    while True:
        if __SOLUTION__(CURRENT_STATE):
            return CURRENT_STATE
        SUCCESSOR = __BEST_SUCCESSOR__(CURRENT_STATE)
        if SUCCESSOR is None:
            return False
        CURRENT_STATE = SUCCESSOR