from .CSP import CSP

def TABU_SEARCH(CSP_PROBLEM: CSP) -> dict | None:

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

    def __BEST_SUCCESSOR__(STATE: dict, TABU_LIST: list) -> dict | None:
        BEST_SUCCESSOR = None
        for SUCCESSOR in __SUCCESSORS__(STATE):
            if SUCCESSOR not in TABU_LIST:
                if BEST_SUCCESSOR is None or __BETTER__(SUCCESSOR, BEST_SUCCESSOR):
                    BEST_SUCCESSOR = SUCCESSOR
        return BEST_SUCCESSOR

    def __SUCCESSORS__(STATE: dict) -> list:
        SUCCESSORS = []
        for VARIABLE in CSP_PROBLEM.VARIABLES:
            for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
                if STATE[VARIABLE] != VALUE:
                    SUCCESSOR = STATE.copy()
                    SUCCESSOR[VARIABLE] = VALUE
                    SUCCESSORS.append(SUCCESSOR)
        return SUCCESSORS

    def __BETTER__(SUCCESSOR: dict, BEST_SUCCESSOR: dict) -> bool:
        if __COST__(SUCCESSOR) < __COST__(BEST_SUCCESSOR):
            return True
        return False

    def __COST__(STATE: dict) -> int:
        COST = 0
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:
            if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
                COST += 1
        return COST

    CURRENT_STATE = __INITIAL_STATE__()
    TABU_LIST = []
    TABU_LIST_SIZE = len(CSP_PROBLEM.VARIABLES)
    while True:
        if __SOLUTION__(CURRENT_STATE):
            return CURRENT_STATE
        CURRENT_STATE = __BEST_SUCCESSOR__(
            CURRENT_STATE, TABU_LIST)
        if CURRENT_STATE is None:
            return None
        TABU_LIST.append(CURRENT_STATE)
        if len(TABU_LIST) > TABU_LIST_SIZE:
            TABU_LIST.pop(0)