from .CSP import CSP

def AC3(CSP_PROBLEM: CSP) -> bool:
    QUEUE = []
    for VARIABLE in CSP_PROBLEM.VARIABLES:
        for NEIGHBOUR in CSP_PROBLEM.NEIGHBOURS[VARIABLE]:
            QUEUE.append((VARIABLE, NEIGHBOUR))

    def __REVISE__(X_I: str, X_J: str) -> bool:
        REVISED = False
        for X in CSP_PROBLEM.DOMAINS[X_I]:
            SATISFIED = False
            for Y in CSP_PROBLEM.DOMAINS[X_J]:
                if X != Y:
                    SATISFIED = True
                    break
            if not SATISFIED:
                del CSP_PROBLEM.DOMAINS[X_I][CSP_PROBLEM.DOMAINS[X_I].index(X)]
                REVISED = True
        return REVISED

    while len(QUEUE) > 0:
        (X_I, X_J) = QUEUE.pop(0)
        if __REVISE__(X_I, X_J):
            if len(CSP_PROBLEM.DOMAINS[X_I]) == 0:
                return False
            for X_K in CSP_PROBLEM.NEIGHBOURS[X_I]:
                if X_K != X_J:
                    QUEUE.append((X_K, X_I))
    return True