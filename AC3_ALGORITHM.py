# AC3(): RUNS THE AC-3 ALGORITHM. RETURNS FALSE IF AN INCONSISTENCY IS FOUND, OTHERWISE RETURNS TRUE.
def AC3(CSP):
    QUEUE = [] # QUEUE, QUEUE OF ARCS, INITIALLY ALL THE ARCS IN THE PROBLEM.
    for VARIABLE in CSP.VARIABLES: # FOR EACH VARIABLE, FIND THE NEIGHBORS OF THE VARIABLE.
        for NEIGHBOR in CSP.NEIGHBORS[VARIABLE]: # FOR EACH NEIGHBOR OF THE VARIABLE, ADD THE ARC (VARIABLE, NEIGHBOR) TO THE QUEUE.
            QUEUE.append((VARIABLE, NEIGHBOR)) # ADD THE ARC (VARIABLE, NEIGHBOR) TO THE QUEUE.
    while len(QUEUE) > 0: # WHILE THE QUEUE IS NOT EMPTY, POP THE FIRST ARC (X_I, X_J) FROM THE QUEUE.
        (X_I, X_J) = QUEUE.pop(0) # POP THE FIRST ARC (X_I, X_J) FROM THE QUEUE.
        if __REVISE__(CSP, X_I, X_J): # IF THE DOMAINS OF THE VARIABLE X_I ARE REVISED, DO THE FOLLOWING:
            if len(CSP.DOMAINS[X_I]) == 0: # IF THE DOMAIN OF THE VARIABLE X_I IS EMPTY, RETURN FALSE.
                return False # RETURN FALSE.
            for X_K in CSP.NEIGHBORS[X_I]: # FOR EACH X_K IN NEIGHBOURS MINUS X_J, ADD THE (X_K, X_I) TO THE QUEUE.
                if X_K != X_J: # IF X_K IS DIFERENT THAN X_J, ADD THE (X_K, X_I) TO THE QUEUE.
                    QUEUE.append((X_K, X_I)) # ADD THE (X_K, X_I) TO THE QUEUE.
    return True # RETURNS TRUE.

# REVISE(): [PRIVATE FUNCTION] REVISES THE DOMAINS OF THE VARIABLE X_I, GIVEN THE DOMAINS OF THE VARIABLE X_J. RETURNS TRUE IF THE DOMAINS OF THE VARIABLE X_I ARE REVISED, OTHERWISE RETURNS FALSE.
def __REVISE__(CSP, X_I, X_J):
    REVISED = False # REVISED, BOOLEAN, INDICATES IF THE DOMAINS OF THE VARIABLE X_I ARE REVISED.
    for X in CSP.DOMAINS[X_I]: # FOR EACH VALUE X OF THE DOMAIN OF THE VARIABLE X_I, CHECK IF THE VALUE X SATISFIES THE CONSTRAINT (X_I, X_J).
        SATISFIED = False # SATISFIED, BOOLEAN, INDICATES IF THE VALUE X SATISFIES THE CONSTRAINT (X_I, X_J).
        for Y in CSP.DOMAINS[X_J]: # FOR EACH VALUE Y OF THE DOMAIN OF THE VARIABLE X_J, CHECK IF THE VALUE X SATISFIES THE CONSTRAINT (X_I, X_J).
            if X != Y: # IF THE VALUE X DOES NOT SATISFY THE CONSTRAINT (X_I, X_J), REMOVE THE VALUE X FROM THE DOMAIN OF THE VARIABLE X_I.
                SATISFIED = True # THE VALUE X SATISFIES THE CONSTRAINT (X_I, X_J).
                break # BREAK THE LOOP.
        if not SATISFIED: # IF THERE'S NO VALUE Y OF THE DOMAIN OF THE VARIABLE X_J THAT ALLOWS (X, Y) TO SATISFY THE CONSTRAINT BETWEEN (X_I, X_J), REMOVE THE VALUE X FROM THE DOMAIN OF THE VARIABLE X_I.
            del CSP.DOMAINS[X_I][CSP.DOMAINS[X_I].index(X)] # REMOVE THE VALUE X FROM THE DOMAIN OF THE VARIABLE X_I.
            REVISED = True # THE DOMAINS OF THE VARIABLE X_I ARE REVISED.
    return REVISED # RETURN THE BOOLEAN REVISED.