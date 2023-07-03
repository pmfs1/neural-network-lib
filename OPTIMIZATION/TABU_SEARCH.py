from .CSP import CSP

# THE `TABU_SEARCH()` FUNCTION IS THE MAIN FUNCTION THAT IMPLEMENTS THE TABU SEARCH ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS).
#     1. `CURRENT_STATE = __INITIAL_STATE__()`: THIS LINE INITIALIZES THE CURRENT STATE BY CALLING THE `__INITIAL_STATE__` FUNCTION, WHICH CREATES A DICTIONARY REPRESENTING THE INITIAL ASSIGNMENT OF VARIABLES.
#     2. `TABU_LIST = []`: AN EMPTY LIST IS CREATED TO STORE THE TABU LIST, WHICH WILL KEEP TRACK OF RECENTLY VISITED STATES.
#     3. `TABU_LIST_SIZE = LEN(CSP_PROBLEM.VARIABLES)`: THE `TABU_LIST_SIZE` VARIABLE IS SET TO THE NUMBER OF VARIABLES IN THE CSP PROBLEM. THIS DETERMINES THE MAXIMUM SIZE OF THE TABU LIST.
#     4. `WHILE TRUE:`: THE ALGORITHM ENTERS AN INFINITE LOOP, WHICH WILL BE TERMINATED WHEN A SOLUTION IS FOUND OR NO MORE VALID SUCCESSORS CAN BE GENERATED.
#     5. `IF __SOLUTION__(CURRENT_STATE):`: THIS CONDITION CHECKS IF THE CURRENT STATE IS A SOLUTION BY CALLING THE `__SOLUTION__` FUNCTION, WHICH VERIFIES IF ALL CONSTRAINTS IN THE CSP PROBLEM ARE SATISFIED. IF A SOLUTION IS FOUND, THE CURRENT STATE IS RETURNED.
#     6. `CURRENT_STATE = __BEST_SUCCESSOR__(CURRENT_STATE, TABU_LIST)`: IF THE CURRENT STATE IS NOT A SOLUTION, THE ALGORITHM GENERATES THE BEST SUCCESSOR STATE BY CALLING THE `__BEST_SUCCESSOR__` FUNCTION. THIS FUNCTION EXAMINES ALL POSSIBLE SUCCESSORS OF THE CURRENT STATE, EXCLUDING THOSE IN THE TABU LIST, AND SELECTS THE BEST ONE BASED ON A COST FUNCTION.
#     7. `IF CURRENT_STATE IS NONE:`: THIS CONDITION CHECKS IF A VALID SUCCESSOR STATE WAS FOUND. IF `CURRENT_STATE` IS `NONE`, IT MEANS THAT NO VALID SUCCESSORS WERE GENERATED. IN SUCH CASES, THE ALGORITHM RETURNS `NONE` TO INDICATE THAT NO SOLUTION IS POSSIBLE.
#     8. `TABU_LIST.APPEND(CURRENT_STATE)`: THE CURRENT STATE IS ADDED TO THE TABU LIST.
#     9. `IF LEN(TABU_LIST) > TABU_LIST_SIZE:`: THIS CONDITION CHECKS IF THE TABU LIST HAS EXCEEDED ITS MAXIMUM SIZE.
#     10. `TABU_LIST.POP(0)`: IF THE TABU LIST HAS EXCEEDED ITS MAXIMUM SIZE, THE OLDEST ENTRY IN THE LIST (AT INDEX 0) IS REMOVED USING THE `POP()` METHOD. THIS ENSURES THAT THE TABU LIST MAINTAINS A FIXED SIZE.
#     11. THE LOOP CONTINUES FROM STEP 4 UNTIL A SOLUTION IS FOUND OR NO VALID SUCCESSORS CAN BE GENERATED.
# IN SUMMARY, THE `TABU_SEARCH()` FUNCTION IMPLEMENTS THE MAIN LOOP OF THE TABU SEARCH ALGORITHM. IT INITIALIZES THE CURRENT STATE, MAINTAINS THE TABU LIST, GENERATES SUCCESSOR STATES, SELECTS THE BEST SUCCESSOR, AND UPDATES THE TABU LIST. THE ALGORITHM CONTINUES ITERATING UNTIL A SOLUTION IS FOUND OR A TERMINATION CONDITION IS MET.
def TABU_SEARCH(CSP_PROBLEM: CSP) -> dict | None:
    """THE `TABU_SEARCH()` FUNCTION IMPLEMENTS THE TABU SEARCH ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS)."""

    # THE `__INITIAL_STATE__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO INITIALIZE THE INITIAL STATE FOR SOLVING A CONSTRAINT SATISFACTION PROBLEM (CSP).
    #     1. `INITIAL_STATE = {}`: AN EMPTY DICTIONARY NAMED `INITIAL_STATE` IS CREATED. THIS DICTIONARY WILL STORE THE INITIAL ASSIGNMENT OF VARIABLES IN THE CSP PROBLEM.
    #     2. `FOR VARIABLE IN CSP_PROBLEM.VARIABLES:`: THIS LINE STARTS A LOOP THAT ITERATES OVER EACH VARIABLE IN THE CSP PROBLEM. `CSP_PROBLEM.VARIABLES` IS A LIST OF ALL THE VARIABLES IN THE PROBLEM.
    #     3. `INITIAL_STATE[VARIABLE] = CSP_PROBLEM.DOMAINS[VARIABLE][0]`: WITHIN THE LOOP, THE INITIAL ASSIGNMENT FOR EACH VARIABLE IS SET. IT ASSIGNS THE FIRST VALUE FROM THE VARIABLE'S DOMAIN TO THE VARIABLE IN THE `INITIAL_STATE` DICTIONARY. `CSP_PROBLEM.DOMAINS[VARIABLE]` IS A LIST OF ALL THE POSSIBLE VALUES (DOMAIN) FOR THE VARIABLE.
    #     4. THE LOOP CONTINUES UNTIL ALL VARIABLES HAVE BEEN ASSIGNED INITIAL VALUES.
    #     5. FINALLY, THE `INITIAL_STATE` DICTIONARY IS RETURNED AS THE RESULT OF THE FUNCTION.
    # IN SUMMARY, THE `__INITIAL_STATE__()` FUNCTION TAKES A CSP PROBLEM AS INPUT AND INITIALIZES THE INITIAL STATE FOR THE TABU SEARCH ALGORITHM. IT CREATES A DICTIONARY WHERE EACH VARIABLE IS ASSIGNED THE FIRST VALUE FROM ITS CORRESPONDING DOMAIN. THIS INITIAL STATE SERVES AS THE STARTING POINT FOR THE TABU SEARCH ALGORITHM TO EXPLORE THE SOLUTION SPACE AND FIND A VALID SOLUTION TO THE CSP PROBLEM.
    def __INITIAL_STATE__() -> dict:
        """THE `__INITIAL_STATE__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO INITIALIZE THE INITIAL STATE FOR SOLVING A CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        INITIAL_STATE = {}  # INITIALIZE INITIAL STATE
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # FOR EACH VARIABLE IN CSP PROBLEM
            # ASSIGN FIRST VALUE FROM VARIABLE'S DOMAIN TO VARIABLE IN INITIAL STATE
            INITIAL_STATE[VARIABLE] = CSP_PROBLEM.DOMAINS[VARIABLE][0]
        return INITIAL_STATE  # RETURN INITIAL STATE

    # THE `__SOLUTION__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO DETERMINE WHETHER A GIVEN STATE SATISFIES ALL THE CONSTRAINTS OF A CONSTRAINT SATISFACTION PROBLEM (CSP).
    #     1. `FOR CONSTRAINT IN CSP_PROBLEM.CONSTRAINTS:`: THIS LINE STARTS A LOOP THAT ITERATES OVER EACH CONSTRAINT IN THE CSP PROBLEM. `CSP_PROBLEM.CONSTRAINTS` IS A LIST OF ALL THE CONSTRAINTS IN THE PROBLEM.
    #     2. `IF NOT __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):`: WITHIN THE LOOP, THIS CONDITION CHECKS IF THE GIVEN STATE VIOLATES THE CURRENT CONSTRAINT BY CALLING THE `__CONSTRAINT_SATISFIED__()` FUNCTION. IT PASSES THE STATE DICTIONARY AND THE CURRENT CONSTRAINT TUPLE AS ARGUMENTS.
    #     3. `RETURN FALSE`: IF THE CURRENT CONSTRAINT IS NOT SATISFIED (I.E., THE CONDITION IN STEP 2 IS MET), THE FUNCTION IMMEDIATELY RETURNS `FALSE`, INDICATING THAT THE STATE DOES NOT SATISFY ALL THE CONSTRAINTS.
    #     4. THE LOOP CONTINUES UNTIL ALL CONSTRAINTS HAVE BEEN CHECKED.
    #     5. IF THE LOOP COMPLETES WITHOUT ENCOUNTERING ANY VIOLATED CONSTRAINTS, IT MEANS THAT THE STATE SATISFIES ALL THE CONSTRAINTS.
    #     6. `RETURN TRUE`: IN THIS CASE, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE GIVEN STATE IS A VALID SOLUTION THAT SATISFIES ALL THE CONSTRAINTS OF THE CSP PROBLEM.
    # IN SUMMARY, THE `__SOLUTION__()` FUNCTION CHECKS WHETHER A GIVEN STATE SATISFIES ALL THE CONSTRAINTS IN A CSP PROBLEM. IT ITERATES OVER EACH CONSTRAINT AND, IF ANY OF THE CONSTRAINTS ARE VIOLATED, IT RETURNS `FALSE`. IF ALL CONSTRAINTS ARE SATISFIED, IT RETURNS `TRUE`, INDICATING THAT THE STATE IS A VALID SOLUTION. THIS FUNCTION IS USED WITHIN THE TABU SEARCH ALGORITHM TO DETERMINE IF THE CURRENT STATE IS A SOLUTION OR NOT.
    def __SOLUTION__(STATE: dict) -> bool:
        """THE `__SOLUTION__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO DETERMINE WHETHER A GIVEN STATE SATISFIES ALL THE CONSTRAINTS OF A CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # FOR EACH CONSTRAINT IN CSP PROBLEM
            # IF CONSTRAINT IS NOT SATISFIED
            if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
                return False  # RETURN FALSE
        return True  # RETURN TRUE

    # THE `__CONSTRAINT_SATISFIED__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO CHECK WHETHER A GIVEN STATE SATISFIES A SPECIFIC CONSTRAINT IN A CONSTRAINT SATISFACTION PROBLEM (CSP).
    #     1. `FOR VARIABLE_1, VARIABLE_2 IN CONSTRAINT:`: THIS LINE STARTS A LOOP THAT ITERATES OVER EACH VARIABLE PAIR IN THE GIVEN CONSTRAINT. `CONSTRAINT` IS A TUPLE THAT CONTAINS VARIABLE PAIRS.
    #     2. `IF STATE[VARIABLE_1] == STATE[VARIABLE_2]:`: WITHIN THE LOOP, THIS CONDITION CHECKS IF THE VALUES ASSIGNED TO THE VARIABLES IN THE GIVEN STATE VIOLATE THE CONSTRAINT. IT COMPARES THE VALUES ASSIGNED TO `VARIABLE_1` AND `VARIABLE_2` IN THE STATE DICTIONARY `STATE`.
    #     3. `RETURN FALSE`: IF THE ASSIGNED VALUES VIOLATE THE CONSTRAINT (I.E., THE CONDITION IN STEP 2 IS MET), THE FUNCTION IMMEDIATELY RETURNS `FALSE`, INDICATING THAT THE CONSTRAINT IS NOT SATISFIED BY THE STATE.
    #     4. THE LOOP CONTINUES UNTIL ALL VARIABLE PAIRS IN THE CONSTRAINT HAVE BEEN CHECKED.
    #     5. IF THE LOOP COMPLETES WITHOUT ENCOUNTERING ANY VIOLATED CONSTRAINTS, IT MEANS THAT ALL VARIABLE PAIRS SATISFY THE CONSTRAINT.
    #     6. `RETURN TRUE`: IN THIS CASE, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE GIVEN STATE SATISFIES THE CONSTRAINT.
    # IN SUMMARY, THE `__CONSTRAINT_SATISFIED__()` FUNCTION CHECKS WHETHER A GIVEN STATE SATISFIES A SPECIFIC CONSTRAINT IN A CSP PROBLEM. IT ITERATES OVER EACH VARIABLE PAIR IN THE CONSTRAINT AND, IF ANY OF THE VARIABLE ASSIGNMENTS VIOLATE THE CONSTRAINT, IT RETURNS `FALSE`. IF ALL VARIABLE ASSIGNMENTS SATISFY THE CONSTRAINT, IT RETURNS `TRUE`, INDICATING THAT THE CONSTRAINT IS SATISFIED BY THE STATE. THIS FUNCTION IS USED WITHIN THE TABU SEARCH ALGORITHM TO DETERMINE IF A SPECIFIC CONSTRAINT IS SATISFIED BY A STATE OR NOT.
    def __CONSTRAINT_SATISFIED__(STATE: dict, CONSTRAINT: tuple) -> bool:
        """THE `__CONSTRAINT_SATISFIED__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO CHECK WHETHER A GIVEN STATE SATISFIES A SPECIFIC CONSTRAINT IN A CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        for VARIABLE_1, VARIABLE_2 in CONSTRAINT:  # FOR EACH VARIABLE PAIR IN CONSTRAINT
            # IF VARIABLE ASSIGNMENTS VIOLATE CONSTRAINT
            if STATE[VARIABLE_1] == STATE[VARIABLE_2]:
                return False  # RETURN FALSE
        return True  # RETURN TRUE

    # THE `__BEST_SUCCESSOR__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO SELECT THE BEST SUCCESSOR STATE FROM A SET OF GENERATED SUCCESSORS.
    #     1. `BEST_SUCCESSOR = NONE`: THIS LINE INITIALIZES THE `BEST_SUCCESSOR` VARIABLE TO `NONE`. THIS VARIABLE WILL STORE THE BEST SUCCESSOR STATE FOUND SO FAR.
    #     2. `FOR SUCCESSOR IN __SUCCESSORS__(STATE):`: THIS LINE STARTS A LOOP THAT ITERATES OVER EACH SUCCESSOR STATE GENERATED FROM THE CURRENT STATE. IT CALLS THE `__SUCCESSORS__()` FUNCTION TO GENERATE THE SUCCESSORS, PASSING THE CURRENT STATE `STATE` AND THE CSP PROBLEM `CSP_PROBLEM` AS ARGUMENTS.
    #     3. `IF SUCCESSOR NOT IN TABU_LIST:`: WITHIN THE LOOP, THIS CONDITION CHECKS IF THE CURRENT SUCCESSOR STATE IS NOT IN THE TABU LIST `TABU_LIST`. IT ENSURES THAT THE SUCCESSOR STATE HAS NOT BEEN RECENTLY VISITED.
    #     4. `IF BEST_SUCCESSOR IS NONE OR __BETTER__(SUCCESSOR, BEST_SUCCESSOR): BEST_SUCCESSOR = SUCCESSOR: IF `BEST_SUCCESSOR` IS STILL `NONE`, INDICATING THAT NO BEST SUCCESSOR HAS BEEN ASSIGNED YET, OR IF `BEST_SUCCESSOR` IS NOT `NONE`, THE CONDITION `__BETTER__(SUCCESSOR, BEST_SUCCESSOR)` IS CHECKED. IT CALLS THE `__BETTER__()` FUNCTION TO DETERMINE IF THE CURRENT SUCCESSOR IS BETTER THAN THE CURRENT BEST SUCCESSOR. IF IT IS, THE CURRENT SUCCESSOR STATE REPLACES THE BEST SUCCESSOR.
    #     6. AFTER THE LOOP COMPLETES, THE FUNCTION RETURNS THE BEST SUCCESSOR STATE STORED IN THE `BEST_SUCCESSOR` VARIABLE. IF NO VALID SUCCESSOR STATE WAS FOUND, `BEST_SUCCESSOR` REMAINS `NONE`.
    # IN SUMMARY, THE `__BEST_SUCCESSOR__()` FUNCTION ITERATES OVER A SET OF GENERATED SUCCESSOR STATES FROM THE CURRENT STATE. IT CHECKS EACH SUCCESSOR STATE TO ENSURE IT IS NOT IN THE TABU LIST AND THEN COMPARES IT TO THE CURRENT BEST SUCCESSOR STATE. THE BEST SUCCESSOR STATE IS DETERMINED BASED ON A COMPARISON FUNCTION, `__BETTER__()`, WHICH EVALUATES THE QUALITY OF THE SUCCESSORS USING A COST FUNCTION. THE FUNCTION RETURNS THE BEST SUCCESSOR STATE FOUND, WHICH WILL BE USED AS THE NEXT STATE IN THE TABU SEARCH ALGORITHM.
    def __BEST_SUCCESSOR__(STATE: dict, TABU_LIST: list) -> dict | None:
        """THE `__BEST_SUCCESSOR__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO SELECT THE BEST SUCCESSOR STATE FROM A SET OF GENERATED SUCCESSORS."""
        BEST_SUCCESSOR = None  # INITIALIZE BEST SUCCESSOR TO NONE
        # FOR EACH SUCCESSOR STATE
        for SUCCESSOR in __SUCCESSORS__(STATE):
            if SUCCESSOR not in TABU_LIST:  # IF SUCCESSOR NOT IN TABU LIST
                # IF BEST SUCCESSOR IS NONE OR SUCCESSOR IS BETTER THAN BEST SUCCESSOR
                if BEST_SUCCESSOR is None or __BETTER__(SUCCESSOR, BEST_SUCCESSOR):
                    BEST_SUCCESSOR = SUCCESSOR  # SET BEST SUCCESSOR TO SUCCESSOR
        return BEST_SUCCESSOR  # RETURN BEST SUCCESSOR

    # THE `__SUCCESSORS__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO GENERATE A SET OF SUCCESSOR STATES FROM A GIVEN STATE IN A CONSTRAINT SATISFACTION PROBLEM (CSP).
    #     1. `SUCCESSORS = []`: THIS LINE INITIALIZES AN EMPTY LIST NAMED `SUCCESSORS` TO STORE THE GENERATED SUCCESSOR STATES.
    #     2. `FOR VARIABLE IN CSP_PROBLEM.VARIABLES:`: THIS LINE STARTS A LOOP THAT ITERATES OVER EACH VARIABLE IN THE CSP PROBLEM. `CSP_PROBLEM.VARIABLES` IS A LIST OF ALL THE VARIABLES IN THE PROBLEM.
    #     3. `FOR VALUE IN CSP_PROBLEM.DOMAINS[VARIABLE]:`: WITHIN THE LOOP, THIS LINE STARTS ANOTHER LOOP THAT ITERATES OVER EACH VALUE IN THE DOMAIN OF THE CURRENT VARIABLE. `CSP_PROBLEM.DOMAINS[VARIABLE]` IS A LIST OF ALL THE POSSIBLE VALUES (DOMAIN) FOR THE VARIABLE.
    #     4. `IF STATE[VARIABLE] != VALUE:`: THIS CONDITION CHECKS IF THE VALUE OF THE CURRENT VARIABLE IN THE GIVEN STATE IS DIFFERENT FROM THE CURRENT VALUE BEING CONSIDERED. IF THEY ARE DIFFERENT, IT MEANS THAT A VALID SUCCESSOR STATE CAN BE GENERATED.
    #     5. `SUCCESSOR = STATE.COPY()`: IF A VALID SUCCESSOR STATE CAN BE GENERATED, A COPY OF THE CURRENT STATE DICTIONARY IS MADE, AND THE COPY IS ASSIGNED TO THE `SUCCESSOR` VARIABLE.
    #     6. `SUCCESSOR[VARIABLE] = VALUE`: IN THE `SUCCESSOR` STATE, THE VALUE OF THE CURRENT VARIABLE IS UPDATED WITH THE CURRENT VALUE BEING CONSIDERED. THIS CREATES A NEW SUCCESSOR STATE BY CHANGING THE VALUE OF A SINGLE VARIABLE.
    #     7. `SUCCESSORS.APPEND(SUCCESSOR)`: THE GENERATED SUCCESSOR STATE IS ADDED TO THE `SUCCESSORS` LIST.
    #     8. THE NESTED LOOP CONTINUES UNTIL ALL POSSIBLE VALUES FOR ALL VARIABLES HAVE BEEN CONSIDERED, GENERATING ALL POSSIBLE SUCCESSOR STATES.
    #     9. FINALLY, THE FUNCTION RETURNS THE LIST OF GENERATED SUCCESSOR STATES STORED IN THE `SUCCESSORS` LIST.
    # IN SUMMARY, THE `__SUCCESSORS__()` FUNCTION GENERATES A SET OF SUCCESSOR STATES FROM A GIVEN STATE IN A CSP PROBLEM. IT ITERATES OVER EACH VARIABLE AND EACH VALUE IN THE DOMAIN OF THAT VARIABLE, CREATING A NEW SUCCESSOR STATE BY UPDATING THE VALUE OF A SINGLE VARIABLE WHILE KEEPING THE VALUES OF OTHER VARIABLES UNCHANGED. THESE GENERATED SUCCESSOR STATES REPRESENT POSSIBLE MOVES FROM THE CURRENT STATE IN THE TABU SEARCH ALGORITHM, ALLOWING EXPLORATION OF THE SOLUTION SPACE.
    def __SUCCESSORS__(STATE: dict) -> list:
        """THE `__SUCCESSORS__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO GENERATE A SET OF SUCCESSOR STATES FROM A GIVEN STATE IN A CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        SUCCESSORS = []  # INITIALIZE SUCCESSORS LIST
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # FOR EACH VARIABLE IN CSP PROBLEM
            # FOR EACH VALUE IN DOMAIN OF VARIABLE
            for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
                if STATE[VARIABLE] != VALUE:  # IF STATE VARIABLE NOT EQUAL TO VALUE
                    SUCCESSOR = STATE.copy()  # COPY STATE TO SUCCESSOR
                    # UPDATE SUCCESSOR VARIABLE WITH VALUE
                    SUCCESSOR[VARIABLE] = VALUE
                    SUCCESSORS.append(SUCCESSOR)  # ADD SUCCESSOR TO SUCCESSORS
        return SUCCESSORS  # RETURN SUCCESSORS

    # THE `__BETTER__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO COMPARE TWO SUCCESSOR STATES AND DETERMINE WHICH ONE IS CONSIDERED "BETTER" BASED ON A COST FUNCTION.
    #     1. `IF __COST__(SUCCESSOR) < __COST__(BEST_SUCCESSOR):`: THIS LINE COMPARES THE COST OF THE CURRENT SUCCESSOR STATE (`SUCCESSOR`) WITH THE COST OF THE CURRENT BEST SUCCESSOR STATE (`BEST_SUCCESSOR`). IT CALLS THE `__COST__()` FUNCTION TO CALCULATE THE COST OF EACH STATE, PASSING THE SUCCESSOR STATES AND THE CSP PROBLEM AS ARGUMENTS.
    #     2. `RETURN TRUE`: IF THE COST OF THE CURRENT SUCCESSOR STATE IS LOWER THAN THE COST OF THE CURRENT BEST SUCCESSOR STATE, IT INDICATES THAT THE CURRENT SUCCESSOR STATE IS CONSIDERED "BETTER". IN THIS CASE, THE FUNCTION RETURNS `TRUE`.
    #     3. IF THE CONDITION IN STEP 1 IS NOT MET, THE FUNCTION PROCEEDS TO THE NEXT LINE.
    #     4. `RETURN FALSE`: IF THE COST OF THE CURRENT SUCCESSOR STATE IS EQUAL TO OR HIGHER THAN THE COST OF THE CURRENT BEST SUCCESSOR STATE, IT INDICATES THAT THE CURRENT SUCCESSOR STATE IS NOT CONSIDERED "BETTER". IN THIS CASE, THE FUNCTION RETURNS `FALSE`.
    # IN SUMMARY, THE `__BETTER__()` FUNCTION COMPARES TWO SUCCESSOR STATES BASED ON THEIR COSTS USING A COST FUNCTION (`__COST__()`). IT RETURNS `TRUE` IF THE COST OF THE FIRST STATE IS LOWER THAN THE COST OF THE SECOND STATE, INDICATING THAT THE FIRST STATE IS CONSIDERED "BETTER". OTHERWISE, IT RETURNS `FALSE`. THIS FUNCTION IS USED IN THE TABU SEARCH ALGORITHM TO DETERMINE WHICH SUCCESSOR STATE SHOULD BE CHOSEN AS THE NEXT STATE BASED ON ITS COST IN ORDER TO GUIDE THE SEARCH TOWARDS BETTER SOLUTIONS.
    def __BETTER__(SUCCESSOR: dict, BEST_SUCCESSOR: dict) -> bool:
        """THE `__BETTER__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO COMPARE TWO SUCCESSOR STATES AND DETERMINE WHICH ONE IS CONSIDERED "BETTER" BASED ON A COST FUNCTION."""
        # IF SUCCESSOR COST < BEST SUCCESSOR COST
        if __COST__(SUCCESSOR) < __COST__(BEST_SUCCESSOR):
            return True  # RETURN TRUE
        return False  # RETURN FALSE

    # THE `__COST__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO CALCULATE THE COST OF A GIVEN STATE IN A CONSTRAINT SATISFACTION PROBLEM (CSP).
    #     1. `COST = 0`: THIS LINE INITIALIZES THE `COST` VARIABLE TO 0. THIS VARIABLE WILL STORE THE COST ASSOCIATED WITH THE GIVEN STATE.
    #     2. `FOR CONSTRAINT IN CSP_PROBLEM.CONSTRAINTS:`: THIS LINE STARTS A LOOP THAT ITERATES OVER EACH CONSTRAINT IN THE CSP PROBLEM. `CSP_PROBLEM.CONSTRAINTS` IS A LIST OF ALL THE CONSTRAINTS IN THE PROBLEM.
    #     3. `IF NOT __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):`: WITHIN THE LOOP, THIS CONDITION CHECKS IF THE GIVEN STATE VIOLATES THE CURRENT CONSTRAINT BY CALLING THE `__CONSTRAINT_SATISFIED__()` FUNCTION. IT PASSES THE STATE DICTIONARY AND THE CURRENT CONSTRAINT TUPLE AS ARGUMENTS.
    #     4. `COST += 1`: IF THE CURRENT CONSTRAINT IS VIOLATED (I.E., THE CONDITION IN STEP 3 IS MET), THE COST IS INCREMENTED BY 1. THIS INDICATES THAT THERE IS A VIOLATION OF A CONSTRAINT IN THE GIVEN STATE.
    #     5. THE LOOP CONTINUES UNTIL ALL CONSTRAINTS HAVE BEEN CHECKED.
    #     6. AFTER THE LOOP COMPLETES, THE FUNCTION RETURNS THE CALCULATED COST STORED IN THE `COST` VARIABLE.
    # IN SUMMARY, THE `__COST__()` FUNCTION CALCULATES THE COST OF A GIVEN STATE IN A CSP PROBLEM. IT ITERATES OVER EACH CONSTRAINT AND, FOR EACH VIOLATED CONSTRAINT, INCREMENTS THE COST BY 1. THE RESULTING COST REFLECTS THE NUMBER OF VIOLATED CONSTRAINTS IN THE GIVEN STATE. THIS COST FUNCTION IS USED IN THE TABU SEARCH ALGORITHM TO EVALUATE THE QUALITY OF A STATE AND GUIDE THE SEARCH TOWARDS FINDING SOLUTIONS WITH LOWER COSTS, AIMING TO IMPROVE THE QUALITY OF THE SOLUTION.
    def __COST__(STATE: dict) -> int:
        """THE `__COST__()` FUNCTION IS A HELPER FUNCTION USED IN THE TABU SEARCH ALGORITHM TO CALCULATE THE COST OF A GIVEN STATE IN A CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        COST = 0  # INITIALIZE COST TO 0
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # FOR EACH CONSTRAINT IN CSP PROBLEM
            # IF CONSTRAINT NOT SATISFIED
            if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
                COST += 1  # INCREMENT COST BY 1
        return COST  # RETURN COST

    CURRENT_STATE = __INITIAL_STATE__()  # INITIALIZE CURRENT STATE
    TABU_LIST = []  # INITIALIZE TABU LIST
    TABU_LIST_SIZE = len(CSP_PROBLEM.VARIABLES)  # SET TABU LIST SIZE
    while True:  # LOOP UNTIL SOLUTION IS FOUND OR NO MORE VALID SUCCESSORS CAN BE GENERATED
        if __SOLUTION__(CURRENT_STATE):  # IF CURRENT STATE IS A SOLUTION
            return CURRENT_STATE  # RETURN CURRENT STATE
        CURRENT_STATE = __BEST_SUCCESSOR__(
            CURRENT_STATE, TABU_LIST)  # GENERATE BEST SUCCESSOR
        if CURRENT_STATE is None:  # IF NO VALID SUCCESSORS WERE GENERATED
            return None  # RETURN NONE
        TABU_LIST.append(CURRENT_STATE)  # ADD CURRENT STATE TO TABU LIST
        if len(TABU_LIST) > TABU_LIST_SIZE:  # IF TABU LIST HAS EXCEEDED MAXIMUM SIZE
            TABU_LIST.pop(0)  # REMOVE OLDEST ENTRY FROM TABU LIST