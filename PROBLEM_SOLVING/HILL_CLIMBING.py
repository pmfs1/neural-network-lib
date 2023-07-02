from .CSP import CSP

# THE `HILL_CLIMBING` FUNCTION IS AN IMPLEMENTATION OF THE HILL CLIMBING ALGORITHM FOR CONSTRAINT SATISFACTION PROBLEMS (CSP). THE HILL CLIMBING ALGORITHM IS A LOCAL SEARCH ALGORITHM THAT STARTS WITH AN INITIAL STATE AND ITERATIVELY MOVES TO ITS NEIGHBORING STATES, SELECTING THE BEST SUCCESSOR BASED ON SOME EVALUATION FUNCTION. THE ALGORITHM TERMINATES WHEN IT REACHES A SOLUTION STATE OR CANNOT FIND ANY BETTER SUCCESSOR STATE.
#     1. THE FUNCTION TAKES A `CSP_PROBLEM` AS INPUT, WHICH IS AN INSTANCE OF THE CSP CLASS. THE CSP CLASS HOLDS INFORMATION ABOUT THE VARIABLES, DOMAINS, AND CONSTRAINTS OF THE CSP PROBLEM.
#     2. SEVERAL HELPER FUNCTIONS ARE DEFINED WITHIN THE `HILL_CLIMBING` FUNCTION. THESE FUNCTIONS ARE USED TO AID IN THE IMPLEMENTATION OF THE HILL CLIMBING ALGORITHM:
#         - `__INITIAL_STATE__`: INITIALIZES THE INITIAL STATE OF THE CSP PROBLEM. IT SETS EACH VARIABLE TO THE FIRST VALUE IN ITS DOMAIN.
#         - `__SOLUTION__`: CHECKS IF A GIVEN STATE IS A VALID SOLUTION FOR THE CSP PROBLEM. IT CHECKS IF ALL CONSTRAINTS ARE SATISFIED IN THE STATE.
#         - `__CONSTRAINT_SATISFIED__`: CHECKS IF A GIVEN CONSTRAINT IS SATISFIED IN A STATE. A CONSTRAINT IS A TUPLE OF VARIABLES, AND IT REPRESENTS A CONSTRAINT BETWEEN THOSE VARIABLES.
#         - `__BEST_SUCCESSOR__`: SELECTS THE BEST SUCCESSOR STATE FROM A SET OF SUCCESSOR STATES. IT EVALUATES EACH SUCCESSOR STATE BASED ON A SET OF CRITERIA AND SELECTS THE ONE THAT IMPROVES THE CURRENT STATE THE MOST.
#         - `__SUCCESSORS__`: GENERATES A SET OF SUCCESSOR STATES BY MAKING ONE VARIABLE ASSIGNMENT DIFFERENT FROM THE CURRENT STATE. IT EXPLORES ALL POSSIBLE SINGLE-VARIABLE ASSIGNMENTS.
#         - `__CONSISTENT__`: CHECKS IF A GIVEN SUCCESSOR STATE SATISFIES ALL THE CONSTRAINTS OF THE CSP PROBLEM.
#         - `__BETTER__`: COMPARES TWO STATES AND DETERMINES IF A GIVEN SUCCESSOR STATE IS BETTER THAN THE CURRENT BEST SUCCESSOR STATE. IT IS USED AS A CRITERION TO CHOOSE THE BEST SUCCESSOR.
#         - `__NUMBER_OF_VIOLATIONS__`: COUNTS THE NUMBER OF CONSTRAINT VIOLATIONS IN A GIVEN STATE. IT PROVIDES A MEASURE OF THE QUALITY OR FITNESS OF A STATE BASED ON THE NUMBER OF CONSTRAINTS THAT ARE NOT SATISFIED.
#     3. THE `HILL_CLIMBING` FUNCTION STARTS BY INITIALIZING THE `CURRENT_STATE` USING THE `__INITIAL_STATE__` HELPER FUNCTION.
#     4. THE FUNCTION THEN ENTERS A LOOP THAT CONTINUES UNTIL A SOLUTION IS FOUND OR NO BETTER SUCCESSOR STATE CAN BE GENERATED.
#     5. IN EACH ITERATION, THE FUNCTION CHECKS IF THE `CURRENT_STATE` IS A VALID SOLUTION USING THE `__SOLUTION__` HELPER FUNCTION. IF IT IS A SOLUTION, THE FUNCTION RETURNS THE CURRENT STATE.
#     6. IF THE `CURRENT_STATE` IS NOT A SOLUTION, THE FUNCTION CALLS THE `__BEST_SUCCESSOR__` HELPER FUNCTION TO FIND THE BEST SUCCESSOR STATE AMONG ALL THE STATES GENERATED BY THE `__SUCCESSORS__` FUNCTION.
#     7. IF A BETTER SUCCESSOR STATE IS FOUND, THE `CURRENT_STATE` IS UPDATED TO THE BEST SUCCESSOR, AND THE LOOP CONTINUES TO THE NEXT ITERATION.
#     8. IF NO BETTER SUCCESSOR STATE IS FOUND (I.E., `SUCCESSOR` IS `NONE`), THE FUNCTION RETURNS `FALSE`, INDICATING THAT THE ALGORITHM COULDN'T FIND A SOLUTION TO THE CSP PROBLEM.
# IT'S IMPORTANT TO NOTE THAT HILL CLIMBING IS A LOCAL SEARCH ALGORITHM, AND ITS PERFORMANCE HEAVILY DEPENDS ON THE INITIAL STATE AND THE SEARCH SPACE. IT MAY NOT ALWAYS FIND THE GLOBAL OPTIMAL SOLUTION FOR COMPLEX CSP PROBLEMS, AND IT MIGHT GET STUCK IN LOCAL OPTIMA. DIFFERENT VARIATIONS AND ENHANCEMENTS OF THE HILL CLIMBING ALGORITHM EXIST TO IMPROVE ITS PERFORMANCE AND ADDRESS THESE LIMITATIONS.
def HILL_CLIMBING(CSP_PROBLEM: CSP) -> dict | bool:
    """THE `HILL_CLIMBING` FUNCTION IS AN IMPLEMENTATION OF THE HILL CLIMBING ALGORITHM FOR CONSTRAINT SATISFACTION PROBLEMS (CSP). THE HILL CLIMBING ALGORITHM IS A LOCAL SEARCH ALGORITHM THAT STARTS WITH AN INITIAL STATE AND ITERATIVELY MOVES TO ITS NEIGHBORING STATES, SELECTING THE BEST SUCCESSOR BASED ON SOME EVALUATION FUNCTION. THE ALGORITHM TERMINATES WHEN IT REACHES A SOLUTION STATE OR CANNOT FIND ANY BETTER SUCCESSOR STATE."""
    
    # THE `__INITIAL_STATE__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO INITIALIZE THE INITIAL STATE OF THE CSP PROBLEM.
    #     - IT STARTS BY CREATING AN EMPTY DICTIONARY CALLED `INITIAL_STATE` TO STORE THE INITIAL ASSIGNMENT OF VALUES TO VARIABLES.
    #     - IT THEN ITERATES OVER EACH VARIABLE IN THE `CSP_PROBLEM` OBJECT BY ACCESSING THE `VARIABLES` ATTRIBUTE.
    #     - FOR EACH VARIABLE, IT ASSIGNS THE FIRST VALUE FROM ITS CORRESPONDING DOMAIN TO THE VARIABLE IN THE `INITIAL_STATE` DICTIONARY. THE DOMAINS OF THE VARIABLES ARE ACCESSED FROM THE `DOMAINS` ATTRIBUTE OF THE `CSP_PROBLEM`.
    #     - AFTER ITERATING OVER ALL VARIABLES, THE FUNCTION RETURNS THE `INITIAL_STATE` DICTIONARY REPRESENTING THE INITIAL ASSIGNMENT OF VALUES TO VARIABLES.
    # IN SUMMARY, THE `__INITIAL_STATE__` FUNCTION CREATES AN INITIAL ASSIGNMENT OF VALUES TO VARIABLES IN THE CSP PROBLEM BY ASSIGNING THE FIRST VALUE FROM EACH VARIABLE'S DOMAIN. IT RETURNS A DICTIONARY REPRESENTING THE INITIAL STATE.
    def __INITIAL_STATE__() -> dict:
        """THE `__INITIAL_STATE__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO INITIALIZE THE INITIAL STATE OF THE CSP PROBLEM."""
        INITIAL_STATE = {}  # INITIALIZE THE INITIAL STATE
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # ITERATE OVER EACH VARIABLE IN THE CSP PROBLEM
            # ASSIGN THE FIRST VALUE FROM THE VARIABLE'S DOMAIN TO THE VARIABLE IN THE INITIAL STATE
            INITIAL_STATE[VARIABLE] = CSP_PROBLEM.DOMAINS[VARIABLE][0]
        return INITIAL_STATE  # RETURN THE INITIAL STATE

    # THE `__SOLUTION__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO CHECK IF A GIVEN STATE IS A VALID SOLUTION FOR THE CONSTRAINT SATISFACTION PROBLEM (CSP).
    # `__SOLUTION__(STATE)`: THIS FUNCTION TAKES ONE PARAMETER: `STATE`, WHICH REPRESENTS A STATE (ASSIGNMENT OF VALUES TO VARIABLES).
    #     - IT STARTS BY ITERATING OVER EACH CONSTRAINT IN THE `CONSTRAINTS` ATTRIBUTE OF THE `CSP_PROBLEM`.
    #     - FOR EACH CONSTRAINT, IT CALLS THE `__CONSTRAINT_SATISFIED__` FUNCTION TO CHECK IF THE CONSTRAINT IS SATISFIED IN THE GIVEN `STATE`. THE `__CONSTRAINT_SATISFIED__` FUNCTION CHECKS IF THE ASSIGNED VALUES OF THE VARIABLES IN THE CONSTRAINT SATISFY THE CONSTRAINT'S CONDITIONS.
    #     - IF ANY CONSTRAINT IS NOT SATISFIED (I.E., `__CONSTRAINT_SATISFIED__` RETURNS `FALSE`), IT MEANS THE GIVEN STATE VIOLATES AT LEAST ONE CONSTRAINT. IN THIS CASE, THE FUNCTION RETURNS `FALSE` TO INDICATE THAT THE STATE IS NOT A SOLUTION.
    #     - IF ALL CONSTRAINTS ARE SATISFIED (I.E., `__CONSTRAINT_SATISFIED__` RETURNS `TRUE` FOR ALL CONSTRAINTS), IT MEANS THE GIVEN STATE SATISFIES ALL THE CONSTRAINTS. IN THIS CASE, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE STATE IS A VALID SOLUTION.
    # IN SUMMARY, THE `__SOLUTION__` FUNCTION CHECKS IF A GIVEN STATE SATISFIES ALL THE CONSTRAINTS IN THE CSP PROBLEM. IT ITERATES OVER THE CONSTRAINTS AND USES THE `__CONSTRAINT_SATISFIED__` FUNCTION TO CHECK EACH CONSTRAINT. IF ANY CONSTRAINT IS VIOLATED, IT RETURNS `FALSE`, INDICATING THAT THE STATE IS NOT A SOLUTION. IF ALL CONSTRAINTS ARE SATISFIED, IT RETURNS `TRUE`, INDICATING THAT THE STATE IS A VALID SOLUTION.
    def __SOLUTION__(STATE: dict) -> bool:
        """THE `__SOLUTION__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO CHECK IF A GIVEN STATE IS A VALID SOLUTION FOR THE CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # ITERATE OVER EACH CONSTRAINT IN THE CSP PROBLEM
            # IF THE CONSTRAINT IS NOT SATISFIED IN THE GIVEN STATE, RETURN FALSE
            if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
                return False  # RETURN FALSE
        return True  # RETURN TRUE

    # THE `__CONSTRAINT_SATISFIED__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO CHECK IF A GIVEN CONSTRAINT IS SATISFIED IN A STATE.
    # `__CONSTRAINT_SATISFIED__(STATE, CONSTRAINT)`: THIS FUNCTION TAKES TWO PARAMETERS: `STATE`, WHICH REPRESENTS A STATE (ASSIGNMENT OF VALUES TO VARIABLES), AND `CONSTRAINT`, WHICH IS A CONSTRAINT TO BE CHECKED.
    #     - IT STARTS BY ITERATING OVER EACH VARIABLE PAIR (`VARIABLE_1`, `VARIABLE_2`) IN THE `CONSTRAINT`.
    #     - FOR EACH VARIABLE PAIR, IT CHECKS IF THE VALUES ASSIGNED TO `VARIABLE_1` AND `VARIABLE_2` IN THE GIVEN `STATE` ARE EQUAL. IF THE VALUES ARE EQUAL, IT MEANS THE CONSTRAINT IS NOT SATISFIED.
    #     - IF ANY VARIABLE PAIR HAS EQUAL VALUES (I.E., THE CONSTRAINT IS NOT SATISFIED), THE FUNCTION RETURNS `FALSE` TO INDICATE THAT THE CONSTRAINT IS VIOLATED.
    #     - IF ALL VARIABLE PAIRS HAVE DISTINCT VALUES (I.E., THE CONSTRAINT IS SATISFIED), THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE CONSTRAINT IS SATISFIED.
    # IN SUMMARY, THE `__CONSTRAINT_SATISFIED__` FUNCTION CHECKS IF A GIVEN CONSTRAINT IS SATISFIED IN A STATE BY COMPARING THE ASSIGNED VALUES OF THE VARIABLE PAIRS IN THE CONSTRAINT. IF ANY VARIABLE PAIR HAS EQUAL VALUES, IT RETURNS `FALSE`, INDICATING THAT THE CONSTRAINT IS VIOLATED. IF ALL VARIABLE PAIRS HAVE DISTINCT VALUES, IT RETURNS `TRUE`, INDICATING THAT THE CONSTRAINT IS SATISFIED.
    def __CONSTRAINT_SATISFIED__(STATE: dict, CONSTRAINT: tuple) -> bool:
        """THE `__CONSTRAINT_SATISFIED__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO CHECK IF A GIVEN CONSTRAINT IS SATISFIED IN A STATE."""
        for VARIABLE_1, VARIABLE_2 in CONSTRAINT:  # ITERATE OVER EACH VARIABLE PAIR IN THE CONSTRAINT
            # IF THE VALUES ASSIGNED TO THE VARIABLES IN THE GIVEN STATE ARE EQUAL, RETURN FALSE
            if STATE[VARIABLE_1] == STATE[VARIABLE_2]:
                return False  # RETURN FALSE
        return True  # RETURN TRUE

    # THE `__BEST_SUCCESSOR__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO SELECT THE BEST SUCCESSOR STATE FROM A SET OF SUCCESSOR STATES. IT EVALUATES EACH SUCCESSOR STATE BASED ON A SET OF CRITERIA AND SELECTS THE ONE THAT IMPROVES THE CURRENT STATE THE MOST.
    # `__BEST_SUCCESSOR__(STATE)`: THIS FUNCTION TAKES ONE PARAMETER: `STATE`, WHICH REPRESENTS THE CURRENT STATE.
    #     - IT INITIALIZES A VARIABLE CALLED `BEST_SUCCESSOR` TO `FALSE`. THIS VARIABLE WILL STORE THE BEST SUCCESSOR STATE FOUND SO FAR.
    #     - THE FUNCTION THEN ITERATES OVER EACH SUCCESSOR STATE GENERATED BY CALLING THE `__SUCCESSORS__` FUNCTION. THIS FUNCTION GENERATES ALL POSSIBLE SUCCESSOR STATES BY MAKING ONE VARIABLE ASSIGNMENT DIFFERENT FROM THE CURRENT STATE.
    #     - FOR EACH SUCCESSOR STATE, IT EVALUATES WHETHER IT IS A BETTER SUCCESSOR THAN THE CURRENT `BEST_SUCCESSOR` BY CALLING THE `__BETTER__` FUNCTION. THE `__BETTER__` FUNCTION COMPARES THE TWO STATES BASED ON A SET OF CRITERIA AND DETERMINES IF THE NEW SUCCESSOR IS BETTER.
    #     - IF THE CURRENT SUCCESSOR IS BETTER THAN THE CURRENT `BEST_SUCCESSOR`, IT UPDATES `BEST_SUCCESSOR` TO THE CURRENT SUCCESSOR STATE.
    #     - AFTER EVALUATING ALL SUCCESSOR STATES, THE FUNCTION RETURNS THE `BEST_SUCCESSOR`. IF NO BETTER SUCCESSOR STATE IS FOUND (I.E., ALL SUCCESSOR STATES ARE WORSE OR THE SAME AS THE CURRENT STATE), THE FUNCTION RETURNS `NONE`.
    # IN SUMMARY, THE `__BEST_SUCCESSOR__` FUNCTION ITERATES OVER ALL SUCCESSOR STATES AND EVALUATES EACH ONE USING THE `__BETTER__` FUNCTION. IT SELECTS THE SUCCESSOR STATE THAT IMPROVES THE CURRENT STATE THE MOST ACCORDING TO THE DEFINED CRITERIA. THE SELECTED SUCCESSOR STATE IS THEN RETURNED AS THE BEST SUCCESSOR.
    def __BEST_SUCCESSOR__(STATE: dict) -> dict | None:
        """THE `__BEST_SUCCESSOR__` FUNCTION IS A HELPER FUNCTION USED BY THE `HILL_CLIMBING` FUNCTION TO SELECT THE BEST SUCCESSOR STATE FROM A SET OF SUCCESSOR STATES. IT EVALUATES EACH SUCCESSOR STATE BASED ON A SET OF CRITERIA AND SELECTS THE ONE THAT IMPROVES THE CURRENT STATE THE MOST."""
        BEST_SUCCESSOR = None  # INITIALIZE BEST_SUCCESSOR TO NONE
        # ITERATE OVER EACH SUCCESSOR STATE
        for SUCCESSOR in __SUCCESSORS__(STATE):
            # IF THE SUCCESSOR IS BETTER THAN THE CURRENT BEST_SUCCESSOR, UPDATE BEST_SUCCESSOR TO THE SUCCESSOR
            if __BETTER__(SUCCESSOR, BEST_SUCCESSOR):
                BEST_SUCCESSOR = SUCCESSOR  # UPDATE BEST_SUCCESSOR TO THE SUCCESSOR
        return BEST_SUCCESSOR  # RETURN BEST_SUCCESSOR

    # THE `__SUCCESSORS__` FUNCTION IS A HELPER FUNCTION USED BY THE `__BEST_SUCCESSOR__` FUNCTION IN THE HILL CLIMBING ALGORITHM. ITS PURPOSE IS TO GENERATE A SET OF SUCCESSOR STATES BY MAKING ONE VARIABLE ASSIGNMENT DIFFERENT FROM THE CURRENT STATE.
    # `__SUCCESSORS__(STATE)`: THIS FUNCTION TAKES ONE PARAMETER: `STATE`, WHICH REPRESENTS THE CURRENT STATE.
    #     - IT INITIALIZES AN EMPTY LIST CALLED `SUCCESSORS` TO STORE THE GENERATED SUCCESSOR STATES.
    #     - THE FUNCTION ITERATES OVER EACH VARIABLE IN THE `VARIABLES` ATTRIBUTE OF THE `CSP_PROBLEM`.
    #     - FOR EACH VARIABLE, IT ITERATES OVER EACH VALUE IN THE DOMAIN OF THAT VARIABLE (ACCESSED FROM THE `DOMAINS` ATTRIBUTE OF THE `CSP_PROBLEM`).
    #     - FOR EACH VALUE, IT CREATES A NEW SUCCESSOR STATE BY MAKING A COPY OF THE CURRENT STATE AND UPDATING THE VALUE OF THE CURRENT VARIABLE TO THE NEW VALUE.
    #     - THE NEW SUCCESSOR STATE IS ADDED TO THE `SUCCESSORS` LIST IF IT SATISFIES THE CONSISTENCY CHECK PERFORMED BY THE `__CONSISTENT__` FUNCTION. THE `__CONSISTENT__` FUNCTION CHECKS IF THE SUCCESSOR STATE SATISFIES ALL THE CONSTRAINTS IN THE `CSP_PROBLEM`.
    #     - AFTER ITERATING OVER ALL VARIABLES AND THEIR VALUES, THE FUNCTION RETURNS THE `SUCCESSORS` LIST CONTAINING ALL THE GENERATED SUCCESSOR STATES.
    # IN SUMMARY, THE `__SUCCESSORS__` FUNCTION GENERATES A SET OF SUCCESSOR STATES BY MAKING ONE VARIABLE ASSIGNMENT DIFFERENT FROM THE CURRENT STATE. IT EXPLORES ALL POSSIBLE ASSIGNMENTS FOR EACH VARIABLE AND CHECKS THE CONSISTENCY OF EACH SUCCESSOR STATE. THE GENERATED SUCCESSOR STATES ARE STORED IN A LIST AND RETURNED FOR FURTHER EVALUATION IN THE HILL CLIMBING ALGORITHM.
    def __SUCCESSORS__(STATE: dict) -> list:
        """THE `__SUCCESSORS__` FUNCTION IS A HELPER FUNCTION USED BY THE `__BEST_SUCCESSOR__` FUNCTION IN THE HILL CLIMBING ALGORITHM. ITS PURPOSE IS TO GENERATE A SET OF SUCCESSOR STATES BY MAKING ONE VARIABLE ASSIGNMENT DIFFERENT FROM THE CURRENT STATE."""
        SUCCESSORS = []  # INITIALIZE SUCCESSORS TO AN EMPTY LIST
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # ITERATE OVER EACH VARIABLE IN THE CSP PROBLEM
            # ITERATE OVER EACH VALUE IN THE DOMAIN OF THE VARIABLE
            for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
                # IF THE VALUE IS DIFFERENT FROM THE CURRENT VALUE OF THE VARIABLE IN THE STATE
                if VALUE != STATE[VARIABLE]:
                    SUCCESSOR = {}  # INITIALIZE A NEW SUCCESSOR STATE
                    for VARIABLE_2 in CSP_PROBLEM.VARIABLES:  # ITERATE OVER EACH VARIABLE IN THE CSP PROBLEM
                        # COPY THE CURRENT STATE TO THE SUCCESSOR STATE
                        SUCCESSOR[VARIABLE_2] = STATE[VARIABLE_2]
                    # UPDATE THE VALUE OF THE VARIABLE IN THE SUCCESSOR STATE
                    SUCCESSOR[VARIABLE] = VALUE
                    # IF THE SUCCESSOR STATE IS CONSISTENT
                    if __CONSISTENT__(SUCCESSOR):
                        # ADD THE SUCCESSOR STATE TO THE LIST OF SUCCESSORS
                        SUCCESSORS.append(SUCCESSOR)
        return SUCCESSORS  # RETURN THE LIST OF SUCCESSORS

    # THE `__CONSISTENT__` FUNCTION IS A HELPER FUNCTION USED IN THE HILL CLIMBING ALGORITHM TO CHECK THE CONSISTENCY OF A SUCCESSOR STATE. ITS PURPOSE IS TO DETERMINE IF A GIVEN SUCCESSOR STATE SATISFIES ALL THE CONSTRAINTS OF THE CONSTRAINT SATISFACTION PROBLEM (CSP).
    # `__CONSISTENT__(SUCCESSOR)`: THIS FUNCTION TAKES ONE PARAMETER: `SUCCESSOR`, WHICH REPRESENTS A SUCCESSOR STATE (AN ASSIGNMENT OF VALUES TO VARIABLES).
    #     - THE FUNCTION ITERATES OVER EACH CONSTRAINT IN THE `CONSTRAINTS` ATTRIBUTE OF THE `CSP_PROBLEM`.
    #     - FOR EACH CONSTRAINT, IT CALLS THE `__CONSTRAINT_SATISFIED__` FUNCTION TO CHECK IF THE CONSTRAINT IS SATISFIED IN THE `SUCCESSOR` STATE. THE `__CONSTRAINT_SATISFIED__` FUNCTION CHECKS IF THE ASSIGNED VALUES OF THE VARIABLES IN THE CONSTRAINT SATISFY THE CONSTRAINT'S CONDITIONS.
    #     - IF ANY CONSTRAINT IS NOT SATISFIED (I.E., `__CONSTRAINT_SATISFIED__` RETURNS `FALSE`), IT MEANS THE SUCCESSOR STATE VIOLATES AT LEAST ONE CONSTRAINT. IN THIS CASE, THE FUNCTION RETURNS `FALSE` TO INDICATE THAT THE SUCCESSOR STATE IS NOT CONSISTENT.
    #     - IF ALL CONSTRAINTS ARE SATISFIED (I.E., `__CONSTRAINT_SATISFIED__` RETURNS `TRUE` FOR ALL CONSTRAINTS), IT MEANS THE SUCCESSOR STATE SATISFIES ALL THE CONSTRAINTS. IN THIS CASE, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE SUCCESSOR STATE IS CONSISTENT.
    # IN SUMMARY, THE `__CONSISTENT__` FUNCTION CHECKS IF A GIVEN SUCCESSOR STATE SATISFIES ALL THE CONSTRAINTS OF THE CSP PROBLEM. IT ITERATES OVER THE CONSTRAINTS AND USES THE `__CONSTRAINT_SATISFIED__` FUNCTION TO CHECK EACH CONSTRAINT. IF ANY CONSTRAINT IS VIOLATED, IT RETURNS `FALSE`, INDICATING THAT THE SUCCESSOR STATE IS NOT CONSISTENT. IF ALL CONSTRAINTS ARE SATISFIED, IT RETURNS `TRUE`, INDICATING THAT THE SUCCESSOR STATE IS CONSISTENT.
    def __CONSISTENT__(SUCCESSOR: dict) -> bool:
        """THE `__CONSISTENT__` FUNCTION IS A HELPER FUNCTION USED BY THE `__SUCCESSORS__` FUNCTION IN THE HILL CLIMBING ALGORITHM. ITS PURPOSE IS TO CHECK IF A GIVEN SUCCESSOR STATE SATISFIES ALL THE CONSTRAINTS OF THE CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # ITERATE OVER EACH CONSTRAINT IN THE CSP PROBLEM
            # IF THE CONSTRAINT IS NOT SATISFIED IN THE SUCCESSOR STATE
            if not __CONSTRAINT_SATISFIED__(SUCCESSOR, CONSTRAINT):
                return False  # RETURN FALSE TO INDICATE THAT THE SUCCESSOR STATE IS NOT CONSISTENT
        return True  # RETURN TRUE TO INDICATE THAT THE SUCCESSOR STATE IS CONSISTENT

    # THE `__BETTER__` FUNCTION IS A HELPER FUNCTION USED IN THE HILL CLIMBING ALGORITHM TO COMPARE TWO STATES AND DETERMINE IF A GIVEN SUCCESSOR STATE IS BETTER THAN THE CURRENT BEST SUCCESSOR STATE. IT CONSIDERS MULTIPLE CRITERIA TO EVALUATE THE QUALITY OF A SUCCESSOR STATE.
    # `__BETTER__(SUCCESSOR, BEST_SUCCESSOR)`: THIS FUNCTION TAKES TWO PARAMETERS: `SUCCESSOR`, WHICH REPRESENTS A SUCCESSOR STATE, `BEST_SUCCESSOR`, WHICH REPRESENTS THE CURRENT BEST SUCCESSOR STATE.
    #     - THE FUNCTION FIRST CHECKS IF THE CURRENT `BEST_SUCCESSOR` IS `FALSE`. IF IT IS, IT MEANS NO BETTER SUCCESSOR HAS BEEN FOUND YET, SO THE `SUCCESSOR` IS CONSIDERED BETTER. IN THIS CASE, THE FUNCTION RETURNS `TRUE`.
    #     - IF THE CURRENT `BEST_SUCCESSOR` IS NOT `FALSE`, THE FUNCTION PERFORMS FURTHER EVALUATIONS BASED ON THE FOLLOWING CRITERIA:
    #         1. IT CHECKS IF THE `SUCCESSOR` IS CONSISTENT BY CALLING THE `__CONSISTENT__` FUNCTION. IF THE `SUCCESSOR` IS CONSISTENT, IT IS CONSIDERED BETTER THAN THE `BEST_SUCCESSOR`. IN THIS CASE, THE FUNCTION RETURNS `TRUE`.
    #         2. IF THE `SUCCESSOR` IS NOT CONSISTENT, IT CHECKS IF THE `BEST_SUCCESSOR` IS ALSO INCONSISTENT. IF THE `BEST_SUCCESSOR` IS INCONSISTENT, THE `SUCCESSOR` IS CONSIDERED BETTER BECAUSE IT HAS A LOWER NUMBER OF CONSTRAINT VIOLATIONS. IN THIS CASE, THE FUNCTION RETURNS `TRUE`.
    #         3. IF BOTH THE `SUCCESSOR` AND `BEST_SUCCESSOR` ARE CONSISTENT, THE FUNCTION COMPARES THE NUMBER OF CONSTRAINT VIOLATIONS IN EACH STATE. IT CALLS THE `__NUMBER_OF_VIOLATIONS__` FUNCTION TO COUNT THE NUMBER OF CONSTRAINT VIOLATIONS IN EACH STATE. IF THE `SUCCESSOR` HAS FEWER VIOLATIONS THAN THE `BEST_SUCCESSOR`, IT IS CONSIDERED BETTER. IN THIS CASE, THE FUNCTION RETURNS `TRUE`.
    #     - IF NONE OF THE ABOVE CONDITIONS ARE MET, IT MEANS THE `SUCCESSOR` IS NOT BETTER THAN THE `BEST_SUCCESSOR`. IN THIS CASE, THE FUNCTION RETURNS `FALSE`.
    # IN SUMMARY, THE `__BETTER__` FUNCTION COMPARES TWO STATES, THE `SUCCESSOR` AND THE `BEST_SUCCESSOR`, BASED ON SEVERAL CRITERIA. IT CONSIDERS CONSISTENCY, THE NUMBER OF CONSTRAINT VIOLATIONS, AND THE CURRENT STATE OF THE `BEST_SUCCESSOR`. IF THE `SUCCESSOR` MEETS ANY OF THE CONDITIONS THAT INDICATE IT IS BETTER, THE FUNCTION RETURNS `TRUE`, OTHERWISE IT RETURNS `FALSE`.
    def __BETTER__(SUCCESSOR: dict, BEST_SUCCESSOR: dict | None) -> bool:
        """THE `__BETTER__` FUNCTION IS A HELPER FUNCTION USED BY THE `__SUCCESSORS__` FUNCTION IN THE HILL CLIMBING ALGORITHM. ITS PURPOSE IS TO COMPARE TWO STATES AND DETERMINE IF A GIVEN SUCCESSOR STATE IS BETTER THAN THE CURRENT BEST SUCCESSOR STATE."""
        if BEST_SUCCESSOR is None:  # IF THE BEST SUCCESSOR IS FALSE
            return True  # RETURN TRUE TO INDICATE THAT THE SUCCESSOR IS BETTER
        if __CONSISTENT__(SUCCESSOR):  # IF THE SUCCESSOR IS CONSISTENT
            return True  # RETURN TRUE TO INDICATE THAT THE SUCCESSOR IS BETTER
        # IF THE BEST SUCCESSOR IS NOT CONSISTENT
        if not __CONSISTENT__(BEST_SUCCESSOR):
            return False  # RETURN FALSE TO INDICATE THAT THE SUCCESSOR IS NOT BETTER
        # IF THE SUCCESSOR HAS FEWER CONSTRAINT VIOLATIONS THAN THE BEST SUCCESSOR
        if __NUMBER_OF_VIOLATIONS__(SUCCESSOR) < __NUMBER_OF_VIOLATIONS__(BEST_SUCCESSOR):
            return True  # RETURN TRUE TO INDICATE THAT THE SUCCESSOR IS BETTER
        return False  # RETURN FALSE TO INDICATE THAT THE SUCCESSOR IS NOT BETTER

    # THE `__NUMBER_OF_VIOLATIONS__` FUNCTION IS A HELPER FUNCTION USED IN THE HILL CLIMBING ALGORITHM TO COUNT THE NUMBER OF CONSTRAINT VIOLATIONS IN A GIVEN STATE. IT PROVIDES A MEASURE OF THE QUALITY OR FITNESS OF A STATE BASED ON THE NUMBER OF CONSTRAINTS THAT ARE NOT SATISFIED.
    # `__NUMBER_OF_VIOLATIONS__(STATE)`: THIS FUNCTION TAKES ONE PARAMETER: `STATE`, WHICH REPRESENTS A STATE (AN ASSIGNMENT OF VALUES TO VARIABLES).
    #     - IT INITIALIZES A VARIABLE CALLED `NUMBER_OF_VIOLATIONS` TO 0. THIS VARIABLE WILL STORE THE COUNT OF CONSTRAINT VIOLATIONS.
    #     - THE FUNCTION ITERATES OVER EACH CONSTRAINT IN THE `CONSTRAINTS` ATTRIBUTE OF THE `CSP_PROBLEM`.
    #     - FOR EACH CONSTRAINT, IT CALLS THE `__CONSTRAINT_SATISFIED__` FUNCTION TO CHECK IF THE CONSTRAINT IS SATISFIED IN THE `STATE` STATE. THE `__CONSTRAINT_SATISFIED__` FUNCTION CHECKS IF THE ASSIGNED VALUES OF THE VARIABLES IN THE CONSTRAINT SATISFY THE CONSTRAINT'S CONDITIONS.
    #     - IF THE CONSTRAINT IS NOT SATISFIED (I.E., `__CONSTRAINT_SATISFIED__` RETURNS `FALSE`), IT MEANS THERE IS A VIOLATION. IN THIS CASE, THE `NUMBER_OF_VIOLATIONS` VARIABLE IS INCREMENTED BY 1.
    #     - AFTER ITERATING OVER ALL CONSTRAINTS, THE FUNCTION RETURNS THE FINAL VALUE OF `NUMBER_OF_VIOLATIONS`, REPRESENTING THE COUNT OF CONSTRAINT VIOLATIONS IN THE GIVEN STATE.
    # IN SUMMARY, THE `__NUMBER_OF_VIOLATIONS__` FUNCTION COUNTS THE NUMBER OF CONSTRAINT VIOLATIONS IN A STATE BY ITERATING OVER THE CONSTRAINTS AND CHECKING EACH ONE USING THE `__CONSTRAINT_SATISFIED__` FUNCTION. IT INCREMENTS A COUNTER FOR EACH CONSTRAINT VIOLATION ENCOUNTERED AND RETURNS THE FINAL COUNT. A LOWER NUMBER OF VIOLATIONS INDICATES A BETTER STATE IN TERMS OF SATISFYING THE CONSTRAINTS.
    def __NUMBER_OF_VIOLATIONS__(STATE: dict) -> int:
        """THE `__NUMBER_OF_VIOLATIONS__` FUNCTION IS A HELPER FUNCTION USED IN THE HILL CLIMBING ALGORITHM TO COUNT THE NUMBER OF CONSTRAINT VIOLATIONS IN A GIVEN STATE. IT PROVIDES A MEASURE OF THE QUALITY OR FITNESS OF A STATE BASED ON THE NUMBER OF CONSTRAINTS THAT ARE NOT SATISFIED."""
        NUMBER_OF_VIOLATIONS = 0  # INITIALIZE A COUNTER FOR THE NUMBER OF CONSTRAINT VIOLATIONS
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # FOR EACH CONSTRAINT IN THE CSP PROBLEM
            # IF THE CONSTRAINT IS NOT SATISFIED IN THE STATE
            if not __CONSTRAINT_SATISFIED__(STATE, CONSTRAINT):
                NUMBER_OF_VIOLATIONS += 1  # INCREMENT THE NUMBER OF VIOLATIONS BY 1
        return NUMBER_OF_VIOLATIONS  # RETURN THE FINAL NUMBER OF VIOLATIONS

    CURRENT_STATE = __INITIAL_STATE__()  # INITIALIZE THE CURRENT STATE
    while True:  # LOOP UNTIL A SOLUTION IS FOUND OR NO BETTER SUCCESSOR IS AVAILABLE
        # IF THE CURRENT STATE IS A SOLUTION, RETURN IT
        if __SOLUTION__(CURRENT_STATE):
            return CURRENT_STATE  # RETURN THE CURRENT STATE
        # GENERATE THE BEST SUCCESSOR STATE
        SUCCESSOR = __BEST_SUCCESSOR__(CURRENT_STATE)
        if SUCCESSOR is None:  # IF NO SUCCESSOR STATE IS AVAILABLE, RETURN FALSE
            return False  # RETURN FALSE
        CURRENT_STATE = SUCCESSOR  # SET THE CURRENT STATE TO THE SUCCESSOR STATE
