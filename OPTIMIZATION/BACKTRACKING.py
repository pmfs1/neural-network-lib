from .CSP import CSP

# THE `BACKTRACKING` FUNCTION IS AN IMPLEMENTATION OF THE BACKTRACKING ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS). IT TAKES A CSP PROBLEM AS INPUT AND RETURNS EITHER A DICTIONARY REPRESENTING A VALID ASSIGNMENT OF VALUES TO VARIABLES THAT SATISFIES ALL CONSTRAINTS OR `FALSE` IF NO SOLUTION EXISTS.
# THE FUNCTION STARTS BY DEFINING SEVERAL HELPER FUNCTIONS THAT ARE USED INTERNALLY WITHIN THE `BACKTRACKING` FUNCTION. THESE HELPER FUNCTIONS PERFORM VARIOUS TASKS SUCH AS SELECTING AN UNASSIGNED VARIABLE, CHECKING IF THE ASSIGNMENT IS COMPLETE, PERFORMING INFERENCE, AND CHECKING THE CONSISTENCY OF VARIABLE ASSIGNMENTS. THESE HELPER FUNCTIONS HELP IN MODULARIZING THE CODE AND IMPROVE CODE READABILITY.
#     1. THE `ASSIGNMENT` DICTIONARY IS INITIALIZED AS AN EMPTY DICTIONARY. THIS DICTIONARY WILL HOLD THE VARIABLE ASSIGNMENTS.
#     2. THE `STACK` IS INITIALIZED AS A LIST CONTAINING THE `ASSIGNMENT` DICTIONARY. THE STACK WILL BE USED TO KEEP TRACK OF THE ASSIGNMENTS MADE DURING THE BACKTRACKING PROCESS.
#     3. THE MAIN LOOP BEGINS, WHICH CONTINUES UNTIL THE STACK IS EMPTY. THIS LOOP REPRESENTS THE BACKTRACKING PROCESS.
#     4. INSIDE THE LOOP, THE CURRENT ASSIGNMENT IS RETRIEVED FROM THE TOP OF THE STACK USING `CURRENT_ASSIGNMENT = STACK.POP()`.
#     5. THE FUNCTION CHECKS IF THE ASSIGNMENT IS COMPLETE BY CALLING THE `__COMPLETE__` HELPER FUNCTION, WHICH ITERATES OVER THE VARIABLES IN THE CSP PROBLEM AND CHECKS IF ALL VARIABLES HAVE BEEN ASSIGNED VALUES. IF THE ASSIGNMENT IS COMPLETE, THE CURRENT ASSIGNMENT IS A VALID SOLUTION, SO IT IS RETURNED.
#     6. IF THE ASSIGNMENT IS NOT COMPLETE, THE FUNCTION SELECTS AN UNASSIGNED VARIABLE BY CALLING THE `__SELECT_UNASSIGNED_VARIABLE__` HELPER FUNCTION. THIS FUNCTION ITERATES OVER THE VARIABLES IN THE CSP PROBLEM AND RETURNS THE FIRST VARIABLE THAT HAS NOT BEEN ASSIGNED A VALUE YET.
#     7. IF THERE ARE NO UNASSIGNED VARIABLES (I.E., `VARIABLE` IS `NONE`), IT MEANS THAT THE CURRENT ASSIGNMENT IS NOT VALID AND CANNOT LEAD TO A SOLUTION. IN THIS CASE, THE FUNCTION CONTINUES TO THE NEXT ITERATION OF THE LOOP.
#     8. IF AN UNASSIGNED VARIABLE IS FOUND, THE FUNCTION ITERATES OVER THE DOMAIN VALUES OF THAT VARIABLE. THESE DOMAIN VALUES REPRESENT THE POSSIBLE VALUES THAT CAN BE ASSIGNED TO THE VARIABLE.
#     9. FOR EACH VALUE IN THE DOMAIN OF THE VARIABLE, THE FUNCTION CHECKS IF THE ASSIGNMENT IS CONSISTENT BY CALLING THE `__CONSISTENT__` HELPER FUNCTION. THIS FUNCTION CHECKS IF THE CURRENT ASSIGNMENT OF THE VARIABLE SATISFIES ALL CONSTRAINTS WITH OTHER ASSIGNED VARIABLES.
#     10. IF THE ASSIGNMENT IS CONSISTENT, A NEW ASSIGNMENT IS CREATED BY MAKING A COPY OF THE CURRENT ASSIGNMENT (`NEW_ASSIGNMENT = DICT(CURRENT_ASSIGNMENT)`) AND ASSIGNING THE SELECTED VALUE TO THE VARIABLE (`NEW_ASSIGNMENT[VARIABLE] = VALUE`).
#     11. INFERENCE IS PERFORMED BY CALLING THE `__INFERENCE__` HELPER FUNCTION, WHICH UPDATES THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT. THIS HELPS TO PRUNE UNNECESSARY ASSIGNMENTS AND REDUCE THE SEARCH SPACE.
#     12. THE NEW ASSIGNMENT IS PUSHED ONTO THE STACK (`STACK.APPEND(NEW_ASSIGNMENT)`) TO CONTINUE THE BACKTRACKING PROCESS.
#     13. THE LOOP CONTINUES UNTIL A SOLUTION IS FOUND OR UNTIL THE STACK BECOMES EMPTY (INDICATING THAT NO SOLUTION EXISTS).
#     14. IF THE LOOP FINISHES WITHOUT FINDING A SOLUTION, THE FUNCTION RETURNS `FALSE` TO INDICATE THAT NO SOLUTION EXISTS FOR THE GIVEN CSP PROBLEM.
# IN SUMMARY, THE `BACKTRACKING` FUNCTION USES A STACK-BASED BACKTRACKING APPROACH TO SYSTEMATICALLY EXPLORE POSSIBLE ASSIGNMENTS OF VALUES TO VARIABLES IN ORDER TO FIND A SOLUTION THAT SATISFIES ALL CONSTRAINTS. IT USES HELPER FUNCTIONS TO SELECT UNASSIGNED VARIABLES, CHECK ASSIGNMENT COMPLETENESS, PERFORM INFERENCE, AND ENSURE THE CONSISTENCY OF VARIABLE ASSIGNMENTS.
def BACKTRACKING(CSP_PROBLEM: CSP) -> dict | bool:
    """THE `BACKTRACKING` FUNCTION IS AN IMPLEMENTATION OF THE BACKTRACKING ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS). IT TAKES A CSP PROBLEM AS INPUT AND RETURNS EITHER A DICTIONARY REPRESENTING A VALID ASSIGNMENT OF VALUES TO VARIABLES THAT SATISFIES ALL CONSTRAINTS OR `FALSE` IF NO SOLUTION EXISTS."""

    # THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO SELECT AN UNASSIGNED VARIABLE FROM THE CSP PROBLEM.
    #     1. THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION TAKES ONE ARGUMENT: `ASSIGNMENT`, WHICH IS A DICTIONARY CONTAINING THE CURRENT VARIABLE ASSIGNMENTS.
    #     2. THE FUNCTION ITERATES OVER THE VARIABLES IN THE CSP PROBLEM.
    #     3. FOR EACH VARIABLE, IT CHECKS IF THE VARIABLE IS NOT PRESENT IN THE `ASSIGNMENT` DICTIONARY. IF THE VARIABLE IS NOT PRESENT, IT MEANS THAT THE VARIABLE HAS NOT BEEN ASSIGNED A VALUE YET.
    #     4. IF AN UNASSIGNED VARIABLE IS FOUND, THE FUNCTION RETURNS THAT VARIABLE.
    #     5. IF NO UNASSIGNED VARIABLE IS FOUND AFTER ITERATING OVER ALL VARIABLES, THE FUNCTION RETURNS `NONE` TO INDICATE THAT ALL VARIABLES HAVE BEEN ASSIGNED.
    # THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION SIMPLY FINDS THE FIRST VARIABLE THAT HAS NOT BEEN ASSIGNED A VALUE YET, PROVIDING A WAY TO SELECT AN UNASSIGNED VARIABLE FOR ASSIGNMENT IN THE BACKTRACKING ALGORITHM.
    def __SELECT_UNASSIGNED_VARIABLE__(ASSIGNMENT: dict) -> str | None:
        """THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO SELECT AN UNASSIGNED VARIABLE FROM THE CSP PROBLEM."""
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # ITERATE OVER VARIABLES
            if VARIABLE not in ASSIGNMENT:  # IF VARIABLE IS NOT IN ASSIGNMENT
                return VARIABLE  # RETURN VARIABLE
        return None  # IF NO UNASSIGNED VARIABLE IS FOUND, RETURN NONE

    # THE `__COMPLETE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO CHECK IF THE VARIABLE ASSIGNMENT IS COMPLETE FOR THE CSP PROBLEM.
    #     1. THE `__COMPLETE__` FUNCTION TAKES ONE ARGUMENT: `ASSIGNMENT`, WHICH IS A DICTIONARY CONTAINING THE CURRENT VARIABLE ASSIGNMENTS.
    #     2. THE FUNCTION ITERATES OVER THE VARIABLES IN THE CSP PROBLEM.
    #     3. FOR EACH VARIABLE, IT CHECKS IF THE VARIABLE IS NOT PRESENT IN THE `ASSIGNMENT` DICTIONARY. IF THE VARIABLE IS NOT PRESENT, IT MEANS THAT THE VARIABLE HAS NOT BEEN ASSIGNED A VALUE YET.
    #     4. IF ANY VARIABLE IS FOUND THAT HAS NOT BEEN ASSIGNED A VALUE, THE FUNCTION RETURNS `FALSE` TO INDICATE THAT THE ASSIGNMENT IS NOT COMPLETE.
    #     5. IF ALL VARIABLES HAVE BEEN ASSIGNED A VALUE (I.E., NONE OF THEM ARE MISSING IN THE `ASSIGNMENT` DICTIONARY), THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE ASSIGNMENT IS COMPLETE.
    # THE `__COMPLETE__` FUNCTION SIMPLY CHECKS IF ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES, PROVIDING A WAY TO DETERMINE IF THE ASSIGNMENT IS COMPLETE IN THE BACKTRACKING ALGORITHM.
    def __COMPLETE__(ASSIGNMENT: dict) -> bool:
        """THE `__COMPLETE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO CHECK IF THE VARIABLE ASSIGNMENT IS COMPLETE FOR THE CSP PROBLEM."""
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # ITERATE OVER VARIABLES
            if VARIABLE not in ASSIGNMENT:  # IF VARIABLE IS NOT IN ASSIGNMENT
                return False  # RETURN FALSE
        return True  # IF ALL VARIABLES ARE IN ASSIGNMENT, RETURN TRUE

    # THE `__INFERENCE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO PERFORM INFERENCE. IT UPDATES THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT OF A VARIABLE.
    #     1. THE `__INFERENCE__` FUNCTION TAKES TWO ARGUMENTS: `VARIABLE`, WHICH IS THE SELECTED VARIABLE THAT WAS ASSIGNED A VALUE, AND `VALUE`, WHICH IS THE ASSIGNED VALUE TO THE VARIABLE.
    #     2. THE FUNCTION ITERATES OVER THE NEIGHBORS OF THE `VARIABLE`. NEIGHBORS ARE THE VARIABLES THAT SHARE A CONSTRAINT WITH THE SELECTED VARIABLE.
    #     3. FOR EACH NEIGHBOR VARIABLE, IT CHECKS IF THE `VALUE` IS IN THE DOMAIN OF THE NEIGHBOR VARIABLE.
    #     4. IF THE `VALUE` IS IN THE DOMAIN OF THE NEIGHBOR VARIABLE, IT MEANS THAT THE DOMAIN OF THE NEIGHBOR NEEDS TO BE UPDATED.
    #     5. IN SUCH A CASE, THE FUNCTION REMOVES THE `VALUE` FROM THE DOMAIN OF THE NEIGHBOR VARIABLE, REDUCING THE SEARCH SPACE FOR THAT VARIABLE. THIS IS BASED ON THE CONSTRAINT THAT THE NEIGHBOR VARIABLE CANNOT HAVE THE SAME VALUE AS THE SELECTED VARIABLE.
    #     6. THE FUNCTION CONTINUES THIS PROCESS FOR ALL NEIGHBOR VARIABLES.
    # THE `__INFERENCE__` FUNCTION PERFORMS INFERENCE BY UPDATING THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT. THIS REDUCES THE SEARCH SPACE AND HELPS TO PRUNE UNNECESSARY ASSIGNMENTS, IMPROVING THE EFFICIENCY OF THE BACKTRACKING ALGORITHM.
    def __INFERENCE__(VARIABLE: str, VALUE: str) -> None:
        """THE `__INFERENCE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO PERFORM INFERENCE. IT UPDATES THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT OF A VARIABLE."""
        for NEIGHBOR in CSP_PROBLEM.NEIGHBORS[VARIABLE]:  # ITERATE OVER NEIGHBORS
            # IF VALUE IS IN DOMAIN OF NEIGHBOR
            if VALUE in CSP_PROBLEM.DOMAINS[NEIGHBOR]:
                # REMOVE VALUE FROM DOMAIN OF NEIGHBOR
                CSP_PROBLEM.DOMAINS[NEIGHBOR].remove(VALUE)

    # THE `__CONSISTENT__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO CHECK THE CONSISTENCY OF VARIABLE ASSIGNMENTS. IT ENSURES THAT THE CURRENT ASSIGNMENT OF A VARIABLE IS CONSISTENT WITH THE CONSTRAINTS OF THE CSP PROBLEM.
    #     1. THE `__CONSISTENT__` FUNCTION TAKES THREE ARGUMENTS: `ASSIGNMENT`, WHICH IS A DICTIONARY CONTAINING THE CURRENT VARIABLE ASSIGNMENTS, `VARIABLE`, WHICH IS THE SELECTED VARIABLE, AND `VALUE`, WHICH IS THE ASSIGNED VALUE TO THE VARIABLE.
    #     2. THE FUNCTION ITERATES OVER THE CONSTRAINTS OF THE CSP PROBLEM. CONSTRAINTS DEFINE RELATIONSHIPS BETWEEN VARIABLES.
    #     3. FOR EACH CONSTRAINT, IT CHECKS IF THE SELECTED VARIABLE `VARIABLE` IS INVOLVED IN THE CONSTRAINT.
    #     4. IF THE SELECTED VARIABLE IS INVOLVED, IT ITERATES OVER THE VARIABLES IN THE CONSTRAINT.
    #     5. FOR EACH VARIABLE `VARIABLE_2` IN THE CONSTRAINT, IT CHECKS IF `VARIABLE_2` IS NOT EQUAL TO `VARIABLE`.
    #     6. IF `VARIABLE_2` IS NOT EQUAL TO `VARIABLE`, IT CHECKS IF `VARIABLE_2` IS PRESENT IN THE `ASSIGNMENT` DICTIONARY, INDICATING THAT `VARIABLE_2` HAS BEEN ASSIGNED A VALUE.
    #     7. IF `VARIABLE_2` IS ASSIGNED A VALUE, IT COMPARES THE VALUE OF `VARIABLE_2` WITH THE ASSIGNED VALUE `VALUE`.
    #     8. IF THE VALUES ARE EQUAL, IT MEANS THAT THE ASSIGNMENT VIOLATES THE CONSTRAINT, AND THE FUNCTION RETURNS `FALSE` TO INDICATE INCONSISTENCY.
    #     9. IF NO INCONSISTENCIES ARE FOUND AMONG THE CONSTRAINTS, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE ASSIGNMENT IS CONSISTENT.
    # THE `__CONSISTENT__` FUNCTION ENSURES THAT THE CURRENT ASSIGNMENT OF A VARIABLE DOES NOT VIOLATE ANY CONSTRAINTS IN THE CSP PROBLEM. IT CHECKS FOR CONSISTENCY BETWEEN THE SELECTED VARIABLE AND OTHER ASSIGNED VARIABLES, PROVIDING A MECHANISM TO ENSURE THAT ASSIGNMENTS FOLLOW THE CONSTRAINTS OF THE PROBLEM DURING THE BACKTRACKING SEARCH.
    def __CONSISTENT__(ASSIGNMENT: dict, VARIABLE: str, VALUE: str) -> bool:
        """THE `__CONSISTENT__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO CHECK THE CONSISTENCY OF VARIABLE ASSIGNMENTS. IT ENSURES THAT THE CURRENT ASSIGNMENT OF A VARIABLE IS CONSISTENT WITH THE CONSTRAINTS OF THE CSP PROBLEM."""
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # ITERATE OVER CONSTRAINTS
            if VARIABLE in CONSTRAINT:  # IF VARIABLE IS IN CONSTRAINT
                for VARIABLE_2 in CONSTRAINT:  # ITERATE OVER VARIABLES IN CONSTRAINT
                    if VARIABLE_2 != VARIABLE:  # IF VARIABLE_2 IS NOT EQUAL TO VARIABLE
                        if VARIABLE_2 in ASSIGNMENT:  # IF VARIABLE_2 IS IN ASSIGNMENT
                            # IF VALUE IS EQUAL TO ASSIGNMENT[VARIABLE_2]
                            if VALUE == ASSIGNMENT[VARIABLE_2]:
                                return False  # RETURN FALSE
        return True  # RETURN TRUE

    ASSIGNMENT = {}  # INITIALIZE ASSIGNMENT
    STACK = [ASSIGNMENT]  # INITIALIZE STACK
    while STACK:  # WHILE STACK IS NOT EMPTY
        CURRENT_ASSIGNMENT = STACK.pop()  # POP ASSIGNMENT FROM STACK
        if __COMPLETE__(CURRENT_ASSIGNMENT):  # IF ASSIGNMENT IS COMPLETE
            return CURRENT_ASSIGNMENT  # RETURN ASSIGNMENT
        VARIABLE = __SELECT_UNASSIGNED_VARIABLE__(
            CURRENT_ASSIGNMENT)  # SELECT UNASSIGNED VARIABLE
        if VARIABLE is None:  # IF NO UNASSIGNED VARIABLE
            continue  # CONTINUE
        # ITERATE OVER DOMAIN OF VARIABLE
        for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
            # IF ASSIGNMENT IS CONSISTENT
            if __CONSISTENT__(CURRENT_ASSIGNMENT, VARIABLE, VALUE):
                NEW_ASSIGNMENT = dict(CURRENT_ASSIGNMENT)  # COPY ASSIGNMENT
                NEW_ASSIGNMENT[VARIABLE] = VALUE  # ASSIGN VALUE TO VARIABLE
                __INFERENCE__(VARIABLE, VALUE)  # PERFORM INFERENCE
                STACK.append(NEW_ASSIGNMENT)  # PUSH ASSIGNMENT TO STACK
    return False  # RETURN FALSE