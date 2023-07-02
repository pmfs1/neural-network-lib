from .CSP import CSP

# THE `BACKTRACKING` FUNCTION IS AN IMPLEMENTATION OF THE BACKTRACKING ALGORITHM USED TO SOLVE CONSTRAINT SATISFACTION PROBLEMS (CSPS).
#     1. THE `BACKTRACKING` FUNCTION TAKES A `CSP_PROBLEM` AS INPUT, WHICH IS AN INSTANCE OF THE CSP CLASS REPRESENTING THE PROBLEM TO BE SOLVED.
#     2. INSIDE THE FUNCTION, AN EMPTY `ASSIGNMENT` DICTIONARY IS INITIALIZED TO KEEP TRACK OF VARIABLE ASSIGNMENTS.
#     3. THE FUNCTION THEN CALLS THE PRIVATE HELPER FUNCTION `__BACKTRACK__` WITH THE `CSP_PROBLEM` AND `ASSIGNMENT` AS ARGUMENTS.
#     4. THE `__BACKTRACK__` FUNCTION IS THE MAIN RECURSIVE FUNCTION THAT PERFORMS THE BACKTRACKING SEARCH.
#     5. THE FIRST STEP IN `__BACKTRACK__` IS TO CHECK IF THE ASSIGNMENT IS COMPLETE BY CALLING THE PRIVATE HELPER FUNCTION `__COMPLETE__`. THIS FUNCTION CHECKS IF ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES. IF THE ASSIGNMENT IS COMPLETE, IT RETURNS THE `ASSIGNMENT` AS THE SOLUTION.
#     6. IF THE ASSIGNMENT IS NOT COMPLETE, THE FUNCTION SELECTS AN UNASSIGNED VARIABLE BY CALLING THE PRIVATE HELPER FUNCTION `__SELECT_UNASSIGNED_VARIABLE__`. THIS FUNCTION ITERATES OVER THE VARIABLES IN THE CSP PROBLEM AND RETURNS THE FIRST VARIABLE THAT IS NOT PRESENT IN THE `ASSIGNMENT` DICTIONARY.
#     7. ONCE AN UNASSIGNED VARIABLE IS SELECTED, THE FUNCTION ITERATES OVER THE DOMAIN OF THAT VARIABLE AND TRIES TO ASSIGN A VALUE TO IT. THIS IS DONE USING A LOOP THAT ITERATES OVER `CSP_PROBLEM.DOMAINS[VARIABLE]`, WHERE `VARIABLE` IS THE SELECTED UNASSIGNED VARIABLE.
#     8. BEFORE MAKING AN ASSIGNMENT, THE FUNCTION CHECKS IF THE ASSIGNMENT WOULD BE CONSISTENT WITH THE CONSTRAINTS OF THE CSP PROBLEM. IT DOES THIS BY CALLING THE PRIVATE HELPER FUNCTION `__CONSISTENT__`, WHICH CHECKS IF THE VALUE ASSIGNMENT VIOLATES ANY CONSTRAINTS. IF THE ASSIGNMENT IS NOT CONSISTENT, THE FUNCTION MOVES ON TO THE NEXT VALUE IN THE VARIABLE'S DOMAIN.
#     9. IF THE ASSIGNMENT IS CONSISTENT, THE VALUE IS ADDED TO THE `ASSIGNMENT` DICTIONARY FOR THE SELECTED VARIABLE. THEN, THE FUNCTION PERFORMS INFERENCE BY CALLING THE PRIVATE HELPER FUNCTION `__INFERENCE__`. INFERENCE INVOLVES UPDATING THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT TO REDUCE THE SEARCH SPACE.
#     10. AFTER PERFORMING INFERENCE, THE FUNCTION MAKES A RECURSIVE CALL TO `__BACKTRACK__` WITH THE UPDATED `CSP_PROBLEM` AND `ASSIGNMENT`. THIS RECURSIVE CALL CONTINUES THE SEARCH FOR A SOLUTION BY ASSIGNING VALUES TO THE REMAINING UNASSIGNED VARIABLES.
#     11. IF THE RECURSIVE CALL RETURNS A VALID ASSIGNMENT (I.E., NOT `FALSE`), IT MEANS A SOLUTION HAS BEEN FOUND. IN THIS CASE, THE ASSIGNMENT IS PROPAGATED UP THE RECURSIVE CALL STACK UNTIL IT REACHES THE INITIAL CALL TO `BACKTRACKING`, WHICH THEN RETURNS THE SOLUTION.
#     12. IF THE RECURSIVE CALL RETURNS `FALSE`, IT MEANS THAT NO CONSISTENT ASSIGNMENT CAN BE FOUND WITH THE CURRENT VARIABLE ASSIGNMENT. IN THIS CASE, THE FUNCTION BACKTRACKS BY REMOVING THE CURRENT VARIABLE ASSIGNMENT FROM THE `ASSIGNMENT` DICTIONARY AND UNDOING THE PREVIOUS INFERENCE STEP BY CALLING THE PRIVATE HELPER FUNCTION `__UNDO_INFERENCE__`.
#     13. THE FUNCTION CONTINUES THE LOOP, TRYING THE NEXT VALUE IN THE DOMAIN OF THE SELECTED VARIABLE. IF ALL VALUES IN THE DOMAIN HAVE BEEN TRIED WITHOUT FINDING A CONSISTENT ASSIGNMENT, THE FUNCTION RETURNS `FALSE` TO INDICATE FAILURE.
# THE BACKTRACKING ALGORITHM SYSTEMATICALLY EXPLORES THE SEARCH SPACE OF VARIABLE ASSIGNMENTS, TRYING DIFFERENT COMBINATIONS UNTIL A VALID ASSIGNMENT IS FOUND OR ALL POSSIBILITIES HAVE BEEN EXHAUSTED. IT USES THE PRINCIPLE OF DEPTH-FIRST SEARCH AND BACKTRACKING TO EFFICIENTLY NAVIGATE THE SEARCH SPACE AND FIND SOLUTIONS TO CSPS.
def BACKTRACKING(CSP_PROBLEM: CSP) -> dict | bool:
    """THE `BACKTRACKING` FUNCTION IS AN IMPLEMENTATION OF THE BACKTRACKING ALGORITHM USED TO SOLVE CONSTRAINT SATISFACTION PROBLEMS (CSPS)."""
    ASSIGNMENT = {}  # INITIALIZE EMPTY ASSIGNMENT DICTIONARY
    # CALL PRIVATE HELPER FUNCTION TO PERFORM BACKTRACKING SEARCH
    return __BACKTRACK__(CSP_PROBLEM, ASSIGNMENT)

# THE `__BACKTRACK__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO PERFORM THE BACKTRACKING SEARCH. IT RECURSIVELY EXPLORES THE SEARCH SPACE OF VARIABLE ASSIGNMENTS AND TRIES TO FIND A VALID ASSIGNMENT FOR THE GIVEN CSP PROBLEM.
#     1. THE `__BACKTRACK__` FUNCTION TAKES TWO ARGUMENTS: `CSP_PROBLEM`, WHICH IS AN INSTANCE OF THE CSP CLASS REPRESENTING THE PROBLEM, AND `ASSIGNMENT`, WHICH IS A DICTIONARY CONTAINING THE CURRENT VARIABLE ASSIGNMENTS.
#     2. THE FIRST STEP IN `__BACKTRACK__` IS TO CHECK IF THE ASSIGNMENT IS COMPLETE. THIS IS DONE BY CALLING THE PRIVATE HELPER FUNCTION `__COMPLETE__` WITH `CSP_PROBLEM` AND `ASSIGNMENT` AS ARGUMENTS. THE `__COMPLETE__` FUNCTION CHECKS IF ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES. IF THE ASSIGNMENT IS COMPLETE, IT RETURNS THE `ASSIGNMENT` AS THE SOLUTION.
#     3. IF THE ASSIGNMENT IS NOT COMPLETE, THE FUNCTION PROCEEDS TO SELECT AN UNASSIGNED VARIABLE. THIS IS DONE BY CALLING THE PRIVATE HELPER FUNCTION `__SELECT_UNASSIGNED_VARIABLE__` WITH `CSP_PROBLEM` AND `ASSIGNMENT` AS ARGUMENTS. THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION ITERATES OVER THE VARIABLES IN THE CSP PROBLEM AND RETURNS THE FIRST VARIABLE THAT IS NOT PRESENT IN THE `ASSIGNMENT` DICTIONARY.
#     4. ONCE AN UNASSIGNED VARIABLE IS SELECTED, THE FUNCTION ENTERS A LOOP THAT ITERATES OVER THE DOMAIN OF THE SELECTED VARIABLE. THE DOMAIN OF A VARIABLE REPRESENTS THE POSSIBLE VALUES IT CAN TAKE. THE LOOP ITERATES OVER `CSP_PROBLEM.DOMAINS[VARIABLE]`, WHERE `VARIABLE` IS THE SELECTED UNASSIGNED VARIABLE.
#     5. WITHIN THE LOOP, THE FUNCTION CHECKS IF THE ASSIGNMENT OF THE CURRENT VALUE TO THE SELECTED VARIABLE IS CONSISTENT WITH THE CONSTRAINTS OF THE CSP PROBLEM. THIS IS DONE BY CALLING THE PRIVATE HELPER FUNCTION `__CONSISTENT__` WITH `CSP_PROBLEM`, `ASSIGNMENT`, THE SELECTED VARIABLE, AND THE CURRENT VALUE AS ARGUMENTS. THE `__CONSISTENT__` FUNCTION CHECKS IF THE ASSIGNMENT VIOLATES ANY CONSTRAINTS. IF THE ASSIGNMENT IS NOT CONSISTENT, THE FUNCTION MOVES ON TO THE NEXT VALUE IN THE VARIABLE'S DOMAIN.
#     6. IF THE ASSIGNMENT IS CONSISTENT, THE VALUE IS ADDED TO THE `ASSIGNMENT` DICTIONARY FOR THE SELECTED VARIABLE. THEN, THE FUNCTION PERFORMS INFERENCE BY CALLING THE PRIVATE HELPER FUNCTION `__INFERENCE__` WITH `CSP_PROBLEM`, THE SELECTED VARIABLE, AND THE ASSIGNED VALUE AS ARGUMENTS. INFERENCE INVOLVES UPDATING THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT TO REDUCE THE SEARCH SPACE.
#     7. AFTER PERFORMING INFERENCE, THE FUNCTION MAKES A RECURSIVE CALL TO `__BACKTRACK__` WITH THE UPDATED `CSP_PROBLEM` AND `ASSIGNMENT`. THIS RECURSIVE CALL CONTINUES THE SEARCH FOR A SOLUTION BY ASSIGNING VALUES TO THE REMAINING UNASSIGNED VARIABLES.
#     8. IF THE RECURSIVE CALL RETURNS A VALID ASSIGNMENT (I.E., NOT `FALSE`), IT MEANS A SOLUTION HAS BEEN FOUND. IN THIS CASE, THE ASSIGNMENT IS PROPAGATED UP THE RECURSIVE CALL STACK UNTIL IT REACHES THE INITIAL CALL TO `BACKTRACKING`, WHICH THEN RETURNS THE SOLUTION.
#     9. IF THE RECURSIVE CALL RETURNS `FALSE`, IT MEANS THAT NO CONSISTENT ASSIGNMENT CAN BE FOUND WITH THE CURRENT VARIABLE ASSIGNMENT. IN THIS CASE, THE FUNCTION BACKTRACKS BY REMOVING THE CURRENT VARIABLE ASSIGNMENT FROM THE `ASSIGNMENT` DICTIONARY AND UNDOING THE PREVIOUS INFERENCE STEP BY CALLING THE PRIVATE HELPER FUNCTION `__UNDO_INFERENCE__` WITH `CSP_PROBLEM`, THE SELECTED VARIABLE, AND THE ASSIGNED VALUE AS ARGUMENTS.
#     10. THE FUNCTION CONTINUES THE LOOP, TRYING THE NEXT VALUE IN THE DOMAIN OF THE SELECTED VARIABLE. IF ALL VALUES IN THE DOMAIN HAVE BEEN TRIED WITHOUT FINDING A CONSISTENT ASSIGNMENT, THE FUNCTION RETURNS `FALSE` TO INDICATE FAILURE.
# THE `__BACKTRACK__` FUNCTION COMBINES THE PRINCIPLES OF DEPTH-FIRST SEARCH AND BACKTRACKING TO EXPLORE THE SEARCH SPACE OF VARIABLE ASSIGNMENTS, BACKTRACKING WHEN NECESSARY TO FIND A VALID ASSIGNMENT FOR THE CSP PROBLEM.
def __BACKTRACK__(CSP_PROBLEM: CSP, ASSIGNMENT: dict) -> dict | bool:
    """THE `__BACKTRACK__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `BACKTRACKING` FUNCTION TO PERFORM THE BACKTRACKING SEARCH. IT RECURSIVELY EXPLORES THE SEARCH SPACE OF VARIABLE ASSIGNMENTS AND TRIES TO FIND A VALID ASSIGNMENT FOR THE GIVEN CSP PROBLEM."""
    if __COMPLETE__(CSP_PROBLEM, ASSIGNMENT):  # CHECK IF ASSIGNMENT IS COMPLETE
        return ASSIGNMENT  # IF ASSIGNMENT IS COMPLETE, RETURN ASSIGNMENT
    VARIABLE = __SELECT_UNASSIGNED_VARIABLE__(
        CSP_PROBLEM, ASSIGNMENT)  # SELECT AN UNASSIGNED VARIABLE
    if VARIABLE is None:  # IF NO UNASSIGNED VARIABLE IS FOUND
        return False  # RETURN FALSE TO INDICATE FAILURE
    # ITERATE OVER THE DOMAIN OF THE SELECTED VARIABLE
    for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:
        # CHECK IF ASSIGNMENT IS CONSISTENT
        if __CONSISTENT__(CSP_PROBLEM, ASSIGNMENT, VARIABLE, VALUE):
            # IF ASSIGNMENT IS CONSISTENT, ADD VALUE TO ASSIGNMENT
            ASSIGNMENT[VARIABLE] = VALUE
            __INFERENCE__(CSP_PROBLEM, VARIABLE, VALUE)  # PERFORM INFERENCE
            # MAKE RECURSIVE CALL TO BACKTRACK
            RESULT = __BACKTRACK__(CSP_PROBLEM, ASSIGNMENT)
            if RESULT is not False:  # IF RECURSIVE CALL RETURNS A VALID ASSIGNMENT
                return RESULT  # RETURN ASSIGNMENT
            # IF RECURSIVE CALL RETURNS FALSE, BACKTRACK
            del ASSIGNMENT[VARIABLE]
            __UNDO_INFERENCE__(CSP_PROBLEM, VARIABLE, VALUE)  # UNDO INFERENCE
    return False  # IF NO ASSIGNMENT IS FOUND, RETURN FALSE

# THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO SELECT AN UNASSIGNED VARIABLE FROM THE CSP PROBLEM.
#     1. THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION TAKES TWO ARGUMENTS: `CSP_PROBLEM`, WHICH IS AN INSTANCE OF THE CSP CLASS REPRESENTING THE PROBLEM, AND `ASSIGNMENT`, WHICH IS A DICTIONARY CONTAINING THE CURRENT VARIABLE ASSIGNMENTS.
#     2. THE FUNCTION ITERATES OVER THE VARIABLES IN THE CSP PROBLEM.
#     3. FOR EACH VARIABLE, IT CHECKS IF THE VARIABLE IS NOT PRESENT IN THE `ASSIGNMENT` DICTIONARY. IF THE VARIABLE IS NOT PRESENT, IT MEANS THAT THE VARIABLE HAS NOT BEEN ASSIGNED A VALUE YET.
#     4. IF AN UNASSIGNED VARIABLE IS FOUND, THE FUNCTION RETURNS THAT VARIABLE.
#     5. IF NO UNASSIGNED VARIABLE IS FOUND AFTER ITERATING OVER ALL VARIABLES, THE FUNCTION RETURNS `NONE` TO INDICATE THAT ALL VARIABLES HAVE BEEN ASSIGNED.
# THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION SIMPLY FINDS THE FIRST VARIABLE THAT HAS NOT BEEN ASSIGNED A VALUE YET, PROVIDING A WAY TO SELECT AN UNASSIGNED VARIABLE FOR ASSIGNMENT IN THE BACKTRACKING ALGORITHM.
def __SELECT_UNASSIGNED_VARIABLE__(CSP_PROBLEM: CSP, ASSIGNMENT: dict) -> str | None:
    """THE `__SELECT_UNASSIGNED_VARIABLE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO SELECT AN UNASSIGNED VARIABLE FROM THE CSP PROBLEM."""
    for VARIABLE in CSP_PROBLEM.VARIABLES:  # ITERATE OVER VARIABLES
        if VARIABLE not in ASSIGNMENT:  # IF VARIABLE IS NOT IN ASSIGNMENT
            return VARIABLE  # RETURN VARIABLE
    return None  # IF NO UNASSIGNED VARIABLE IS FOUND, RETURN NONE

# THE `__COMPLETE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO CHECK IF THE VARIABLE ASSIGNMENT IS COMPLETE FOR THE CSP PROBLEM.
#     1. THE `__COMPLETE__` FUNCTION TAKES TWO ARGUMENTS: `CSP_PROBLEM`, WHICH IS AN INSTANCE OF THE CSP CLASS REPRESENTING THE PROBLEM, AND `ASSIGNMENT`, WHICH IS A DICTIONARY CONTAINING THE CURRENT VARIABLE ASSIGNMENTS.
#     2. THE FUNCTION ITERATES OVER THE VARIABLES IN THE CSP PROBLEM.
#     3. FOR EACH VARIABLE, IT CHECKS IF THE VARIABLE IS NOT PRESENT IN THE `ASSIGNMENT` DICTIONARY. IF THE VARIABLE IS NOT PRESENT, IT MEANS THAT THE VARIABLE HAS NOT BEEN ASSIGNED A VALUE YET.
#     4. IF ANY VARIABLE IS FOUND THAT HAS NOT BEEN ASSIGNED A VALUE, THE FUNCTION RETURNS `FALSE` TO INDICATE THAT THE ASSIGNMENT IS NOT COMPLETE.
#     5. IF ALL VARIABLES HAVE BEEN ASSIGNED A VALUE (I.E., NONE OF THEM ARE MISSING IN THE `ASSIGNMENT` DICTIONARY), THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE ASSIGNMENT IS COMPLETE.
# THE `__COMPLETE__` FUNCTION SIMPLY CHECKS IF ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES, PROVIDING A WAY TO DETERMINE IF THE ASSIGNMENT IS COMPLETE IN THE BACKTRACKING ALGORITHM.
def __COMPLETE__(CSP_PROBLEM: CSP, ASSIGNMENT: dict) -> bool:
    """THE `__COMPLETE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO CHECK IF THE VARIABLE ASSIGNMENT IS COMPLETE FOR THE CSP PROBLEM."""
    for VARIABLE in CSP_PROBLEM.VARIABLES:  # ITERATE OVER VARIABLES
        if VARIABLE not in ASSIGNMENT:  # IF VARIABLE IS NOT IN ASSIGNMENT
            return False  # RETURN FALSE
    return True  # IF ALL VARIABLES ARE IN ASSIGNMENT, RETURN TRUE

# THE `__UNDO_INFERENCE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO UNDO THE INFERENCE STEP. IT REVERSES THE EFFECTS OF INFERENCE BY RESTORING THE DOMAIN VALUES OF NEIGHBORING VARIABLES THAT WERE MODIFIED DURING INFERENCE.
#     1. THE `__UNDO_INFERENCE__` FUNCTION TAKES THREE ARGUMENTS: `CSP_PROBLEM`, WHICH IS AN INSTANCE OF THE CSP CLASS REPRESENTING THE PROBLEM, `VARIABLE`, WHICH IS THE SELECTED VARIABLE THAT WAS ASSIGNED A VALUE, AND `VALUE`, WHICH IS THE ASSIGNED VALUE TO THE VARIABLE.
#     2. THE FUNCTION ITERATES OVER THE NEIGHBORS OF THE `VARIABLE`. NEIGHBORS ARE THE VARIABLES THAT SHARE A CONSTRAINT WITH THE SELECTED VARIABLE.
#     3. FOR EACH NEIGHBOR VARIABLE, IT CHECKS IF THE `VALUE` IS IN THE DOMAIN OF THE NEIGHBOR VARIABLE.
#     4. IF THE `VALUE` IS NOT IN THE DOMAIN OF THE NEIGHBOR VARIABLE, IT MEANS THAT THE DOMAIN OF THE NEIGHBOR WAS MODIFIED DURING INFERENCE.
#     5. IN SUCH A CASE, THE FUNCTION APPENDS THE `VALUE` BACK TO THE DOMAIN OF THE NEIGHBOR VARIABLE, RESTORING THE DOMAIN TO ITS ORIGINAL STATE.
#     6. THE FUNCTION CONTINUES THIS PROCESS FOR ALL NEIGHBOR VARIABLES.
# THE `__UNDO_INFERENCE__` FUNCTION UNDOES THE INFERENCE STEP BY RESTORING THE ORIGINAL DOMAIN VALUES OF NEIGHBOR VARIABLES THAT WERE MODIFIED DURING INFERENCE. THIS ALLOWS THE BACKTRACKING ALGORITHM TO BACKTRACK AND EXPLORE OTHER POSSIBILITIES AFTER A FAILED ASSIGNMENT.
def __UNDO_INFERENCE__(CSP_PROBLEM: CSP, VARIABLE: str, VALUE: str) -> None:
    """THE `__UNDO_INFERENCE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO UNDO THE INFERENCE STEP. IT REVERSES THE EFFECTS OF INFERENCE BY RESTORING THE DOMAIN VALUES OF NEIGHBORING VARIABLES THAT WERE MODIFIED DURING INFERENCE."""
    for NEIGHBOR in CSP_PROBLEM.NEIGHBORS[VARIABLE]:  # ITERATE OVER NEIGHBORS
        # IF VALUE IS IN DOMAIN OF NEIGHBOR
        if VALUE in CSP_PROBLEM.DOMAINS[NEIGHBOR]:
            # APPEND VALUE TO DOMAIN OF NEIGHBOR
            CSP_PROBLEM.DOMAINS[NEIGHBOR].append(VALUE)

# THE `__INFERENCE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO PERFORM INFERENCE. IT UPDATES THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT OF A VARIABLE.
#     1. THE `__INFERENCE__` FUNCTION TAKES THREE ARGUMENTS: `CSP_PROBLEM`, WHICH IS AN INSTANCE OF THE CSP CLASS REPRESENTING THE PROBLEM, `VARIABLE`, WHICH IS THE SELECTED VARIABLE THAT WAS ASSIGNED A VALUE, AND `VALUE`, WHICH IS THE ASSIGNED VALUE TO THE VARIABLE.
#     2. THE FUNCTION ITERATES OVER THE NEIGHBORS OF THE `VARIABLE`. NEIGHBORS ARE THE VARIABLES THAT SHARE A CONSTRAINT WITH THE SELECTED VARIABLE.
#     3. FOR EACH NEIGHBOR VARIABLE, IT CHECKS IF THE `VALUE` IS IN THE DOMAIN OF THE NEIGHBOR VARIABLE.
#     4. IF THE `VALUE` IS IN THE DOMAIN OF THE NEIGHBOR VARIABLE, IT MEANS THAT THE DOMAIN OF THE NEIGHBOR NEEDS TO BE UPDATED.
#     5. IN SUCH A CASE, THE FUNCTION REMOVES THE `VALUE` FROM THE DOMAIN OF THE NEIGHBOR VARIABLE, REDUCING THE SEARCH SPACE FOR THAT VARIABLE. THIS IS BASED ON THE CONSTRAINT THAT THE NEIGHBOR VARIABLE CANNOT HAVE THE SAME VALUE AS THE SELECTED VARIABLE.
#     6. THE FUNCTION CONTINUES THIS PROCESS FOR ALL NEIGHBOR VARIABLES.
# THE `__INFERENCE__` FUNCTION PERFORMS INFERENCE BY UPDATING THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT. THIS REDUCES THE SEARCH SPACE AND HELPS TO PRUNE UNNECESSARY ASSIGNMENTS, IMPROVING THE EFFICIENCY OF THE BACKTRACKING ALGORITHM.
def __INFERENCE__(CSP_PROBLEM: CSP, VARIABLE: str, VALUE: str) -> None:
    """THE `__INFERENCE__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO PERFORM INFERENCE. IT UPDATES THE DOMAINS OF NEIGHBORING VARIABLES BASED ON THE CURRENT ASSIGNMENT OF A VARIABLE."""
    for NEIGHBOR in CSP_PROBLEM.NEIGHBORS[VARIABLE]:  # ITERATE OVER NEIGHBORS
        # IF VALUE IS IN DOMAIN OF NEIGHBOR
        if VALUE in CSP_PROBLEM.DOMAINS[NEIGHBOR]:
            # REMOVE VALUE FROM DOMAIN OF NEIGHBOR
            CSP_PROBLEM.DOMAINS[NEIGHBOR].remove(VALUE)

# THE `__CONSISTENT__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO CHECK THE CONSISTENCY OF VARIABLE ASSIGNMENTS. IT ENSURES THAT THE CURRENT ASSIGNMENT OF A VARIABLE IS CONSISTENT WITH THE CONSTRAINTS OF THE CSP PROBLEM.
#     1. THE `__CONSISTENT__` FUNCTION TAKES FOUR ARGUMENTS: `CSP_PROBLEM`, WHICH IS AN INSTANCE OF THE CSP CLASS REPRESENTING THE PROBLEM, `ASSIGNMENT`, WHICH IS A DICTIONARY CONTAINING THE CURRENT VARIABLE ASSIGNMENTS, `VARIABLE`, WHICH IS THE SELECTED VARIABLE, AND `VALUE`, WHICH IS THE ASSIGNED VALUE TO THE VARIABLE.
#     2. THE FUNCTION ITERATES OVER THE CONSTRAINTS OF THE CSP PROBLEM. CONSTRAINTS DEFINE RELATIONSHIPS BETWEEN VARIABLES.
#     3. FOR EACH CONSTRAINT, IT CHECKS IF THE SELECTED VARIABLE `VARIABLE` IS INVOLVED IN THE CONSTRAINT.
#     4. IF THE SELECTED VARIABLE IS INVOLVED, IT ITERATES OVER THE VARIABLES IN THE CONSTRAINT.
#     5. FOR EACH VARIABLE `VARIABLE_2` IN THE CONSTRAINT, IT CHECKS IF `VARIABLE_2` IS NOT EQUAL TO `VARIABLE`.
#     6. IF `VARIABLE_2` IS NOT EQUAL TO `VARIABLE`, IT CHECKS IF `VARIABLE_2` IS PRESENT IN THE `ASSIGNMENT` DICTIONARY, INDICATING THAT `VARIABLE_2` HAS BEEN ASSIGNED A VALUE.
#     7. IF `VARIABLE_2` IS ASSIGNED A VALUE, IT COMPARES THE VALUE OF `VARIABLE_2` WITH THE ASSIGNED VALUE `VALUE`.
#     8. IF THE VALUES ARE EQUAL, IT MEANS THAT THE ASSIGNMENT VIOLATES THE CONSTRAINT, AND THE FUNCTION RETURNS `FALSE` TO INDICATE INCONSISTENCY.
#     9. IF NO INCONSISTENCIES ARE FOUND AMONG THE CONSTRAINTS, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE ASSIGNMENT IS CONSISTENT.
# THE `__CONSISTENT__` FUNCTION ENSURES THAT THE CURRENT ASSIGNMENT OF A VARIABLE DOES NOT VIOLATE ANY CONSTRAINTS IN THE CSP PROBLEM. IT CHECKS FOR CONSISTENCY BETWEEN THE SELECTED VARIABLE AND OTHER ASSIGNED VARIABLES, PROVIDING A MECHANISM TO ENSURE THAT ASSIGNMENTS FOLLOW THE CONSTRAINTS OF THE PROBLEM DURING THE BACKTRACKING SEARCH.
def __CONSISTENT__(CSP_PROBLEM: CSP, ASSIGNMENT: dict, VARIABLE: str, VALUE: str) -> bool:
    """THE `__CONSISTENT__` FUNCTION IS A PRIVATE HELPER FUNCTION USED WITHIN THE `__BACKTRACK__` FUNCTION TO CHECK THE CONSISTENCY OF VARIABLE ASSIGNMENTS. IT ENSURES THAT THE CURRENT ASSIGNMENT OF A VARIABLE IS CONSISTENT WITH THE CONSTRAINTS OF THE CSP PROBLEM."""
    for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # ITERATE OVER CONSTRAINTS
        if VARIABLE in CONSTRAINT:  # IF VARIABLE IS IN CONSTRAINT
            for VARIABLE_2 in CONSTRAINT:  # ITERATE OVER VARIABLES IN CONSTRAINT
                if VARIABLE_2 != VARIABLE:  # IF VARIABLE_2 IS NOT EQUAL TO VARIABLE
                    if VARIABLE_2 in ASSIGNMENT:  # IF VARIABLE_2 IS IN ASSIGNMENT
                        # IF VALUE IS EQUAL TO ASSIGNMENT[VARIABLE_2]
                        if VALUE == ASSIGNMENT[VARIABLE_2]:
                            return False  # RETURN FALSE
    return True  # RETURN TRUE
