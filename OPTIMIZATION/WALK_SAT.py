import numpy as np
from .CSP import CSP

# THE `WALK_SAT()` FUNCTION IMPLEMENTS THE WALKSAT ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS).
#     1. THE FUNCTION TAKES THREE PARAMETERS:
#       - `CSP_PROBLEM`: AN INSTANCE OF THE `CSP` CLASS REPRESENTING THE CSP TO BE SOLVED.
#       - `MAX_FLIPS`: THE MAXIMUM NUMBER OF VARIABLE FLIPS ALLOWED IN EACH RESTART.
#       - `MAX_RESTARTS`: THE MAXIMUM NUMBER OF RESTARTS ALLOWED IN THE ALGORITHM.
#     2. INSIDE THE FUNCTION, AN EMPTY DICTIONARY CALLED `ASSIGNMENT` AND AN EMPTY LIST CALLED `CLAUSES` ARE INITIALIZED.
#     3. THE `CLAUSES` LIST IS POPULATED BY ITERATING OVER EACH `CONSTRAINT` IN `CSP_PROBLEM.CONSTRAINTS`. EACH `CONSTRAINT` IS CONVERTED INTO A LIST OF LITERALS (VARIABLES), AND THE RESULTING LIST IS APPENDED TO `CLAUSES`.
#     4. THE FUNCTION THEN CALLS A HELPER FUNCTION CALLED `WALK_SAT` WITH THE NECESSARY PARAMETERS (`CSP_PROBLEM`, `ASSIGNMENT`, `MAX_FLIPS`, `MAX_RESTARTS`, AND `CLAUSES`).
#     5. THE `WALK_SAT` FUNCTION BEGINS BY INITIALIZING THE `ASSIGNMENT` DICTIONARY. IT RANDOMLY ASSIGNS A VALUE FROM THE DOMAIN OF EACH VARIABLE IN `CSP_PROBLEM.VARIABLES` TO INITIALIZE THE ASSIGNMENT.
#     6. THE ALGORITHM THEN ENTERS A LOOP THAT REPEATS FOR A MAXIMUM OF `MAX_RESTARTS` TIMES. THIS LOOP IS RESPONSIBLE FOR RESTARTING THE ALGORITHM IF A SATISFYING ASSIGNMENT IS NOT FOUND.
#     7. WITHIN EACH RESTART, ANOTHER LOOP IS EXECUTED FOR A MAXIMUM OF `MAX_FLIPS` TIMES. THIS LOOP PERFORMS VARIABLE FLIPS TO TRY AND IMPROVE THE ASSIGNMENT.
#     8. AT EACH ITERATION OF THE INNER LOOP, THE ALGORITHM CHECKS IF THE ASSIGNMENT SATISFIES ALL THE CONSTRAINTS IN THE CSP. THIS IS DONE BY CALLING A HELPER FUNCTION CALLED `__COMPLETE__`.
#     9. IF THE ASSIGNMENT IS COMPLETE AND SATISFIES ALL THE CONSTRAINTS, IT IS A SATISFYING ASSIGNMENT, AND IT IS RETURNED.
#     10. IF THE ASSIGNMENT IS NOT COMPLETE, THE ALGORITHM SELECTS A CLAUSE THAT IS NOT SATISFIED BY THE CURRENT ASSIGNMENT. THIS IS DONE BY CALLING THE `__SELECT_CLAUSE__` FUNCTION, WHICH ITERATES THROUGH THE LIST OF CLAUSES AND RETURNS THE FIRST UNSATISFIED CLAUSE IT ENCOUNTERS.
#     11. ONCE A CLAUSE IS SELECTED, THE ALGORITHM CHOOSES A VARIABLE FROM THAT CLAUSE THAT IS NOT ALREADY ASSIGNED A VALUE. THIS IS DONE USING THE `__SELECT_VARIABLE__` FUNCTION, WHICH ITERATES THROUGH THE VARIABLES IN THE CLAUSE AND RETURNS THE FIRST UNASSIGNED VARIABLE IT FINDS.
#     12. NEXT, THE ALGORITHM SELECTS A VALUE FROM THE DOMAIN OF THE SELECTED VARIABLE THAT SATISFIES THE CLAUSE WHEN ASSIGNED. THIS IS DONE BY CALLING THE `__SELECT_VALUE__` FUNCTION, WHICH ITERATES THROUGH THE VALUES IN THE VARIABLE'S DOMAIN AND RETURNS THE FIRST VALUE THAT SATISFIES THE CLAUSE.
#     13. THE SELECTED VALUE IS THEN ASSIGNED TO THE SELECTED VARIABLE IN THE `ASSIGNMENT` DICTIONARY.
#     14. STEPS 10-13 ARE REPEATED UNTIL EITHER A SATISFYING ASSIGNMENT IS FOUND OR THE MAXIMUM NUMBER OF FLIPS IS REACHED.
#     15. IF A SATISFYING ASSIGNMENT IS FOUND, IT IS RETURNED.
#     16. IF NO SATISFYING ASSIGNMENT IS FOUND WITHIN THE ALLOWED NUMBER OF RESTARTS, THE FUNCTION RETURNS `FALSE` TO INDICATE THAT NO SOLUTION WAS FOUND.
#     17. THE `WALK_SAT()` FUNCTION SERVES AS A WRAPPER FUNCTION THAT PREPARES THE NECESSARY DATA STRUCTURES AND PARAMETERS AND CALLS THE `WALK_SAT()` FUNCTION TO PERFORM THE ACTUAL WALKSAT ALGORITHM.
# IN SUMMARY, THE WALKSAT ALGORITHM STARTS WITH A RANDOM ASSIGNMENT AND ITERATIVELY TRIES TO IMPROVE IT BY FLIPPING THE VALUES OF VARIABLES THAT VIOLATE CONSTRAINTS. THE ALGORITHM USES RANDOMIZATION AND RESTARTS TO EXPLORE DIFFERENT PARTS OF THE SEARCH SPACE AND POTENTIALLY FIND A SATISFYING ASSIGNMENT FOR THE GIVEN CSP.
def WALK_SAT(CSP_PROBLEM: CSP, MAX_FLIPS: int, MAX_RESTARTS: int) -> dict | bool:
    """THE `WALK_SAT()` FUNCTION IMPLEMENTS THE WALKSAT ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS)."""
    
    # THE `__SELECT_CLAUSE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO SELECT AN UNSATISFIED CLAUSE FROM A LIST OF CLAUSES.
    #     1. THE FUNCTION TAKES THREE PARAMETERS:
    #         - `ASSIGNMENT`: A DICTIONARY REPRESENTING THE CURRENT VARIABLE ASSIGNMENT.
    #         - `CLAUSES`: A LIST OF CLAUSES, WHERE EACH CLAUSE IS A LIST OF LITERALS (VARIABLES).
    #     2. THE FUNCTION ITERATES THROUGH EACH CLAUSE IN THE `CLAUSES` LIST.
    #     3. FOR EACH CLAUSE, IT CHECKS IF THE CLAUSE IS SATISFIED BY THE CURRENT ASSIGNMENT USING A HELPER FUNCTION CALLED `__SATISFIED__()`.
    #     4. IF THE CLAUSE IS NOT SATISFIED BY THE CURRENT ASSIGNMENT, IT MEANS THAT THE CLAUSE IS UNSATISFIED. IN THIS CASE, THE FUNCTION IMMEDIATELY RETURNS THE UNSATISFIED CLAUSE.
    #     5. IF ALL CLAUSES HAVE BEEN ITERATED THROUGH AND NO UNSATISFIED CLAUSE IS FOUND, THE FUNCTION RANDOMLY SELECTS A CLAUSE FROM THE `CLAUSES` LIST AND RETURNS IT.
    # IN SUMMARY, THE `__SELECT_CLAUSE__()` FUNCTION SCANS THROUGH THE LIST OF CLAUSES AND RETURNS THE FIRST UNSATISFIED CLAUSE IT ENCOUNTERS. IF ALL CLAUSES ARE SATISFIED, IT RANDOMLY SELECTS A CLAUSE AND RETURNS IT. THIS FUNCTION IS USED IN THE WALKSAT ALGORITHM TO IDENTIFY CLAUSES THAT NEED TO BE SATISFIED OR FLIPPED DURING THE SEARCH FOR A SATISFYING ASSIGNMENT.
    def __SELECT_CLAUSE__(ASSIGNMENT: dict, CLAUSES: list) -> list:
        """THE `__SELECT_CLAUSE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO SELECT AN UNSATISFIED CLAUSE FROM A LIST OF CLAUSES."""
        for CLAUSE in CLAUSES:  # FOR EACH CLAUSE IN CLAUSES
            # IF THE CLAUSE IS NOT SATISFIED
            if not __SATISFIED__(ASSIGNMENT, CLAUSE):
                return CLAUSE  # RETURN THE CLAUSE
        # RETURN A RANDOM CLAUSE IF ALL CLAUSES ARE SATISFIED
        return CLAUSES[np.random.randint(0, len(CLAUSES) - 1)]

    # THE `__SELECT_VARIABLE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO SELECT AN UNASSIGNED VARIABLE FROM A GIVEN CLAUSE.
    #     1. THE FUNCTION TAKES TWO PARAMETERS:
    #         - `ASSIGNMENT`: A DICTIONARY REPRESENTING THE CURRENT VARIABLE ASSIGNMENT.
    #         - `CLAUSE`: A LIST OF LITERALS (VARIABLES) REPRESENTING A CLAUSE.
    #     2. THE FUNCTION ITERATES THROUGH EACH VARIABLE IN THE `CLAUSE` LIST.
    #     3. FOR EACH VARIABLE, IT CHECKS IF THE VARIABLE IS ALREADY ASSIGNED A VALUE IN THE `ASSIGNMENT` DICTIONARY.
    #     4. IF THE VARIABLE IS NOT ASSIGNED A VALUE (I.E., IT IS UNASSIGNED), THE FUNCTION IMMEDIATELY RETURNS THE UNASSIGNED VARIABLE.
    #     5. IF ALL VARIABLES HAVE BEEN ITERATED THROUGH AND ALL OF THEM ARE ASSIGNED VALUES, THE FUNCTION RANDOMLY SELECTS A VARIABLE FROM THE `CLAUSE` LIST AND RETURNS IT.
    # IN SUMMARY, THE `__SELECT_VARIABLE__()` FUNCTION SCANS THROUGH THE LIST OF VARIABLES IN A CLAUSE AND RETURNS THE FIRST UNASSIGNED VARIABLE IT ENCOUNTERS. IF ALL VARIABLES IN THE CLAUSE ARE ASSIGNED VALUES, IT RANDOMLY SELECTS A VARIABLE FROM THE CLAUSE AND RETURNS IT. THIS FUNCTION IS USED IN THE WALKSAT ALGORITHM TO SELECT VARIABLES THAT NEED TO BE ASSIGNED VALUES DURING THE SEARCH FOR A SATISFYING ASSIGNMENT.
    def __SELECT_VARIABLE__(ASSIGNMENT: dict, CLAUSE: list) -> str:
        """THE `__SELECT_VARIABLE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO SELECT AN UNASSIGNED VARIABLE FROM A GIVEN CLAUSE."""
        for VARIABLE in CLAUSE:  # FOR EACH VARIABLE IN CLAUSE
            if VARIABLE not in ASSIGNMENT:  # IF THE VARIABLE IS NOT ASSIGNED A VALUE
                return VARIABLE  # RETURN THE VARIABLE
        # RETURN A RANDOM VARIABLE IF ALL VARIABLES ARE ASSIGNED VALUES
        return CLAUSE[np.random.randint(0, len(CLAUSE) - 1)]

    # THE `__SELECT_VALUE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO SELECT A VALUE FOR A GIVEN VARIABLE WITHIN A CLAUSE THAT SATISFIES THE CLAUSE WHEN ASSIGNED.
    #     1. THE FUNCTION TAKES FOUR PARAMETERS:
    #         - `ASSIGNMENT`: A DICTIONARY REPRESENTING THE CURRENT VARIABLE ASSIGNMENT.
    #         - `VARIABLE`: THE VARIABLE FOR WHICH A VALUE NEEDS TO BE SELECTED.
    #         - `CLAUSE`: A LIST OF LITERALS (VARIABLES) REPRESENTING A CLAUSE.
    #     2. THE FUNCTION ITERATES THROUGH EACH VALUE IN THE DOMAIN OF THE `VARIABLE` FROM `CSP_PROBLEM.DOMAINS[VARIABLE]`.
    #     3. FOR EACH VALUE, IT ASSIGNS THE VALUE TO THE `VARIABLE` IN THE `ASSIGNMENT` DICTIONARY.
    #     4. IT THEN CHECKS IF THE ASSIGNMENT SATISFIES THE `CLAUSE` BY CALLING A HELPER FUNCTION CALLED `__SATISFIED__()`.
    #     5. IF THE CLAUSE IS SATISFIED BY THE CURRENT ASSIGNMENT, MEANING THE ASSIGNMENT OF THE VALUE TO THE VARIABLE SATISFIES THE CLAUSE, THE FUNCTION RETURNS THE SELECTED VALUE.
    # 6. IF NONE OF THE VALUES IN THE VARIABLE'S DOMAIN SATISFY THE CLAUSE, THE FUNCTION RETURNS THE CURRENT VALUE ASSIGNED TO THE VARIABLE IN THE `ASSIGNMENT` DICTIONARY.
    # IN SUMMARY, THE `__SELECT_VALUE__()` FUNCTION ITERATES THROUGH THE VALUES IN THE DOMAIN OF A GIVEN VARIABLE AND SELECTS THE FIRST VALUE THAT SATISFIES THE CLAUSE WHEN ASSIGNED. IF NO VALUE SATISFIES THE CLAUSE, IT RETURNS THE CURRENT VALUE ASSIGNED TO THE VARIABLE. THIS FUNCTION IS USED IN THE WALKSAT ALGORITHM TO CHOOSE VALUES FOR VARIABLES THAT HELP IN SATISFYING UNSATISFIED CLAUSES DURING THE SEARCH FOR A SATISFYING ASSIGNMENT.
    def __SELECT_VALUE__(ASSIGNMENT: dict, VARIABLE: str, CLAUSE: list) -> str:
        """THE `__SELECT_VALUE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO SELECT A VALUE FOR A GIVEN VARIABLE WITHIN A CLAUSE THAT SATISFIES THE CLAUSE WHEN ASSIGNED."""
        for VALUE in CSP_PROBLEM.DOMAINS[VARIABLE]:  # FOR EACH VALUE IN THE DOMAIN OF VARIABLE
            ASSIGNMENT[VARIABLE] = VALUE  # ASSIGN THE VALUE TO VARIABLE
            # IF THE CLAUSE IS SATISFIED BY THE CURRENT ASSIGNMENT
            if __SATISFIED__(ASSIGNMENT, CLAUSE):
                return VALUE  # RETURN THE SELECTED VALUE
        # RETURN THE CURRENT VALUE ASSIGNED TO VARIABLE
        return ASSIGNMENT[VARIABLE]

    # THE `__SATISFIED__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO DETERMINE WHETHER A GIVEN CLAUSE IS SATISFIED BY THE CURRENT ASSIGNMENT OF VARIABLES.
    #     1. THE FUNCTION TAKES THREE PARAMETERS:
    #         - `ASSIGNMENT`: A DICTIONARY REPRESENTING THE CURRENT VARIABLE ASSIGNMENT.
    #         - `CLAUSE`: A LIST OF LITERALS (VARIABLES) REPRESENTING A CLAUSE.
    #     2. THE FUNCTION ITERATES THROUGH EACH VARIABLE IN THE `CLAUSE` LIST.
    #     3. FOR EACH VARIABLE, IT CHECKS IF THE VARIABLE IS NOT IN THE `ASSIGNMENT` DICTIONARY, INDICATING THAT THE VARIABLE IS NOT ASSIGNED A VALUE.
    #     4. IF A VARIABLE IS NOT IN THE `ASSIGNMENT` DICTIONARY, IT MEANS THAT THE CLAUSE IS NOT SATISFIED BECAUSE A VARIABLE IN THE CLAUSE IS UNASSIGNED. IN THIS CASE, THE FUNCTION IMMEDIATELY RETURNS `FALSE` TO INDICATE THAT THE CLAUSE IS NOT SATISFIED.
    #     5. IF ALL VARIABLES IN THE CLAUSE ARE ASSIGNED VALUES, THE FUNCTION CHECKS IF THE ASSIGNED VALUE FOR EACH VARIABLE IS VALID ACCORDING TO THE DOMAIN OF THE VARIABLE IN THE `CSP_PROBLEM`.
    #     6. IF ANY VARIABLE'S ASSIGNED VALUE IS NOT VALID ACCORDING TO ITS DOMAIN, THE FUNCTION RETURNS `FALSE` TO INDICATE THAT THE CLAUSE IS NOT SATISFIED.
    #     7. IF ALL VARIABLES IN THE CLAUSE HAVE VALID ASSIGNED VALUES, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE CLAUSE IS SATISFIED.
    # IN SUMMARY, THE `__SATISFIED__()` FUNCTION CHECKS WHETHER A GIVEN CLAUSE IS SATISFIED BY THE CURRENT VARIABLE ASSIGNMENT. IT CHECKS IF ALL VARIABLES IN THE CLAUSE ARE ASSIGNED VALUES AND IF THOSE ASSIGNED VALUES ARE VALID ACCORDING TO THE VARIABLE DOMAINS. THIS FUNCTION IS USED IN THE WALKSAT ALGORITHM TO DETERMINE THE SATISFACTION STATUS OF A CLAUSE DURING THE SEARCH FOR A SATISFYING ASSIGNMENT.
    def __SATISFIED__(ASSIGNMENT: dict, CLAUSE: list) -> bool:
        """THE `__SATISFIED__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO DETERMINE WHETHER A GIVEN CLAUSE IS SATISFIED BY THE CURRENT ASSIGNMENT OF VARIABLES."""
        for VARIABLE in CLAUSE:  # FOR EACH VARIABLE IN CLAUSE
            if VARIABLE not in ASSIGNMENT:  # IF THE VARIABLE IS NOT ASSIGNED A VALUE
                return False  # RETURN FALSE
            # IF THE VARIABLE IS ASSIGNED A VALUE AND THE VALUE IS VALID ACCORDING TO THE VARIABLE'S DOMAIN
            if VARIABLE in ASSIGNMENT and ASSIGNMENT[VARIABLE] in CSP_PROBLEM.DOMAINS[VARIABLE]:
                return True  # RETURN TRUE
        return False  # RETURN FALSE

    # THE `__COMPLETE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO CHECK WHETHER THE CURRENT VARIABLE ASSIGNMENT IS COMPLETE, I.E., IF ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES.
    #     1. THE FUNCTION TAKES TWO PARAMETERS:
    #         - `ASSIGNMENT`: A DICTIONARY REPRESENTING THE CURRENT VARIABLE ASSIGNMENT.
    #     2. THE FUNCTION ITERATES THROUGH EACH VARIABLE IN THE `CSP_PROBLEM.VARIABLES` LIST, WHICH CONTAINS ALL THE VARIABLES IN THE CSP PROBLEM.
    #     3. FOR EACH VARIABLE, IT CHECKS IF THE VARIABLE IS NOT IN THE `ASSIGNMENT` DICTIONARY, INDICATING THAT THE VARIABLE IS NOT ASSIGNED A VALUE.
    #     4. IF ANY VARIABLE IS FOUND TO BE UNASSIGNED, THE FUNCTION IMMEDIATELY RETURNS `FALSE` TO INDICATE THAT THE ASSIGNMENT IS NOT COMPLETE.
    #     5. IF ALL VARIABLES IN THE CSP PROBLEM ARE ASSIGNED VALUES, THE FUNCTION RETURNS `TRUE` TO INDICATE THAT THE ASSIGNMENT IS COMPLETE.
    # IN SUMMARY, THE `__COMPLETE__()` FUNCTION CHECKS WHETHER ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES IN THE CURRENT VARIABLE ASSIGNMENT. IT ITERATES THROUGH THE VARIABLES AND CHECKS IF EACH VARIABLE IS PRESENT IN THE ASSIGNMENT DICTIONARY. IF ANY VARIABLE IS FOUND TO BE UNASSIGNED, IT RETURNS `FALSE`. OTHERWISE, IT RETURNS `TRUE` TO INDICATE THAT THE ASSIGNMENT IS COMPLETE. THIS FUNCTION IS USED IN THE WALKSAT ALGORITHM TO DETERMINE IF A SATISFYING ASSIGNMENT HAS BEEN FOUND.
    def __COMPLETE__(ASSIGNMENT: dict) -> bool:
        """THE `__COMPLETE__()` FUNCTION IS A HELPER FUNCTION USED BY THE `WALK_SAT()` FUNCTION IN THE WALKSAT ALGORITHM. ITS PURPOSE IS TO CHECK WHETHER THE CURRENT VARIABLE ASSIGNMENT IS COMPLETE, I.E., IF ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES."""
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # FOR EACH VARIABLE IN THE CSP PROBLEM
            if VARIABLE not in ASSIGNMENT:  # IF THE VARIABLE IS NOT ASSIGNED A VALUE
                return False  # RETURN FALSE
        return True  # RETURN TRUE    
    
    ASSIGNMENT = {}  # INITIALIZE ASSIGNMENT DICTIONARY
    CLAUSES = []  # INITIALIZE CLAUSES LIST
    for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # FOR EACH CONSTRAINT IN CSP_PROBLEM.CONSTRAINTS
        CLAUSE = []  # INITIALIZE CLAUSE LIST
        for VARIABLE in CONSTRAINT:  # FOR EACH VARIABLE IN CONSTRAINT
            CLAUSE.append(VARIABLE)  # APPEND VARIABLE TO CLAUSE
        CLAUSES.append(CLAUSE)  # APPEND CLAUSE TO CLAUSES
    for VARIABLE in CSP_PROBLEM.VARIABLES:  # FOR EACH VARIABLE IN CSP_PROBLEM.VARIABLES
        # ASSIGN A RANDOM VALUE FROM THE VARIABLE'S DOMAIN TO THE VARIABLE IN THE ASSIGNMENT DICTIONARY
        ASSIGNMENT[VARIABLE] = CSP_PROBLEM.DOMAINS[VARIABLE][np.random.randint(
            0, len(CSP_PROBLEM.DOMAINS[VARIABLE]) - 1)]
    for _ in range(MAX_RESTARTS):  # FOR EACH RESTART
        for _ in range(MAX_FLIPS):  # FOR EACH FLIP
            if __COMPLETE__(ASSIGNMENT):  # IF THE ASSIGNMENT IS COMPLETE
                return ASSIGNMENT  # RETURN THE ASSIGNMENT
            # SELECT AN UNSATISFIED CLAUSE
            CLAUSE = __SELECT_CLAUSE__(ASSIGNMENT, CLAUSES)
            # SELECT AN UNASSIGNED VARIABLE FROM THE CLAUSE
            VARIABLE = __SELECT_VARIABLE__(ASSIGNMENT, CLAUSE)
            # SELECT A VALUE FROM THE VARIABLE'S DOMAIN THAT SATISFIES THE CLAUSE
            VALUE = __SELECT_VALUE__(ASSIGNMENT, VARIABLE, CLAUSE)
            # ASSIGN THE VALUE TO THE VARIABLE IN THE ASSIGNMENT DICTIONARY
            ASSIGNMENT[VARIABLE] = VALUE
    return False  # RETURN FALSE IF NO SATISFYING ASSIGNMENT IS FOUND