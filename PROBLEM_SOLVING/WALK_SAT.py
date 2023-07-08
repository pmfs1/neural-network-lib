import numpy as np
from .CSP import CSP

# THE `WALK_SAT` FUNCTION IS AN IMPLEMENTATION OF THE WALKSAT ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS).
#     1. THE FUNCTION TAKES THREE PARAMETERS: `CSP_PROBLEM`, WHICH IS AN INSTANCE OF THE CSP CLASS CONTAINING THE PROBLEM DEFINITION, `MAX_FLIPS`, WHICH SPECIFIES THE MAXIMUM NUMBER OF VARIABLE ASSIGNMENTS TO TRY BEFORE GIVING UP, AND `MAX_RESTARTS`, WHICH SPECIFIES THE MAXIMUM NUMBER OF TIMES TO RESTART THE ALGORITHM IF NO SOLUTION IS FOUND.
#     2. THE FUNCTION DEFINES SEVERAL HELPER FUNCTIONS THAT ARE USED INTERNALLY BY THE WALK_SAT ALGORITHM.
#         - THE `__SELECT_CLAUSE__` FUNCTION SELECTS AN UNSATISFIED CLAUSE FROM A LIST OF CLAUSES. IT ITERATES THROUGH THE LIST OF CLAUSES AND CHECKS IF EACH CLAUSE IS SATISFIED BY THE CURRENT VARIABLE ASSIGNMENT. IF IT FINDS AN UNSATISFIED CLAUSE, IT RETURNS THAT CLAUSE. IF ALL CLAUSES ARE SATISFIED, IT RANDOMLY SELECTS AND RETURNS A CLAUSE FROM THE LIST.
#         - THE `__SELECT_VARIABLE__` FUNCTION SELECTS AN UNASSIGNED VARIABLE FROM A GIVEN CLAUSE. IT ITERATES THROUGH THE VARIABLES IN THE CLAUSE AND CHECKS IF EACH VARIABLE HAS BEEN ASSIGNED A VALUE. IF IT FINDS AN UNASSIGNED VARIABLE, IT RETURNS THAT VARIABLE. IF ALL VARIABLES IN THE CLAUSE HAVE BEEN ASSIGNED VALUES, IT RANDOMLY SELECTS AND RETURNS A VARIABLE FROM THE CLAUSE.
#         - THE `__SELECT_VALUE__` FUNCTION SELECTS A VALUE FOR A GIVEN VARIABLE WITHIN A CLAUSE THAT SATISFIES THE CLAUSE WHEN ASSIGNED. IT ITERATES THROUGH THE POSSIBLE VALUES FOR THE VARIABLE AND ASSIGNS EACH VALUE TO THE VARIABLE IN THE ASSIGNMENT. IT THEN CHECKS IF THE CLAUSE IS SATISFIED WITH THE CURRENT ASSIGNMENT. IF IT FINDS A VALUE THAT SATISFIES THE CLAUSE, IT RETURNS THAT VALUE. IF NONE OF THE VALUES SATISFY THE CLAUSE, IT RETURNS THE CURRENT VALUE ASSIGNED TO THE VARIABLE.
#         - THE `__SATISFIED__` FUNCTION DETERMINES WHETHER A GIVEN CLAUSE IS SATISFIED BY THE CURRENT ASSIGNMENT OF VARIABLES. IT ITERATES THROUGH THE VARIABLES IN THE CLAUSE AND CHECKS IF EACH VARIABLE IS ASSIGNED A VALUE IN THE ASSIGNMENT. IF A VARIABLE IS NOT ASSIGNED A VALUE, OR IF THE ASSIGNED VALUE IS NOT IN THE DOMAIN OF THE VARIABLE, IT RETURNS FALSE. IF ALL VARIABLES IN THE CLAUSE ARE ASSIGNED VALUES AND SATISFY THE DOMAIN CONSTRAINTS, IT RETURNS TRUE.
#         - THE `__COMPLETE__` FUNCTION CHECKS WHETHER THE CURRENT VARIABLE ASSIGNMENT IS COMPLETE, I.E., IF ALL VARIABLES IN THE CSP PROBLEM HAVE BEEN ASSIGNED VALUES. IT ITERATES THROUGH THE VARIABLES IN THE CSP PROBLEM AND CHECKS IF EACH VARIABLE IS ASSIGNED A VALUE IN THE ASSIGNMENT. IF ANY VARIABLE IS NOT ASSIGNED A VALUE, IT RETURNS FALSE. IF ALL VARIABLES HAVE BEEN ASSIGNED VALUES, IT RETURNS TRUE.
#     3. THE MAIN BODY OF THE `WALK_SAT` FUNCTION STARTS BY INITIALIZING AN EMPTY ASSIGNMENT DICTIONARY AND AN EMPTY LIST OF CLAUSES.
#     4. IT THEN CONVERTS THE CONSTRAINTS OF THE CSP PROBLEM INTO A LIST OF CLAUSES. EACH CONSTRAINT IS CONVERTED INTO A CLAUSE BY EXTRACTING THE VARIABLES FROM THE CONSTRAINT.
#     5. NEXT, IT RANDOMLY ASSIGNS VALUES TO EACH VARIABLE IN THE CSP PROBLEM BY SELECTING A RANDOM VALUE FROM THE DOMAIN OF EACH VARIABLE.
#     6. THE ALGORITHM PERFORMS A SPECIFIED NUMBER OF RESTARTS (CONTROLLED BY `MAX_RESTARTS`). WITHIN EACH RESTART, IT PERFORMS A SPECIFIED NUMBER OF VARIABLE FLIPS (CONTROLLED BY `MAX_FLIPS`).
#     7. IN EACH FLIP, THE ALGORITHM CHECKS IF THE ASSIGNMENT IS COMPLETE. IF IT IS COMPLETE, IT MEANS A SOLUTION HAS BEEN FOUND, SO THE ASSIGNMENT IS RETURNED.
#     8. OTHERWISE, IT SELECTS AN UNSATISFIED CLAUSE USING THE `__SELECT_CLAUSE__` FUNCTION.
#     9. IT THEN SELECTS AN UNASSIGNED VARIABLE FROM THE SELECTED CLAUSE USING THE `__SELECT_VARIABLE__` FUNCTION.
#     10. IT SELECTS A VALUE FOR THE SELECTED VARIABLE WITHIN THE CLAUSE USING THE `__SELECT_VALUE__` FUNCTION.
#     11. THE SELECTED VALUE IS ASSIGNED TO THE SELECTED VARIABLE IN THE ASSIGNMENT.
#     12. IF NO SOLUTION IS FOUND AFTER ALL THE RESTARTS AND FLIPS, THE FUNCTION RETURNS FALSE TO INDICATE THAT NO SOLUTION COULD BE FOUND.
# OVERALL, THE WALK_SAT ALGORITHM REPEATEDLY TRIES TO SATISFY THE CONSTRAINTS OF A CSP PROBLEM BY RANDOMLY ASSIGNING VALUES TO VARIABLES AND FLIPPING THE VALUES BASED ON UNSATISFIED CLAUSES. IT EXPLORES DIFFERENT ASSIGNMENTS AND RESTARTS TO SEARCH FOR A VALID SOLUTION.
def WALK_SAT(CSP_PROBLEM: CSP, MAX_FLIPS: int, MAX_RESTARTS: int) -> dict | bool:
    """THE `WALK_SAT` FUNCTION IS AN IMPLEMENTATION OF THE WALKSAT ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS)."""
    
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