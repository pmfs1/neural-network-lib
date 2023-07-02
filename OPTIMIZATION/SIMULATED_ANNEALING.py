import math
import numpy as np
from .CSP import CSP

# THE `SIMULATED_ANNEALING` FUNCTION IS AN IMPLEMENTATION OF THE SIMULATED ANNEALING ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS).
#     1. THE FUNCTION TAKES A `CSP_PROBLEM` AS INPUT, WHICH IS AN INSTANCE OF THE `CSP` CLASS REPRESENTING THE CSP TO BE SOLVED.
#     2. IT INITIALIZES THE `CURRENT_STATE` DICTIONARY, WHICH REPRESENTS THE CURRENT ASSIGNMENT OF VALUES TO VARIABLES IN THE CSP. THE INITIAL ASSIGNMENT IS CHOSEN RANDOMLY FROM THE DOMAINS OF THE VARIABLES.
#     3. IT SETS THE INITIAL `TEMPERATURE` TO 100 AND THE `COOLING_RATE` TO 0.99. THE TEMPERATURE IS GRADUALLY REDUCED DURING THE SEARCH TO CONTROL THE EXPLORATION-EXPLOITATION TRADE-OFF.
#     4. THE ALGORITHM ENTERS A LOOP THAT CONTINUES UNTIL THE TEMPERATURE DROPS BELOW 1.
#     5. INSIDE THE LOOP, A NEW STATE `NEXT_STATE` IS GENERATED BY RANDOMLY SELECTING A VALUE FROM THE DOMAINS OF EACH VARIABLE IN THE CSP.
#     6. THE ALGORITHM EVALUATES THE COST OF THE CURRENT STATE AND THE NEXT STATE USING THE `__COST_FUNCTION__` HELPER FUNCTION. THE COST IS CALCULATED BY COUNTING THE NUMBER OF CONSTRAINTS THAT ARE NOT SATISFIED IN THE CURRENT STATE.
#     7. IF THE COST OF THE NEXT STATE IS LOWER THAN THE COST OF THE CURRENT STATE, INDICATING AN IMPROVEMENT, THE NEXT STATE BECOMES THE CURRENT STATE.
#     8. IF THE COST OF THE NEXT STATE IS HIGHER THAN THE COST OF THE CURRENT STATE, THE ALGORITHM CALCULATES A PROBABILITY OF ACCEPTING THE NEXT STATE BASED ON THE DIFFERENCE IN COST AND THE CURRENT TEMPERATURE. THE PROBABILITY IS DETERMINED USING THE BOLTZMANN DISTRIBUTION WITH THE `MATH.EXP` FUNCTION.
#     9. IF THE CALCULATED PROBABILITY IS GREATER THAN A RANDOM NUMBER BETWEEN 0 AND 1, THE NEXT STATE IS ACCEPTED AS THE CURRENT STATE, EVEN THOUGH IT IS WORSE THAN THE CURRENT STATE. THIS ALLOWS THE ALGORITHM TO ESCAPE LOCAL OPTIMA AND EXPLORE THE SEARCH SPACE.
#     10. AFTER UPDATING THE CURRENT STATE, THE TEMPERATURE IS REDUCED BY MULTIPLYING IT WITH THE COOLING RATE.
#     11. ONCE THE TEMPERATURE DROPS BELOW 1, THE LOOP TERMINATES, AND THE CURRENT STATE, WHICH REPRESENTS A SOLUTION OR AN APPROXIMATION TO THE SOLUTION, IS RETURNED.
# IN SUMMARY, THE `SIMULATED_ANNEALING` FUNCTION IMPLEMENTS THE SIMULATED ANNEALING ALGORITHM TO SOLVE CONSTRAINT SATISFACTION PROBLEMS (CSPS). IT STARTS WITH AN INITIAL RANDOM ASSIGNMENT OF VALUES TO VARIABLES AND GRADUALLY EXPLORES THE SEARCH SPACE BY GENERATING NEW ASSIGNMENTS AND EVALUATING THEIR COSTS. THE ALGORITHM ACCEPTS BETTER ASSIGNMENTS AND SOMETIMES ACCEPTS WORSE ASSIGNMENTS BASED ON A PROBABILITY CALCULATED USING THE TEMPERATURE. THE TEMPERATURE DECREASES OVER TIME TO BALANCE EXPLORATION AND EXPLOITATION. THE ALGORITHM CONTINUES UNTIL THE TEMPERATURE REACHES A MINIMUM VALUE, AND THEN IT RETURNS THE CURRENT ASSIGNMENT, WHICH REPRESENTS A SOLUTION OR AN APPROXIMATION TO THE SOLUTION. THE ALGORITHM USES TWO HELPER FUNCTIONS: `__COST_FUNCTION__` TO CALCULATE THE COST OF AN ASSIGNMENT BASED ON UNSATISFIED CONSTRAINTS, AND `__CONSTRAINT_SATISFIED__` TO CHECK IF A CONSTRAINT IS SATISFIED IN AN ASSIGNMENT.
def SIMULATED_ANNEALING(CSP_PROBLEM: CSP) -> dict:
    """THE `SIMULATED_ANNEALING` FUNCTION IS AN IMPLEMENTATION OF THE SIMULATED ANNEALING ALGORITHM FOR SOLVING CONSTRAINT SATISFACTION PROBLEMS (CSPS)."""

    # THE `__COST_FUNCTION__` IS A HELPER FUNCTION USED BY THE `SIMULATED_ANNEALING` ALGORITHM TO CALCULATE THE COST OF A GIVEN ASSIGNMENT (STATE) IN A CONSTRAINT SATISFACTION PROBLEM (CSP).
    #     1. IT TAKES ONE INPUT: `STATE`, WHICH IS A DICTIONARY REPRESENTING AN ASSIGNMENT OF VALUES TO VARIABLES IN THE CSP.
    #     2. IT INITIALIZES A `COST` VARIABLE TO 0, WHICH WILL BE USED TO COUNT THE NUMBER OF UNSATISFIED CONSTRAINTS IN THE STATE.
    #     3. IT ITERATES OVER EACH CONSTRAINT IN `CSP_PROBLEM.CONSTRAINTS`.
    #     4. FOR EACH CONSTRAINT, IT CALLS THE `__CONSTRAINT_SATISFIED__` HELPER FUNCTION TO CHECK IF THE CONSTRAINT IS SATISFIED IN THE GIVEN `STATE`. IF THE CONSTRAINT IS NOT SATISFIED, THE `COST` IS INCREMENTED BY 1.
    #     5. ONCE ALL CONSTRAINTS HAVE BEEN CHECKED, THE `COST` VALUE REPRESENTS THE NUMBER OF UNSATISFIED CONSTRAINTS IN THE GIVEN STATE.
    #     6. FINALLY, THE `COST` VALUE IS RETURNED AS THE RESULT OF THE `__COST_FUNCTION__`.
    # THE PURPOSE OF THE `__COST_FUNCTION__` IS TO PROVIDE A MEASURE OF HOW "GOOD" OR "BAD" A GIVEN ASSIGNMENT IS IN TERMS OF CONSTRAINT SATISFACTION. THE LOWER THE COST VALUE, THE MORE CONSTRAINTS ARE SATISFIED, AND THUS THE BETTER THE ASSIGNMENT IS CONSIDERED. THE `SIMULATED_ANNEALING` ALGORITHM USES THIS COST FUNCTION TO GUIDE ITS SEARCH FOR BETTER ASSIGNMENTS, AIMING TO MINIMIZE THE COST AND FIND A SATISFACTORY OR OPTIMAL SOLUTION TO THE CSP.
    def __COST_FUNCTION__(STATE: dict) -> int:
        """THE `__COST_FUNCTION__` IS A HELPER FUNCTION USED BY THE `SIMULATED_ANNEALING` ALGORITHM TO CALCULATE THE COST OF A GIVEN ASSIGNMENT (STATE) IN A CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        COST = 0  # INITIALIZE THE COST
        for CONSTRAINT in CSP_PROBLEM.CONSTRAINTS:  # ITERATE OVER EACH CONSTRAINT
            # IF THE CONSTRAINT IS NOT SATISFIED
            if not __CONSTRAINT_SATISFIED__(CONSTRAINT, STATE):
                COST += 1  # INCREMENT THE COST
        return COST  # RETURN THE COST

    # THE `__CONSTRAINT_SATISFIED__` FUNCTION IS A HELPER FUNCTION USED BY THE `__COST_FUNCTION__` TO DETERMINE WHETHER A SPECIFIC CONSTRAINT IS SATISFIED IN A GIVEN ASSIGNMENT (STATE) OF A CONSTRAINT SATISFACTION PROBLEM (CSP).
    #     1. IT TAKES TWO INPUTS: `CONSTRAINT`, WHICH IS A TUPLE REPRESENTING THE VARIABLES INVOLVED IN THE CONSTRAINT, AND `STATE`, WHICH IS A DICTIONARY REPRESENTING THE CURRENT ASSIGNMENT OF VALUES TO VARIABLES IN THE CSP.
    #     2. IT ACCESSES THE VALUES ASSIGNED TO THE VARIABLES INVOLVED IN THE CONSTRAINT BY USING THEIR KEYS IN THE `STATE` DICTIONARY. THE FIRST VARIABLE IN THE CONSTRAINT IS ACCESSED AS `STATE[CONSTRAINT[0]]` AND THE SECOND VARIABLE AS `STATE[CONSTRAINT[1]]`.
    #     3. IT CHECKS WHETHER THE VALUES ASSIGNED TO THE TWO VARIABLES ARE DIFFERENT. IF THE VALUES ARE DIFFERENT, IT MEANS THE CONSTRAINT IS SATISFIED, AND THE FUNCTION RETURNS `TRUE`. OTHERWISE, IF THE VALUES ARE THE SAME, IT MEANS THE CONSTRAINT IS NOT SATISFIED, AND THE FUNCTION RETURNS `FALSE`.
    # THE PURPOSE OF THE `__CONSTRAINT_SATISFIED__` FUNCTION IS TO PROVIDE A SIMPLE CHECK TO DETERMINE WHETHER A SPECIFIC CONSTRAINT IS SATISFIED OR NOT IN A GIVEN ASSIGNMENT. IT IS USED BY THE `__COST_FUNCTION__` TO EVALUATE WHETHER EACH CONSTRAINT IN THE CSP IS SATISFIED OR NOT, CONTRIBUTING TO THE OVERALL COST CALCULATION. BY CHECKING CONSTRAINT SATISFACTION, THE ALGORITHM CAN IDENTIFY UNSATISFIED CONSTRAINTS AND INCREMENT THE COST ACCORDINGLY, ALLOWING IT TO ASSESS THE QUALITY OF DIFFERENT ASSIGNMENTS AND GUIDE THE SEARCH TOWARDS MORE SATISFACTORY SOLUTIONS.
    def __CONSTRAINT_SATISFIED__(CONSTRAINT: tuple, STATE: dict) -> bool:
        """THE `__CONSTRAINT_SATISFIED__` FUNCTION IS A HELPER FUNCTION USED BY THE `__COST_FUNCTION__` TO DETERMINE WHETHER A SPECIFIC CONSTRAINT IS SATISFIED IN A GIVEN ASSIGNMENT (STATE) OF A CONSTRAINT SATISFACTION PROBLEM (CSP)."""
        return STATE[CONSTRAINT[0]] != STATE[CONSTRAINT[1]]  # RETURN WHETHER THE CONSTRAINT IS SATISFIED OR NOT

    CURRENT_STATE = {}  # INITIALIZE THE CURRENT STATE
    for VARIABLE in CSP_PROBLEM.VARIABLES:  # RANDOMLY ASSIGN VALUES TO VARIABLES
        CURRENT_STATE[VARIABLE] = np.random.choice(
            CSP_PROBLEM.DOMAINS[VARIABLE])  # RANDOMLY ASSIGN VALUES TO VARIABLES
    TEMPERATURE = 100  # INITIALIZE THE TEMPERATURE
    COOLING_RATE = 0.99  # INITIALIZE THE COOLING RATE
    while TEMPERATURE > 1:  # LOOP UNTIL THE TEMPERATURE DROPS BELOW 1
        NEXT_STATE = {}  # INITIALIZE THE NEXT STATE
        for VARIABLE in CSP_PROBLEM.VARIABLES:  # RANDOMLY ASSIGN VALUES TO VARIABLES
            # RANDOMLY ASSIGN VALUES TO VARIABLES
            NEXT_STATE[VARIABLE] = np.random.choice(
                CSP_PROBLEM.DOMAINS[VARIABLE])
        # IF THE COST OF THE NEXT STATE IS LOWER THAN THE COST OF THE CURRENT STATE
        if __COST_FUNCTION__(NEXT_STATE) < __COST_FUNCTION__(CURRENT_STATE):
            CURRENT_STATE = NEXT_STATE  # THE NEXT STATE BECOMES THE CURRENT STATE
        else:  # IF THE COST OF THE NEXT STATE IS HIGHER THAN THE COST OF THE CURRENT STATE
            PROBABILITY = math.exp(- (__COST_FUNCTION__(NEXT_STATE) - __COST_FUNCTION__(
                CURRENT_STATE)) / TEMPERATURE)  # CALCULATE THE PROBABILITY OF ACCEPTING THE NEXT STATE
            # IF THE CALCULATED PROBABILITY IS GREATER THAN A RANDOM NUMBER BETWEEN 0 AND 1
            if PROBABILITY > np.random.uniform(0, 1):
                CURRENT_STATE = NEXT_STATE  # THE NEXT STATE BECOMES THE CURRENT STATE
        TEMPERATURE *= COOLING_RATE  # REDUCE THE TEMPERATURE
    return CURRENT_STATE  # RETURN THE CURRENT STATE
