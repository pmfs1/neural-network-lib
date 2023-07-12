import numpy as np

# THE `AFFINITY_PROPAGATION` FUNCTION IS AN IMPLEMENTATION OF THE AFFINITY PROPAGATION CLUSTERING ALGORITHM.
#     1. **INPUT PARAMETERS**:
#         - `X`: THE INPUT DATA MATRIX OF SHAPE `(N_SAMPLES, N_FEATURES)` WHERE `N_SAMPLES` REPRESENTS THE NUMBER OF SAMPLES AND `N_FEATURES` REPRESENTS THE NUMBER OF FEATURES IN THE DATA.
#         - `MAX_ITERATIONS` (OPTIONAL): THE MAXIMUM NUMBER OF ITERATIONS FOR THE ALGORITHM. THE DEFAULT VALUE IS SET TO 200.
#         - `CONVERGENCE_ITERATIONS` (OPTIONAL): THE NUMBER OF ITERATIONS TO CHECK FOR CONVERGENCE. IF THE ALGORITHM DOESN'T CONVERGE WITHIN THESE ITERATIONS, IT WILL STOP. THE DEFAULT VALUE IS SET TO 15.
#         - `DAMPING` (OPTIONAL): THE DAMPING FACTOR USED TO UPDATE THE AVAILABILITIES AND RESPONSIBILITIES. IT CONTROLS THE AMOUNT OF INFLUENCE EACH ITERATION HAS ON THE PREVIOUS ITERATION'S VALUES. THE DEFAULT VALUE IS SET TO 0.5.
#     2. **SIMILARITY MATRIX CALCULATION**: THE FIRST STEP IS TO CALCULATE THE SIMILARITY MATRIX `S` BETWEEN THE SAMPLES IN THE INPUT DATA `X`. THE SIMILARITY BETWEEN TWO SAMPLES IS COMPUTED AS THE NEGATIVE EUCLIDEAN DISTANCE BETWEEN THEIR FEATURE VECTORS. THE SIMILARITY MATRIX `S` IS INITIALIZED WITH NEGATIVE INFINITY VALUES.
#     3. **RESPONSIBILITY AND AVAILABILITY UPDATES**: THE ALGORITHM ITERATIVELY UPDATES THE RESPONSIBILITIES `R` AND AVAILABILITIES `A` MATRICES. THESE MATRICES REPRESENT THE EVIDENCE OF EACH SAMPLE BEING AN EXEMPLAR (CLUSTER CENTRE) OR BEING INFLUENCED BY OTHER EXEMPLARS.
#         A. **RESPONSIBILITY UPDATE**:
#             - THE RESPONSIBILITY MATRIX `R` IS UPDATED BASED ON THE PREVIOUS AVAILABILITIES `A` AND THE SIMILARITY MATRIX `S`.
#             - FOR EACH SAMPLE `I`, THE MAXIMUM VALUE OF THE SUM OF THE AVAILABILITY AND SIMILARITY (A + S) OVER ALL OTHER SAMPLES IS DETERMINED. THE INDEX OF THE MAXIMUM VALUE REPRESENTS THE SAMPLE THAT HAS THE HIGHEST INFLUENCE ON SAMPLE `I`.
#             - THE RESPONSIBILITY OF SAMPLE `I` TOWARDS ITSELF IS SET TO THE SIMILARITY VALUE WITH THE SECOND HIGHEST INFLUENCE SUBTRACTED FROM THE SIMILARITY VALUE WITH THE HIGHEST INFLUENCE.
#             - THE RESPONSIBILITY OF SAMPLE `I` TOWARDS OTHER SAMPLES IS SET TO THE SIMILARITY VALUE WITH THE HIGHEST INFLUENCE SUBTRACTED FROM THE SIMILARITY VALUE WITH THE SECOND HIGHEST INFLUENCE.
#         B. **AVAILABILITY UPDATE**:
#             - THE AVAILABILITY MATRIX `A` IS UPDATED BASED ON THE UPDATED RESPONSIBILITIES `R`.
#             - NON-NEGATIVE RESPONSIBILITIES ARE SELECTED AND STORED IN `R_P`.
#             - THE AVAILABILITY OF SAMPLE `I` IS SET TO THE SUM OF POSITIVE RESPONSIBILITIES FOR SAMPLE `I`.
#             - A DIAGONAL MATRIX `D_A` IS CREATED USING THE AVAILABILITY VALUES ON THE DIAGONAL.
#             - THE AVAILABILITY OF SAMPLE `I` IS SET TO THE MINIMUM VALUE BETWEEN 0 AND THE SUM OF POSITIVE RESPONSIBILITIES FOR SAMPLE `I`, EXCEPT THE DIAGONAL ELEMENT IS SET TO THE CORRESPONDING VALUE FROM `D_A`.
#     4. **CONVERGENCE CHECK**: THE ALGORITHM CHECKS FOR CONVERGENCE BY COMPARING THE UPDATED RESPONSIBILITIES `R` AND AVAILABILITIES `A` WITH THE PREVIOUS ITERATIONS' VALUES. CONVERGENCE IS CONSIDERED WHEN BOTH MATRICES ARE CLOSE TO THEIR PREVIOUS VALUES, WITHIN A SPECIFIED TOLERANCE.
#     5. **DAMPING**: TO PREVENT OSCILLATIONS AND STABILIZE THE CONVERGENCE, THE UPDATED RESPONSIBILITIES `R` AND AVAILABILITIES `A` ARE DAMPED BY A FACTOR OF `DAMPING`. THE DAMPING FACTOR CONTROLS THE INFLUENCE OF THE NEW ITERATION'S VALUES RELATIVE TO THE PREVIOUS ITERATION'S VALUES.
#     6. **CLUSTER CENTER AND LABEL ASSIGNMENT**: ONCE THE ALGORITHM CONVERGES OR REACHES THE MAXIMUM NUMBER OF ITERATIONS, THE SAMPLES THAT HAVE POSITIVE AVAILABILITY AND RESPONSIBILITY (EXEMPLARS) ARE IDENTIFIED. THESE EXEMPLAR SAMPLES ARE CONSIDERED AS CLUSTER CENTRES. THE LABELS FOR ALL SAMPLES ARE ASSIGNED BASED ON THE INDEX OF THE CLUSTER CENTRE WITH THE HIGHEST SIMILARITY TO EACH SAMPLE. THE SAMPLES THAT ARE CLUSTER CENTRES ARE ASSIGNED LABELS FROM 0 TO THE NUMBER OF CLUSTER CENTRES MINUS ONE.
#     7. **OUTPUT**: THE FUNCTION RETURNS TWO ARRAYS:
#         - `CLUSTER_CENTER_IDXS`: THE INDICES OF THE SAMPLES THAT ARE IDENTIFIED AS CLUSTER CENTRES.
#         - `LABELS`: THE CLUSTER LABELS ASSIGNED TO EACH SAMPLE.
# OVERALL, THE AFFINITY PROPAGATION ALGORITHM AIMS TO FIND EXEMPLARS IN THE DATA AND ASSIGN EACH SAMPLE TO ONE OF THESE EXEMPLARS, FORMING CLUSTERS. IT USES A MESSAGE-PASSING APPROACH TO ITERATIVELY UPDATE RESPONSIBILITIES AND AVAILABILITIES UNTIL CONVERGENCE OR THE MAXIMUM NUMBER OF ITERATIONS IS REACHED.


def AFFINITY_PROPAGATION(X, MAX_ITERATIONS=200, CONVERGENCE_ITERATIONS=15, DAMPING=0.5):
    """THE `AFFINITY_PROPAGATION` FUNCTION IMPLEMENTS THE AFFINITY PROPAGATION CLUSTERING ALGORITHM.
        1. SIMILARITY CALCULATION: COMPUTE THE SIMILARITY MATRIX BETWEEN SAMPLES BASED ON THEIR FEATURE VECTORS.
        2. RESPONSIBILITY AND AVAILABILITY UPDATES:
            - ITERATIVELY UPDATE THE RESPONSIBILITIES AND AVAILABILITIES.
            - RESPONSIBILITIES MEASURE THE EVIDENCE OF A SAMPLE BEING AN EXEMPLAR OR INFLUENCING OTHER SAMPLES.
            - AVAILABILITIES REPRESENT THE ACCUMULATED EVIDENCE FOR EACH SAMPLE BEING CHOSEN AS AN EXEMPLAR.
        3. CONVERGENCE CHECK:
            - CHECK IF THE RESPONSIBILITIES AND AVAILABILITIES HAVE CONVERGED.
            - CONVERGENCE IS DETERMINED BY COMPARING THEM TO THE PREVIOUS ITERATION'S VALUES.
        4. DAMPING: DAMPEN THE UPDATED RESPONSIBILITIES AND AVAILABILITIES TO STABILIZE CONVERGENCE AND PREVENT OSCILLATIONS.
        5. CLUSTER CENTER AND LABEL ASSIGNMENT:
            - IDENTIFY SAMPLES AS CLUSTER CENTRES BASED ON POSITIVE AVAILABILITY AND RESPONSIBILITY.
            - ASSIGN LABELS TO EACH SAMPLE BASED ON THE CLUSTER CENTRE WITH THE HIGHEST SIMILARITY.
        6. OUTPUT: RETURN THE INDICES OF THE CLUSTER CENTRES AND THE ASSIGNED LABELS FOR EACH SAMPLE.
    THE AFFINITY PROPAGATION ALGORITHM FINDS EXEMPLARS IN THE DATA AND ASSIGNS EACH SAMPLE TO ONE OF THESE EXEMPLARS, FORMING CLUSTERS. IT ITERATIVELY UPDATES RESPONSIBILITIES AND AVAILABILITIES, CHECKS FOR CONVERGENCE, AND APPLIES DAMPING TO ENSURE STABILITY. FINALLY, IT IDENTIFIES CLUSTER CENTRES AND ASSIGNS LABELS TO SAMPLES BASED ON SIMILARITY."""

    # THE `__CALCULATE_SIMILARITY_MATRIX__` FUNCTION IS RESPONSIBLE FOR COMPUTING THE SIMILARITY MATRIX `S` BASED ON THE INPUT DATA `X`.
    #     1. **INPUT PARAMETER**:
    #         - `X`: THE INPUT DATA MATRIX OF SHAPE `(N_SAMPLES, N_FEATURES)` WHERE `N_SAMPLES` REPRESENTS THE NUMBER OF SAMPLES AND `N_FEATURES` REPRESENTS THE NUMBER OF FEATURES IN THE DATA.
    #     2. **SIMILARITY MATRIX INITIALIZATION**:
    #         - CREATE AN EMPTY SIMILARITY MATRIX `S` WITH DIMENSIONS `(N_SAMPLES, N_SAMPLES)`.
    #         - INITIALIZE ALL ELEMENTS OF `S` WITH NEGATIVE INFINITY (`-NP.INF`).
    #     3. **SIMILARITY CALCULATION**:
    #         - ITERATE OVER EACH PAIR OF SAMPLES IN THE INPUT DATA `X`.
    #         - FOR EACH PAIR OF SAMPLES `(I, J)`, CALCULATE THE SIMILARITY AS THE NEGATIVE EUCLIDEAN DISTANCE BETWEEN THEIR FEATURE VECTORS.
    #         - THE NEGATIVE EUCLIDEAN DISTANCE IS USED SO THAT LARGER DISTANCES CORRESPOND TO SMALLER SIMILARITY VALUES.
    #         - ASSIGN THE CALCULATED SIMILARITY VALUE TO THE CORRESPONDING ELEMENT IN THE SIMILARITY MATRIX `S[I, J]`.
    #     4. **OUTPUT**: RETURN THE SIMILARITY MATRIX `S` WHICH REPRESENTS THE PAIRWISE SIMILARITY BETWEEN SAMPLES IN THE INPUT DATA.
    # THE `__CALCULATE_SIMILARITY_MATRIX__` FUNCTION COMPUTES THE SIMILARITY BETWEEN PAIRS OF SAMPLES IN THE INPUT DATA `X` AND RETURNS A SIMILARITY MATRIX `S`. BY USING THE NEGATIVE EUCLIDEAN DISTANCE AS A MEASURE OF SIMILARITY, LARGER DISTANCES CORRESPOND TO SMALLER SIMILARITY VALUES. THIS SIMILARITY MATRIX SERVES AS THE BASIS FOR SUBSEQUENT STEPS IN THE AFFINITY PROPAGATION ALGORITHM, SUCH AS UPDATING RESPONSIBILITIES AND DETERMINING CLUSTER CENTRES.
    def __CALCULATE_SIMILARITY_MATRIX__(X):
        """THE `__CALCULATE_SIMILARITY_MATRIX__` FUNCTION CALCULATES THE SIMILARITY MATRIX `S` BASED ON THE INPUT DATA `X`.
            1. SIMILARITY MATRIX INITIALIZATION:
                - INITIALIZE AN EMPTY SIMILARITY MATRIX `S` WITH DIMENSIONS `(N_SAMPLES, N_SAMPLES)`.
                - SET ALL ELEMENTS OF `S` TO NEGATIVE INFINITY.
            2. SIMILARITY CALCULATION:
                - ITERATE OVER PAIRS OF SAMPLES IN THE INPUT DATA.
                - CALCULATE THE SIMILARITY BETWEEN EACH PAIR OF SAMPLES AS THE NEGATIVE EUCLIDEAN DISTANCE.
                - ASSIGN THE CALCULATED SIMILARITY VALUE TO THE CORRESPONDING ELEMENT IN THE SIMILARITY MATRIX `S`.
            3. OUTPUT: RETURN THE SIMILARITY MATRIX `S` THAT REPRESENTS THE PAIRWISE SIMILARITY BETWEEN SAMPLES.
        THE `__CALCULATE_SIMILARITY_MATRIX__` FUNCTION COMPUTES THE SIMILARITY MATRIX `S` BY CALCULATING THE NEGATIVE EUCLIDEAN DISTANCE BETWEEN PAIRS OF SAMPLES. THIS MATRIX CAPTURES THE SIMILARITY RELATIONSHIPS BETWEEN SAMPLES AND IS USED AS A FOUNDATION FOR SUBSEQUENT STEPS IN THE AFFINITY PROPAGATION ALGORITHM."""
        N_SAMPLES = X.shape[0]  # NUMBER OF SAMPLES
        # INITIALIZE SIMILARITY MATRIX
        S = -np.inf * np.ones((N_SAMPLES, N_SAMPLES))
        for i in range(N_SAMPLES):  # ITERATE OVER SAMPLES (I)
            for j in range(N_SAMPLES):  # ITERATE OVER SAMPLES (J)
                # CALCULATE SIMILARITY BETWEEN SAMPLES
                S[i, j] = -np.linalg.norm(X[i] - X[j])
        return S  # RETURN SIMILARITY MATRIX

    # THE `__UPDATE_RESPONSIBILITIES__` FUNCTION IS RESPONSIBLE FOR UPDATING THE RESPONSIBILITIES MATRIX `R` IN THE AFFINITY PROPAGATION ALGORITHM.
    #     1. **INPUT PARAMETERS**:
    #         - `A`: THE AVAILABILITY MATRIX OF SHAPE `(N_SAMPLES, N_SAMPLES)`.
    #         - `S`: THE SIMILARITY MATRIX OF SHAPE `(N_SAMPLES, N_SAMPLES)`.
    #     2. **RESPONSIBILITY CALCULATION**:
    #         - CALCULATE THE AUGMENTED MATRIX `AS` BY ELEMENT-WISE ADDITION OF `A` AND `S`.
    #         - FIND THE MAXIMUM VALUES AND THEIR CORRESPONDING INDICES ALONG EACH ROW OF `AS`. THIS DETERMINES THE SAMPLE INDICES THAT HAVE THE HIGHEST INFLUENCE ON EACH SAMPLE.
    #         - STORE THE MAXIMUM VALUES IN THE ARRAY `MAX_VALUES` AND THEIR INDICES IN THE ARRAY `MAX_IDXS`.
    #         - REPLACE THE MAXIMUM VALUES IN `AS` WITH NEGATIVE INFINITY (`-NP.INF`) TO EXCLUDE THEM FROM THE SECOND MAXIMUM CALCULATION.
    #         - CALCULATE THE SECOND MAXIMUM VALUES ALONG EACH ROW OF `AS` AND STORE THEM IN THE ARRAY `SECOND_MAX_VALUES`.
    #         - UPDATE THE RESPONSIBILITIES MATRIX `R` BASED ON THE FOLLOWING RULES:
    #             - FOR EACH SAMPLE `I`:
    #                 - SET THE RESPONSIBILITY OF SAMPLE `I` TOWARDS ITSELF (`R[I, I]`) AS THE SIMILARITY VALUE WITH THE SECOND HIGHEST INFLUENCE SUBTRACTED FROM THE SIMILARITY VALUE WITH THE HIGHEST INFLUENCE.
    #                 - SET THE RESPONSIBILITY OF SAMPLE `I` TOWARDS OTHER SAMPLES (`R[I, J]` WHERE `J != I`) AS THE SIMILARITY VALUE WITH THE HIGHEST INFLUENCE SUBTRACTED FROM THE SIMILARITY VALUE WITH THE SECOND HIGHEST INFLUENCE.
    #     3. **OUTPUT**: RETURN THE UPDATED RESPONSIBILITIES MATRIX `R`.
    # THE `__UPDATE_RESPONSIBILITIES__` FUNCTION CALCULATES THE UPDATED RESPONSIBILITIES MATRIX `R` BASED ON THE AVAILABILITY MATRIX `A` AND THE SIMILARITY MATRIX `S`. IT IDENTIFIES THE SAMPLES THAT HAVE THE HIGHEST INFLUENCE ON EACH SAMPLE AND COMPUTES THE CORRESPONDING RESPONSIBILITIES. BY COMPARING THE SIMILARITY VALUES, IT DETERMINES THE RELATIONSHIPS BETWEEN SAMPLES AND ADJUSTS THE RESPONSIBILITIES ACCORDINGLY. THE UPDATED RESPONSIBILITIES MATRIX IS A CRUCIAL COMPONENT IN THE AFFINITY PROPAGATION ALGORITHM FOR DETERMINING CLUSTER ASSIGNMENTS.
    def __UPDATE_RESPONSIBILITIES__(A, S):
        """THE `__UPDATE_RESPONSIBILITIES__` FUNCTION IS RESPONSIBLE FOR UPDATING THE RESPONSIBILITIES MATRIX `R` IN THE AFFINITY PROPAGATION ALGORITHM.
            1. RESPONSIBILITY CALCULATION:
                - CREATE A COPY OF THE OLD RESPONSIBILITIES MATRIX `R`.
                - AUGMENT THE AVAILABILITY MATRIX `A` WITH THE SIMILARITY MATRIX `S` TO CREATE THE MATRIX `AS`.
                - FIND THE SAMPLES THAT HAVE THE HIGHEST INFLUENCE ON EACH SAMPLE BY LOCATING THE MAXIMUM VALUES IN EACH ROW OF `AS`.
                - DETERMINE THE SECOND HIGHEST VALUES IN EACH ROW OF `AS`.
                - UPDATE THE RESPONSIBILITIES MATRIX `R` BASED ON THE MAXIMUM AND SECOND MAXIMUM VALUES:
                    - FOR EACH SAMPLE, CALCULATE THE RESPONSIBILITY TOWARDS ITSELF AS THE DIFFERENCE BETWEEN THE MAXIMUM AND SECOND MAXIMUM VALUES.
                    - CALCULATE THE RESPONSIBILITY TOWARDS OTHER SAMPLES AS THE DIFFERENCE BETWEEN THE MAXIMUM AND SECOND MAXIMUM VALUES.
            2. OUTPUT: RETURN THE UPDATED RESPONSIBILITIES MATRIX `R`.
        THE `__UPDATE_RESPONSIBILITIES__` FUNCTION CALCULATES THE UPDATED RESPONSIBILITIES MATRIX `R` BY COMPARING THE INFLUENCE OF DIFFERENT SAMPLES ON EACH OTHER. IT IDENTIFIES THE SAMPLES THAT HAVE THE HIGHEST INFLUENCE AND DETERMINES THE RESPONSIBILITIES BASED ON THE SIMILARITY VALUES. THIS PROCESS HELPS REFINE THE CLUSTER ASSIGNMENTS IN THE AFFINITY PROPAGATION ALGORITHM."""
        AS = A + S  # AUGMENT AVAILABILITY MATRIX WITH SIMILARITY MATRIX
        MAX_IDXS = np.argmax(AS, axis=1)  # FIND SAMPLES WITH HIGHEST INFLUENCE
        MAX_VALUES = AS[np.arange(N_SAMPLES), MAX_IDXS]  # STORE MAXIMUM VALUES
        AS[np.arange(N_SAMPLES), MAX_IDXS] = - \
            np.inf  # SET MAXIMUM VALUES TO -INF
        # FIND SAMPLES WITH SECOND HIGHEST INFLUENCE
        SECOND_MAX_VALUES = np.amax(AS, axis=1)
        # CALCULATE RESPONSIBILITIES
        R = S - np.repeat(MAX_VALUES[:, np.newaxis], N_SAMPLES, axis=1)
        R[np.arange(N_SAMPLES), MAX_IDXS] = S[np.arange(
            N_SAMPLES), MAX_IDXS] - SECOND_MAX_VALUES  # CALCULATE SELF-RESPONSIBILITIES
        return R  # RETURN UPDATED RESPONSIBILITIES MATRIX

    # THE `__UPDATE_AVAILABILITIES__` FUNCTION IS RESPONSIBLE FOR UPDATING THE AVAILABILITIES MATRIX `A` IN THE AFFINITY PROPAGATION ALGORITHM.
    #     1. **INPUT PARAMETER**:
    #         - `R`: THE RESPONSIBILITIES MATRIX OF SHAPE `(N_SAMPLES, N_SAMPLES)`.
    #     2. **AVAILABILITY CALCULATION**:
    #         - CALCULATE THE NON-NEGATIVE RESPONSIBILITIES MATRIX `R_P` BY TAKING THE MAXIMUM BETWEEN EACH ELEMENT OF `R` AND 0.
    #         - SET THE DIAGONAL ELEMENTS OF `R_P` TO BE THE CORRESPONDING DIAGONAL ELEMENTS OF `R` TO RETAIN THE ORIGINAL SELF-RESPONSIBILITIES.
    #         - COMPUTE THE AVAILABILITY MATRIX `A` BY SUMMING UP THE POSITIVE RESPONSIBILITIES FOR EACH SAMPLE ACROSS ALL SAMPLES.
    #         - CREATE A DIAGONAL MATRIX `D_A` USING THE AVAILABILITY VALUES ON THE DIAGONAL.
    #         - UPDATE THE AVAILABILITY MATRIX `A` BASED ON THE FOLLOWING RULES:
    #             - FOR EACH SAMPLE `I`:
    #               - SET THE AVAILABILITY OF SAMPLE `I` TO BE THE MINIMUM VALUE BETWEEN 0 AND THE SUM OF POSITIVE RESPONSIBILITIES FOR SAMPLE `I`.
    #               - EXCEPT FOR THE DIAGONAL ELEMENT, SET THE AVAILABILITY OF SAMPLE `I` TO BE THE CORRESPONDING VALUE FROM `D_A`, ENSURING THAT THE ORIGINAL SELF-AVAILABILITIES ARE PRESERVED.
    #     3. **OUTPUT**: RETURN THE UPDATED AVAILABILITY MATRIX `A`.
    # THE `__UPDATE_AVAILABILITIES__` FUNCTION CALCULATES THE UPDATED AVAILABILITIES MATRIX `A` BASED ON THE RESPONSIBILITIES MATRIX `R`. IT ENSURES THAT THE SELF-AVAILABILITIES ARE PRESERVED AND COMPUTES THE AVAILABILITY VALUES BY SUMMING UP POSITIVE RESPONSIBILITIES FOR EACH SAMPLE. BY ADJUSTING THE AVAILABILITIES, THE FUNCTION INFLUENCES THE EXEMPLAR SELECTION PROCESS IN THE AFFINITY PROPAGATION ALGORITHM, LEADING TO REFINED CLUSTER ASSIGNMENTS.
    def __UPDATE_AVAILABILITIES__(R):
        """THE `__UPDATE_AVAILABILITIES__` FUNCTION IS RESPONSIBLE FOR UPDATING THE AVAILABILITIES MATRIX `A` IN THE AFFINITY PROPAGATION ALGORITHM.
            1. AVAILABILITY CALCULATION:
                - CREATE A COPY OF THE OLD AVAILABILITIES MATRIX `A`.
                - CALCULATE THE NON-NEGATIVE RESPONSIBILITIES MATRIX `R_P` BY TAKING THE MAXIMUM BETWEEN EACH ELEMENT OF `R` AND 0.
                - PRESERVE THE ORIGINAL SELF-RESPONSIBILITIES BY SETTING THE DIAGONAL ELEMENTS OF `R_P` TO BE THE CORRESPONDING DIAGONAL ELEMENTS OF `R`.
                - COMPUTE THE AVAILABILITY MATRIX `A` BY SUMMING UP THE POSITIVE RESPONSIBILITIES FOR EACH SAMPLE ACROSS ALL SAMPLES.
                - CREATE A DIAGONAL MATRIX `D_A` USING THE AVAILABILITY VALUES ON THE DIAGONAL.
                - UPDATE THE AVAILABILITY MATRIX `A`:
                    - FOR EACH SAMPLE, SET THE AVAILABILITY TO BE THE MINIMUM VALUE BETWEEN 0 AND THE SUM OF POSITIVE RESPONSIBILITIES FOR THAT SAMPLE.
                    - PRESERVE THE ORIGINAL SELF-AVAILABILITIES BY SETTING THE DIAGONAL ELEMENTS OF `A` TO BE THE CORRESPONDING VALUES FROM `D_A`.
            2. OUTPUT: RETURN THE UPDATED AVAILABILITY MATRIX `A`.
        THE `__UPDATE_AVAILABILITIES__` FUNCTION ADJUSTS THE AVAILABILITIES MATRIX `A` BY CONSIDERING THE POSITIVE RESPONSIBILITIES. IT ENSURES THAT THE SELF-AVAILABILITIES ARE PRESERVED AND UPDATES THE AVAILABILITY VALUES BASED ON THE SUM OF POSITIVE RESPONSIBILITIES FOR EACH SAMPLE. THIS STEP INFLUENCES THE EXEMPLAR SELECTION PROCESS IN THE AFFINITY PROPAGATION ALGORITHM, CONTRIBUTING TO IMPROVED CLUSTER ASSIGNMENTS."""
        R_P = np.maximum(R, 0)  # CALCULATE NON-NEGATIVE RESPONSIBILITIES
        R_P[np.arange(N_SAMPLES), np.arange(N_SAMPLES)] = R[np.arange(
            N_SAMPLES), np.arange(N_SAMPLES)]  # PRESERVE SELF-RESPONSIBILITIES
        A = np.sum(R_P, axis=0)  # CALCULATE AVAILABILITIES
        D_A = np.diag(A)  # CREATE DIAGONAL MATRIX USING AVAILABILITY VALUES
        # SET AVAILABILITIES TO BE MINIMUM OF 0 AND SUM OF POSITIVE RESPONSIBILITIES
        A = np.minimum(A, 0)
        A[np.arange(N_SAMPLES), np.arange(N_SAMPLES)] = np.diag(
            D_A)  # PRESERVE SELF-AVAILABILITIES
        return A  # RETURN UPDATED AVAILABILITIES MATRIX

    # THE `__CHECK_CONVERGENCE__` FUNCTION IS RESPONSIBLE FOR CHECKING THE CONVERGENCE OF THE AFFINITY PROPAGATION ALGORITHM BY COMPARING THE CURRENT RESPONSIBILITIES AND AVAILABILITIES MATRICES WITH THEIR PREVIOUS ITERATIONS.
    #     1. **INPUT PARAMETERS**:
    #         - `R`: THE CURRENT RESPONSIBILITIES MATRIX OF SHAPE `(N_SAMPLES, N_SAMPLES)`.
    #         - `R_OLD`: IS USED TO STORE THE PREVIOUS RESPONSIBILITIES MATRIX DURING EACH ITERATION.
    #         - `A`: THE CURRENT AVAILABILITIES MATRIX OF SHAPE `(N_SAMPLES, N_SAMPLES)`.
    #         - `A_OLD`: IS USED TO STORE THE PREVIOUS AVAILABILITY MATRIX DURING EACH ITERATION.
    #     2. **CONVERGENCE CHECK**:
    #         - CHECK WHETHER THE CURRENT RESPONSIBILITIES `R` AND AVAILABILITIES `A` MATRICES ARE CLOSE TO THEIR RESPECTIVE OLD MATRICES, DENOTED AS `R_OLD` AND `A_OLD`, WITHIN A SPECIFIED TOLERANCE.
    #         - USE THE `NP.ALLCLOSE` FUNCTION TO PERFORM THE ELEMENT-WISE COMPARISON OF THE MATRICES.
    #         - THE `NP.ALLCLOSE` FUNCTION CHECKS IF ALL ELEMENTS ARE WITHIN A GIVEN ABSOLUTE TOLERANCE (SET TO `1E-6` IN THIS CASE) OF THE CORRESPONDING ELEMENTS IN THE OLD MATRICES.
    #         - IF BOTH THE RESPONSIBILITIES AND AVAILABILITIES MATRICES SATISFY THE CONVERGENCE CRITERIA, THE FUNCTION RETURNS `TRUE`, INDICATING CONVERGENCE.
    #         - IF EITHER THE RESPONSIBILITIES OR AVAILABILITIES MATRICES DO NOT MEET THE CONVERGENCE CRITERIA, THE FUNCTION RETURNS `FALSE`, INDICATING NON-CONVERGENCE.
    #     3. **OUTPUT**: RETURN THE BOOLEAN RESULT OF THE CONVERGENCE CHECK.
    # THE `__CHECK_CONVERGENCE__` FUNCTION COMPARES THE CURRENT RESPONSIBILITIES AND AVAILABILITIES MATRICES WITH THEIR PREVIOUS ITERATIONS TO DETERMINE WHETHER THE ALGORITHM HAS CONVERGED. BY EVALUATING THE SIMILARITY OF THE MATRICES WITHIN A SPECIFIED TOLERANCE, IT DETERMINES WHETHER THE ALGORITHM HAS REACHED A STABLE STATE. THIS CONVERGENCE CHECK IS ESSENTIAL FOR CONTROLLING THE TERMINATION CONDITION OF THE AFFINITY PROPAGATION ALGORITHM AND ENSURING THAT THE CLUSTERING RESULTS ARE RELIABLE.
    def __CHECK_CONVERGENCE__(R, R_OLD, A, A_OLD):
        """THE `__CHECK_CONVERGENCE__` FUNCTION IS RESPONSIBLE FOR CHECKING THE CONVERGENCE OF THE AFFINITY PROPAGATION ALGORITHM BY COMPARING THE CURRENT RESPONSIBILITIES AND AVAILABILITIES MATRICES WITH THEIR PREVIOUS ITERATIONS.
            1. CONVERGENCE CHECK:
                - COMPARE THE CURRENT RESPONSIBILITIES AND AVAILABILITIES MATRICES (`R` AND `A`) WITH THEIR RESPECTIVE PREVIOUS ITERATIONS (`R_OLD` AND `A_OLD`).
                - USE THE `NP.ALLCLOSE` FUNCTION TO CHECK IF ALL CORRESPONDING ELEMENTS ARE WITHIN A SPECIFIED TOLERANCE.
                - IF BOTH MATRICES SATISFY THE CONVERGENCE CRITERIA, RETURN `TRUE` TO INDICATE CONVERGENCE.
                - IF EITHER MATRIX DOES NOT MEET THE CONVERGENCE CRITERIA, RETURN `FALSE` TO INDICATE NON-CONVERGENCE.
            2. OUTPUT: RETURN THE BOOLEAN RESULT OF THE CONVERGENCE CHECK.
        THE `__CHECK_CONVERGENCE__` FUNCTION DETERMINES WHETHER THE AFFINITY PROPAGATION ALGORITHM HAS CONVERGED BY COMPARING THE CURRENT AND PREVIOUS RESPONSIBILITIES AND AVAILABILITIES MATRICES. IT ENSURES THAT THE MATRICES HAVE REACHED A STABLE STATE WITHIN A SPECIFIED TOLERANCE. THIS CHECK PLAYS A CRUCIAL ROLE IN CONTROLLING THE TERMINATION CONDITION OF THE ALGORITHM AND ENSURING THE ACCURACY OF THE CLUSTERING RESULTS."""
        return np.allclose(R, R_OLD, atol=1e-6) and np.allclose(A, A_OLD, atol=1e-6)  # CHECK IF THE CURRENT RESPONSIBILITIES AND AVAILABILITIES MATRICES ARE CLOSE TO THEIR RESPECTIVE OLD MATRICES WITHIN A SPECIFIED TOLERANCE

    N_SAMPLES = X.shape[0]  # GET THE NUMBER OF SAMPLES
    S = __CALCULATE_SIMILARITY_MATRIX__(X)  # CALCULATE THE SIMILARITY MATRIX
    # INITIALIZE THE AVAILABILITIES MATRIX
    A = np.zeros((N_SAMPLES, N_SAMPLES))
    # INITIALIZE THE RESPONSIBILITIES MATRIX
    R = np.zeros((N_SAMPLES, N_SAMPLES))
    for _ in range(MAX_ITERATIONS):  # ITERATE FOR THE MAXIMUM NUMBER OF ITERATIONS
        R_OLD = R.copy()  # STORE THE PREVIOUS RESPONSIBILITIES MATRIX
        A_OLD = A.copy()  # STORE THE PREVIOUS AVAILABILITIES MATRIX
        # UPDATE THE RESPONSIBILITIES MATRIX
        R = __UPDATE_RESPONSIBILITIES__(A, S)
        A = __UPDATE_AVAILABILITIES__(R)  # UPDATE THE AVAILABILITIES MATRIX
        # CHECK IF THE ALGORITHM HAS CONVERGED
        if __CHECK_CONVERGENCE__(R, R_OLD, A, A_OLD):
            CONVERGENCE_ITERATIONS -= 1  # DECREMENT THE NUMBER OF CONVERGENCE ITERATIONS
            if CONVERGENCE_ITERATIONS == 0:  # CHECK IF THE ALGORITHM HAS CONVERGED FOR THE SPECIFIED NUMBER OF ITERATIONS
                break  # BREAK OUT OF THE LOOP
        # DAMPEN THE AVAILABILITIES MATRIX
        A = DAMPING * A + (1 - DAMPING) * A_OLD
        # DAMPEN THE RESPONSIBILITIES MATRIX
        R = DAMPING * R + (1 - DAMPING) * R_OLD
    # GET THE INDICES OF THE CLUSTER CENTERS
    CLUSTER_CENTER_IDXS = np.where(np.diag(A + R) > 0)[0]
    # GET THE LABELS OF THE SAMPLES
    LABELS = np.argmax(S[:, CLUSTER_CENTER_IDXS], axis=1)
    # ASSIGN THE CLUSTER CENTER SAMPLES TO THEIR OWN CLUSTERS
    LABELS[CLUSTER_CENTER_IDXS] = np.arange(len(CLUSTER_CENTER_IDXS))
    # RETURN THE CLUSTER CENTER INDICES AND SAMPLE LABELS
    return CLUSTER_CENTER_IDXS, LABELS
