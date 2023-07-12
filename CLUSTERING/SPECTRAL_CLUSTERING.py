import numpy as np
from .K_MEANS import K_MEANS_CLUSTERING

# THE `GAUSSIAN_AFFINITY_MATRIX` FUNCTION IS A UTILITY FUNCTION USED IN THE `SPECTRAL_CLUSTERING` ALGORITHM TO COMPUTE THE AFFINITY MATRIX BASED ON THE INPUT DATA USING A GAUSSIAN SIMILARITY MEASURE.
#     1. THE FUNCTION TAKES ONE PARAMETER AS INPUT:
#         - `X`: THE INPUT DATA MATRIX, WHERE EACH ROW REPRESENTS A DATA POINT AND EACH COLUMN REPRESENTS A FEATURE.
#     2. THE FUNCTION PERFORMS THE FOLLOWING STEPS TO COMPUTE THE GAUSSIAN AFFINITY MATRIX:
#         A. PAIRWISE DISTANCES:
#             - COMPUTE THE PAIRWISE DISTANCES BETWEEN ALL DATA POINTS IN THE INPUT DATA MATRIX.
#             - CALCULATE THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF DATA POINTS USING `NP.LINALG.NORM` WITH `AXIS=2`.
#             - THIS RESULTS IN A DISTANCE MATRIX WHERE EACH ELEMENT REPRESENTS THE EUCLIDEAN DISTANCE BETWEEN TWO DATA POINTS.
#         B. MEDIAN DISTANCE:
#             - CALCULATE THE MEDIAN OF THE PAIRWISE DISTANCES OBTAINED IN THE PREVIOUS STEP.
#             - THE MEDIAN DISTANCE IS A MEASURE OF THE TYPICAL DISTANCE BETWEEN DATA POINTS AND IS USED AS A PARAMETER IN THE GAUSSIAN SIMILARITY MEASURE.
#         C. AFFINITY MATRIX:
#             - CONSTRUCT THE AFFINITY MATRIX USING THE GAUSSIAN SIMILARITY MEASURE.
#             - ITERATE OVER EACH PAIR OF DATA POINTS AND COMPUTE THEIR GAUSSIAN SIMILARITY.
#             - THE GAUSSIAN SIMILARITY BETWEEN TWO DATA POINTS IS CALCULATED AS FOLLOWS:
#                 - TAKE THE NEGATIVE SQUARED EUCLIDEAN DISTANCE BETWEEN THE DATA POINTS.
#                 - DIVIDE BY TWICE THE SQUARED MEDIAN DISTANCE.
#                 - APPLY THE EXPONENTIAL FUNCTION TO THE RESULT.
#             - THE RESULTING SIMILARITY VALUES REPRESENT THE STRENGTH OF THE CONNECTION OR SIMILARITY BETWEEN DATA POINTS.
#     3. FINALLY, THE FUNCTION RETURNS THE COMPUTED GAUSSIAN AFFINITY MATRIX.
# IN SUMMARY, THE `GAUSSIAN_AFFINITY_MATRIX` FUNCTION COMPUTES THE AFFINITY MATRIX USING A GAUSSIAN SIMILARITY MEASURE. IT CALCULATES THE PAIRWISE DISTANCES BETWEEN DATA POINTS, DETERMINES THE MEDIAN DISTANCE AS A PARAMETER, AND APPLIES THE GAUSSIAN FUNCTION TO OBTAIN THE AFFINITY VALUES. THE RESULTING AFFINITY MATRIX CAPTURES THE SIMILARITY RELATIONSHIPS BETWEEN DATA POINTS, WHICH IS AN ESSENTIAL COMPONENT FOR SPECTRAL CLUSTERING TO IDENTIFY MEANINGFUL CLUSTERS.
def GAUSSIAN_AFFINITY_MATRIX(X):
    """THE `GAUSSIAN_AFFINITY_MATRIX` FUNCTION IS A UTILITY FUNCTION USED IN THE SPECTRAL CLUSTERING ALGORITHM. IT PERFORMS THE FOLLOWING STEPS:
        1. PAIRWISE DISTANCES:
            - COMPUTE THE PAIRWISE DISTANCES BETWEEN ALL DATA POINTS IN THE INPUT DATA MATRIX.
            - THIS STEP MEASURES THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF DATA POINTS.
        2. MEDIAN DISTANCE:
            - CALCULATE THE MEDIAN OF THE PAIRWISE DISTANCES.
            - THE MEDIAN DISTANCE REPRESENTS A TYPICAL DISTANCE BETWEEN DATA POINTS AND SERVES AS A PARAMETER IN THE GAUSSIAN SIMILARITY MEASURE.
        3. AFFINITY MATRIX:
            - CONSTRUCT THE AFFINITY MATRIX BASED ON THE GAUSSIAN SIMILARITY MEASURE.
            - ITERATE OVER EACH PAIR OF DATA POINTS AND COMPUTE THEIR GAUSSIAN SIMILARITY.
            - THE SIMILARITY IS CALCULATED USING THE NEGATIVE SQUARED EUCLIDEAN DISTANCE DIVIDED BY TWICE THE SQUARED MEDIAN DISTANCE, FOLLOWED BY THE EXPONENTIAL FUNCTION.
            - THE RESULTING SIMILARITY VALUES INDICATE THE STRENGTH OF THE CONNECTION OR SIMILARITY BETWEEN DATA POINTS.
        4. OUTPUT: RETURN THE COMPUTED GAUSSIAN AFFINITY MATRIX.
    THE `GAUSSIAN_AFFINITY_MATRIX` FUNCTION PLAYS A CRUCIAL ROLE IN SPECTRAL CLUSTERING BY DETERMINING THE AFFINITY BETWEEN DATA POINTS. BY CALCULATING THE GAUSSIAN SIMILARITY BASED ON PAIRWISE DISTANCES, IT CAPTURES THE UNDERLYING RELATIONSHIPS IN THE DATA, ALLOWING SPECTRAL CLUSTERING TO IDENTIFY CLUSTERS BASED ON THE STRENGTH OF CONNECTIONS BETWEEN DATA POINTS."""
    DISTANCES = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    SIGMA = np.median(DISTANCES)
    AFFINITY_MATRIX = np.exp(-DISTANCES ** 2 / (2 * SIGMA ** 2))
    return AFFINITY_MATRIX

# THE `EPSILON_NEIGHBOURHOOD_AFFINITY_MATRIX` FUNCTION IS A UTILITY FUNCTION USED IN THE `SPECTRAL_CLUSTERING` ALGORITHM TO COMPUTE THE AFFINITY MATRIX BASED ON THE INPUT DATA USING AN EPSILON NEIGHBOURHOOD APPROACH.
#     1. THE FUNCTION TAKES TWO PARAMETERS AS INPUT:
#         - `X`: THE INPUT DATA MATRIX, WHERE EACH ROW REPRESENTS A DATA POINT AND EACH COLUMN REPRESENTS A FEATURE.
#         - `EPSILON`: THE MAXIMUM DISTANCE THRESHOLD FOR CONSIDERING TWO DATA POINTS AS NEIGHBOURS.
#     2. THE FUNCTION PERFORMS THE FOLLOWING STEPS TO COMPUTE THE EPSILON NEIGHBOURHOOD AFFINITY MATRIX:
#         A. PAIRWISE DISTANCES:
#             - COMPUTE THE PAIRWISE DISTANCES BETWEEN ALL DATA POINTS IN THE INPUT DATA MATRIX.
#             - CALCULATE THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF DATA POINTS USING `NP.LINALG.NORM` WITH `AXIS=2`.
#             - THIS RESULTS IN A DISTANCE MATRIX WHERE EACH ELEMENT REPRESENTS THE EUCLIDEAN DISTANCE BETWEEN TWO DATA POINTS.
#         B. AFFINITY MATRIX:
#             - CONSTRUCT THE AFFINITY MATRIX BASED ON THE EPSILON NEIGHBOURHOOD APPROACH.
#             - ITERATE OVER EACH PAIR OF DATA POINTS AND DETERMINE THEIR AFFINITY.
#             - IF THE EUCLIDEAN DISTANCE BETWEEN TWO DATA POINTS IS LESS THAN OR EQUAL TO THE EPSILON THRESHOLD, ASSIGN AN AFFINITY VALUE OF 1 TO INDICATE THEY ARE NEIGHBOURS. OTHERWISE, ASSIGN AN AFFINITY VALUE OF 0 TO INDICATE THEY ARE NOT NEIGHBOURS.
#             - THIS CREATES A BINARY AFFINITY MATRIX WHERE THE PRESENCE OR ABSENCE OF A CONNECTION REPRESENTS THE NEIGHBOURHOOD RELATIONSHIP BETWEEN DATA POINTS.
#     3. FINALLY, THE FUNCTION RETURNS THE COMPUTED EPSILON NEIGHBOURHOOD AFFINITY MATRIX.
# IN SUMMARY, THE `EPSILON_NEIGHBOURHOOD_AFFINITY_MATRIX` FUNCTION COMPUTES THE AFFINITY MATRIX USING AN EPSILON NEIGHBOURHOOD APPROACH. IT CALCULATES THE PAIRWISE DISTANCES BETWEEN DATA POINTS, AND BASED ON A GIVEN EPSILON THRESHOLD, ASSIGNS AFFINITY VALUES OF 1 OR 0 TO INDICATE WHETHER DATA POINTS ARE NEIGHBOURS OR NOT. THE RESULTING AFFINITY MATRIX CAPTURES THE NEIGHBOURHOOD RELATIONSHIPS BETWEEN DATA POINTS, WHICH IS CRUCIAL FOR SPECTRAL CLUSTERING TO IDENTIFY CLUSTERS BASED ON THE CONNECTEDNESS OF DATA POINTS WITHIN A CERTAIN DISTANCE THRESHOLD.
def EPSILON_NEIGHBOURHOOD_AFFINITY_MATRIX(X, EPSILON=0.5):
    """THE `EPSILON_NEIGHBOURHOOD_AFFINITY_MATRIX` FUNCTION IS A UTILITY FUNCTION USED IN THE SPECTRAL CLUSTERING ALGORITHM. IT PERFORMS THE FOLLOWING STEPS:
        1. PAIRWISE DISTANCES:
            - COMPUTE THE PAIRWISE DISTANCES BETWEEN ALL DATA POINTS IN THE INPUT DATA MATRIX.
            - THIS STEP MEASURES THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF DATA POINTS.
        2. AFFINITY MATRIX:
            - CONSTRUCT THE AFFINITY MATRIX BASED ON AN EPSILON NEIGHBOURHOOD APPROACH.
            - ITERATE OVER EACH PAIR OF DATA POINTS AND DETERMINE THEIR AFFINITY BASED ON A GIVEN EPSILON THRESHOLD.
            - IF THE EUCLIDEAN DISTANCE BETWEEN TWO DATA POINTS IS LESS THAN OR EQUAL TO THE EPSILON THRESHOLD, ASSIGN AN AFFINITY VALUE OF 1 TO INDICATE THEY ARE NEIGHBOURS. OTHERWISE, ASSIGN AN AFFINITY VALUE OF 0 TO INDICATE THEY ARE NOT NEIGHBOURS.
            - THE RESULTING AFFINITY MATRIX REPRESENTS THE NEIGHBOURHOOD RELATIONSHIPS BETWEEN DATA POINTS.
        3. OUTPUT: RETURN THE COMPUTED EPSILON NEIGHBOURHOOD AFFINITY MATRIX.
    THE `EPSILON_NEIGHBOURHOOD_AFFINITY_MATRIX` FUNCTION PLAYS A VITAL ROLE IN SPECTRAL CLUSTERING BY ESTABLISHING THE AFFINITY BETWEEN DATA POINTS USING AN EPSILON NEIGHBOURHOOD CRITERION. BY DEFINING NEIGHBOURS BASED ON THE DISTANCE THRESHOLD, IT IDENTIFIES WHICH DATA POINTS ARE CONNECTED WITHIN A CERTAIN RANGE. THIS INFORMATION IS CRUCIAL FOR SPECTRAL CLUSTERING TO IDENTIFY CLUSTERS BASED ON THE NEIGHBOURHOOD RELATIONSHIPS AMONG DATA POINTS."""
    DISTANCES = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    AFFINITY_MATRIX = np.where(DISTANCES <= EPSILON, 1, 0)
    return AFFINITY_MATRIX

# THE `COSINE_SIMILARITY_AFFINITY_MATRIX` FUNCTION IS A UTILITY FUNCTION USED IN THE `SPECTRAL_CLUSTERING` ALGORITHM TO COMPUTE THE AFFINITY MATRIX BASED ON THE INPUT DATA USING COSINE SIMILARITY.
#     1. THE FUNCTION TAKES ONE PARAMETER AS INPUT:
#         - `X`: THE INPUT DATA MATRIX, WHERE EACH ROW REPRESENTS A DATA POINT AND EACH COLUMN REPRESENTS A FEATURE.
#     2. THE FUNCTION PERFORMS THE FOLLOWING STEPS TO COMPUTE THE COSINE SIMILARITY AFFINITY MATRIX:
#         A. COSINE SIMILARITY CALCULATION:
#             - DEFINE A HELPER FUNCTION CALLED `COSINE_SIMILARITY` THAT CALCULATES THE COSINE SIMILARITY BETWEEN TWO VECTORS.
#             - THE COSINE SIMILARITY BETWEEN TWO VECTORS IS OBTAINED BY COMPUTING THE DOT PRODUCT OF THE VECTORS AND DIVIDING IT BY THE PRODUCT OF THEIR MAGNITUDES.
#         B. AFFINITY MATRIX:
#             - CREATE AN AFFINITY MATRIX INITIALIZED WITH ZEROS, WITH DIMENSIONS BASED ON THE NUMBER OF SAMPLES IN THE INPUT DATA.
#             - ITERATE OVER EACH PAIR OF DATA POINTS USING NESTED LOOPS.
#             - FOR EACH PAIR OF DATA POINTS, COMPUTE THE COSINE SIMILARITY BETWEEN THEIR RESPECTIVE FEATURE VECTORS USING THE `COSINE_SIMILARITY` FUNCTION.
#             - ASSIGN THE COMPUTED COSINE SIMILARITY VALUE TO THE CORRESPONDING ENTRY IN THE AFFINITY MATRIX FOR BOTH POSITIONS (I, J) AND (J, I).
#             - THE RESULTING AFFINITY MATRIX CONTAINS COSINE SIMILARITY VALUES THAT REPRESENT THE PAIRWISE SIMILARITY BETWEEN DATA POINTS.
#     3. FINALLY, THE FUNCTION RETURNS THE COMPUTED COSINE SIMILARITY AFFINITY MATRIX.
# IN SUMMARY, THE `COSINE_SIMILARITY_AFFINITY_MATRIX` FUNCTION CALCULATES THE AFFINITY MATRIX USING COSINE SIMILARITY AS A MEASURE OF SIMILARITY BETWEEN DATA POINTS. IT ITERATES OVER EACH PAIR OF DATA POINTS, COMPUTES THEIR COSINE SIMILARITY USING THE DOT PRODUCT OF THEIR FEATURE VECTORS, AND ASSIGNS THE RESULTING VALUES TO THE AFFINITY MATRIX. THE RESULTING AFFINITY MATRIX CAPTURES THE PAIRWISE SIMILARITY RELATIONSHIPS BETWEEN DATA POINTS, WHICH IS CRUCIAL FOR SPECTRAL CLUSTERING TO IDENTIFY CLUSTERS BASED ON THE COSINE SIMILARITY AMONG DATA POINTS' FEATURE VECTORS.
def COSINE_SIMILARITY_AFFINITY_MATRIX(X):
    """THE `COSINE_SIMILARITY_AFFINITY_MATRIX` FUNCTION IS A UTILITY FUNCTION USED IN THE SPECTRAL CLUSTERING ALGORITHM. IT PERFORMS THE FOLLOWING STEPS:
        1. COSINE SIMILARITY CALCULATION:
            - DEFINE A HELPER FUNCTION THAT CALCULATES THE COSINE SIMILARITY BETWEEN TWO VECTORS.
            - THE COSINE SIMILARITY IS COMPUTED AS THE DOT PRODUCT OF THE VECTORS DIVIDED BY THE PRODUCT OF THEIR MAGNITUDES.
        2. AFFINITY MATRIX:
            - CREATE AN AFFINITY MATRIX INITIALIZED WITH ZEROS, WITH DIMENSIONS BASED ON THE NUMBER OF SAMPLES IN THE INPUT DATA.
            - ITERATE OVER EACH PAIR OF DATA POINTS.
            - FOR EACH PAIR, CALCULATE THE COSINE SIMILARITY BETWEEN THEIR RESPECTIVE FEATURE VECTORS USING THE DEFINED HELPER FUNCTION.
            - ASSIGN THE COMPUTED COSINE SIMILARITY VALUE TO THE CORRESPONDING ENTRY IN THE AFFINITY MATRIX FOR BOTH POSITIONS (I, J) AND (J, I).
        3. OUTPUT: RETURN THE COMPUTED COSINE SIMILARITY AFFINITY MATRIX.
    THE `COSINE_SIMILARITY_AFFINITY_MATRIX` FUNCTION CALCULATES THE AFFINITY MATRIX USING COSINE SIMILARITY, A MEASURE OF SIMILARITY BETWEEN DATA POINTS' FEATURE VECTORS. BY COMPUTING THE COSINE SIMILARITY FOR ALL PAIRWISE COMBINATIONS OF DATA POINTS, IT ESTABLISHES THE SIMILARITY RELATIONSHIPS AMONG THE DATA POINTS. THIS AFFINITY MATRIX IS THEN UTILIZED BY THE SPECTRAL CLUSTERING ALGORITHM TO IDENTIFY CLUSTERS BASED ON THE COSINE SIMILARITY OF THE FEATURE VECTORS."""

    # THE `__COSINE_SIMILARITY__` FUNCTION IS A HELPER FUNCTION USED IN THE `COSINE_SIMILARITY_AFFINITY_MATRIX` FUNCTION TO CALCULATE THE COSINE SIMILARITY BETWEEN TWO VECTORS.
    #     1. THE FUNCTION TAKES TWO PARAMETERS AS INPUT:
    #         - `VECTOR_1`: THE FIRST VECTOR.
    #         - `VECTOR_2`: THE SECOND VECTOR.
    #     2. THE FUNCTION PERFORMS THE FOLLOWING STEPS TO CALCULATE THE COSINE SIMILARITY:
    #         A. DOT PRODUCT CALCULATION:
    #             - COMPUTE THE DOT PRODUCT OF `VECTOR_1` AND `VECTOR_2`.
    #             - THE DOT PRODUCT IS OBTAINED BY MULTIPLYING THE CORRESPONDING ELEMENTS OF THE VECTORS AND SUMMING THE RESULTS.
    #         B. MAGNITUDE CALCULATION:
    #             - CALCULATE THE MAGNITUDE (EUCLIDEAN NORM) OF `VECTOR_1`.
    #             - COMPUTE THE SQUARE ROOT OF THE SUM OF THE SQUARES OF THE ELEMENTS OF `VECTOR_1`.
    #             - REPEAT THE SAME PROCESS TO CALCULATE THE MAGNITUDE OF `VECTOR_2`.
    #         C. COSINE SIMILARITY:
    #             - DIVIDE THE DOT PRODUCT OF THE VECTORS BY THE PRODUCT OF THEIR MAGNITUDES.
    #             - THIS CALCULATION YIELDS THE COSINE SIMILARITY VALUE, WHICH REPRESENTS THE COSINE OF THE ANGLE BETWEEN THE VECTORS.
    #     3. FINALLY, THE FUNCTION RETURNS THE COMPUTED COSINE SIMILARITY VALUE.
    # IN SUMMARY, THE `__COSINE_SIMILARITY__` FUNCTION CALCULATES THE COSINE SIMILARITY BETWEEN TWO VECTORS BY PERFORMING THE DOT PRODUCT OF THE VECTORS AND DIVIDING IT BY THE PRODUCT OF THEIR MAGNITUDES. THE RESULTING VALUE REPRESENTS THE COSINE OF THE ANGLE BETWEEN THE VECTORS AND SERVES AS A MEASURE OF THEIR SIMILARITY. IN THE CONTEXT OF SPECTRAL CLUSTERING, COSINE SIMILARITY IS UTILIZED TO CAPTURE THE SIMILARITY RELATIONSHIPS BETWEEN DATA POINTS' FEATURE VECTORS, WHICH AIDS IN IDENTIFYING CLUSTERS.
    def __COSINE_SIMILARITY__(VECTOR_1, VECTOR_2):
        """THE `__COSINE_SIMILARITY__` FUNCTION IS A HELPER FUNCTION USED IN THE SPECTRAL CLUSTERING ALGORITHM. IT PERFORMS THE FOLLOWING STEPS:
            1. DOT PRODUCT CALCULATION:
                - COMPUTE THE DOT PRODUCT OF TWO VECTORS BY MULTIPLYING THEIR CORRESPONDING ELEMENTS AND SUMMING THE RESULTS.
                - THIS STEP CAPTURES THE SIMILARITY BETWEEN THE VECTORS BASED ON THE MAGNITUDES AND ALIGNMENT OF THEIR ELEMENTS.
            2. MAGNITUDE CALCULATION:
                - CALCULATE THE MAGNITUDE (EUCLIDEAN NORM) OF EACH VECTOR BY TAKING THE SQUARE ROOT OF THE SUM OF THE SQUARES OF ITS ELEMENTS.
                - THIS STEP DETERMINES THE LENGTH OR MAGNITUDE OF EACH VECTOR.
            3. COSINE SIMILARITY:
                - DIVIDE THE DOT PRODUCT OF THE VECTORS BY THE PRODUCT OF THEIR MAGNITUDES.
                - THIS COMPUTATION YIELDS THE COSINE OF THE ANGLE BETWEEN THE VECTORS AND REPRESENTS THEIR SIMILARITY.
                - A VALUE OF 1 INDICATES IDENTICAL OR PERFECTLY ALIGNED VECTORS, WHILE A VALUE OF -1 INDICATES OPPOSITE OR PERFECTLY MISALIGNED VECTORS.
            4. OUTPUT: RETURN THE COMPUTED COSINE SIMILARITY VALUE.
        THE `__COSINE_SIMILARITY__` FUNCTION CALCULATES THE COSINE SIMILARITY BETWEEN TWO VECTORS, CONSIDERING THEIR ALIGNMENT AND MAGNITUDES. THIS SIMILARITY MEASURE HELPS ASSESS THE SIMILARITY BETWEEN FEATURE VECTORS IN THE DATA AND IS UTILIZED IN SPECTRAL CLUSTERING TO CAPTURE THE SIMILARITY RELATIONSHIPS BETWEEN DATA POINTS."""
        DOT_PRODUCT = np.dot(VECTOR_1, VECTOR_2.T)
        NORM_PRODUCT = np.linalg.norm(VECTOR_1) * np.linalg.norm(VECTOR_2)
        return DOT_PRODUCT / NORM_PRODUCT

    NUM_SAMPLES = len(X)
    AFFINITY_MATRIX = np.zeros((NUM_SAMPLES, NUM_SAMPLES))
    for i in range(NUM_SAMPLES):
        for j in range(i, NUM_SAMPLES):
            SIMILARITY = __COSINE_SIMILARITY__(X[i].reshape(1, -1), X[j].reshape(1, -1))
            AFFINITY_MATRIX[i, j] = SIMILARITY
            AFFINITY_MATRIX[j, i] = SIMILARITY
    return AFFINITY_MATRIX

# THE `SPECTRAL_CLUSTERING` FUNCTION IS AN IMPLEMENTATION OF THE SPECTRAL CLUSTERING ALGORITHM, A POPULAR TECHNIQUE USED FOR CLUSTERING DATA POINTS.
#     1. THE FUNCTION TAKES THREE PARAMETERS AS INPUT:
#         - `X`: THE INPUT DATA MATRIX, WHERE EACH ROW REPRESENTS A DATA POINT AND EACH COLUMN REPRESENTS A FEATURE.
#         - `K`: THE NUMBER OF CLUSTERS TO BE CREATED.
#         - `AFFINITY_MATRIX_FUNCTION`: A FUNCTION THAT CALCULATES THE AFFINITY MATRIX BASED ON THE INPUT DATA. IF NOT PROVIDED, IT DEFAULTS TO THE `GAUSSIAN_AFFINITY_MATRIX` FUNCTION.
#     2. INSIDE THE FUNCTION, THERE ARE TWO NESTED HELPER FUNCTIONS:
#         - `__COMPUTE_NORMALIZED_LAPLACIAN__`: THIS FUNCTION CALCULATES THE NORMALIZED LAPLACIAN MATRIX GIVEN AN AFFINITY MATRIX. THE LAPLACIAN MATRIX IS COMPUTED AS THE DIFFERENCE BETWEEN THE DEGREE MATRIX AND THE AFFINITY MATRIX. THE DEGREE MATRIX IS A DIAGONAL MATRIX WHERE EACH DIAGONAL ELEMENT REPRESENTS THE SUM OF THE CORRESPONDING ROW OF THE AFFINITY MATRIX. THE NORMALIZED LAPLACIAN MATRIX IS OBTAINED BY MULTIPLYING THE SQUARE ROOT OF THE INVERSE OF THE DEGREE MATRIX WITH THE LAPLACIAN MATRIX.
#         - `__SELECT_EIGENVECTORS__`: THIS FUNCTION SELECTS THE TOP `K` EIGENVECTORS CORRESPONDING TO THE SMALLEST EIGENVALUES FROM THE EIGENVALUE-EIGENVECTOR PAIRS. IT SORTS THE EIGENVALUES IN ASCENDING ORDER, SORTS THE EIGENVECTORS ACCORDINGLY, AND THEN SELECTS THE FIRST `K` EIGENVECTORS.
#     3. THE AFFINITY MATRIX IS COMPUTED BY CALLING THE `AFFINITY_MATRIX_FUNCTION` WITH THE INPUT DATA `X`. IF `AFFINITY_MATRIX_FUNCTION` IS NOT PROVIDED, IT DEFAULTS TO THE `GAUSSIAN_AFFINITY_MATRIX` FUNCTION.
#     4. THE NORMALIZED LAPLACIAN MATRIX IS COMPUTED BY CALLING THE `__COMPUTE_NORMALIZED_LAPLACIAN__` FUNCTION WITH THE AFFINITY MATRIX.
#     5. EIGENVALUES AND EIGENVECTORS ARE OBTAINED BY PERFORMING AN EIGENVALUE DECOMPOSITION ON THE NORMALIZED LAPLACIAN MATRIX USING `NP.LINALG.EIG`.
#     6. THE `__SELECT_EIGENVECTORS__` FUNCTION IS CALLED TO SELECT THE TOP `K` EIGENVECTORS.
#     7. THE SELECTED EIGENVECTORS ARE THEN PASSED TO THE `K_MEANS_CLUSTERING` FUNCTION, WHICH PERFORMS K-MEANS CLUSTERING ON THE EIGENVECTORS TO ASSIGN CLUSTER LABELS TO THE DATA POINTS.
#     8. FINALLY, THE CLUSTER LABELS ARE RETURNED AS THE OUTPUT OF THE `SPECTRAL_CLUSTERING` FUNCTION.
# IN SUMMARY, THE `SPECTRAL_CLUSTERING` FUNCTION PERFORMS SPECTRAL CLUSTERING BY COMPUTING THE AFFINITY MATRIX BASED ON THE INPUT DATA, CALCULATING THE NORMALIZED LAPLACIAN MATRIX, OBTAINING THE EIGENVALUES AND EIGENVECTORS, SELECTING A SUBSET OF THE EIGENVECTORS, AND APPLYING K-MEANS CLUSTERING ON THE SELECTED EIGENVECTORS TO ASSIGN CLUSTER LABELS TO THE DATA POINTS.
def SPECTRAL_CLUSTERING(X, K, AFFINITY_MATRIX_FUNCTION=GAUSSIAN_AFFINITY_MATRIX):
    """THE `SPECTRAL_CLUSTERING` FUNCTION IS AN IMPLEMENTATION OF THE SPECTRAL CLUSTERING ALGORITHM. IT PERFORMS CLUSTERING ON DATA POINTS BY FOLLOWING THESE STEPS:
        1. AFFINITY MATRIX: CALCULATE THE AFFINITY MATRIX BASED ON THE INPUT DATA. THE AFFINITY MATRIX MEASURES THE SIMILARITY BETWEEN DATA POINTS AND DETERMINES THEIR CONNECTIVITY.
        2. NORMALIZED LAPLACIAN: COMPUTE THE NORMALIZED LAPLACIAN MATRIX, WHICH CAPTURES THE UNDERLYING STRUCTURE OF THE DATA BY CONSIDERING BOTH THE AFFINITY MATRIX AND THE DEGREE MATRIX.
        3. EIGENVALUE DECOMPOSITION: PERFORM EIGENVALUE DECOMPOSITION ON THE NORMALIZED LAPLACIAN MATRIX TO OBTAIN EIGENVALUES AND EIGENVECTORS.
        4. EIGENVECTOR SELECTION: SELECT THE TOP `K` EIGENVECTORS CORRESPONDING TO THE SMALLEST EIGENVALUES. THESE EIGENVECTORS CONTAIN CRUCIAL INFORMATION ABOUT THE DATA'S CLUSTERING STRUCTURE.
        5. K-MEANS CLUSTERING: APPLY THE K-MEANS CLUSTERING ALGORITHM ON THE SELECTED EIGENVECTORS TO ASSIGN CLUSTER LABELS TO THE DATA POINTS.
        6. OUTPUT: RETURN THE CLUSTER LABELS AS THE RESULT OF THE SPECTRAL CLUSTERING ALGORITHM.
    OVERALL, SPECTRAL CLUSTERING LEVERAGES THE SPECTRAL PROPERTIES OF THE DATA TO IDENTIFY MEANINGFUL CLUSTERS. IT STARTS BY CONSTRUCTING AN AFFINITY MATRIX, THEN PROCEEDS WITH EIGENVALUE DECOMPOSITION, EIGENVECTOR SELECTION, AND K-MEANS CLUSTERING TO PRODUCE THE FINAL CLUSTERING RESULT."""

    # THE `__COMPUTE_NORMALIZED_LAPLACIAN__` FUNCTION IS A HELPER FUNCTION USED IN THE `SPECTRAL_CLUSTERING` ALGORITHM TO COMPUTE THE NORMALIZED LAPLACIAN MATRIX GIVEN AN AFFINITY MATRIX.
    #     1. THE FUNCTION TAKES ONE PARAMETER AS INPUT:
    #         - `AFFINITY_MATRIX`: THE AFFINITY MATRIX CALCULATED BASED ON THE INPUT DATA.
    #     2. THE FUNCTION PERFORMS THE FOLLOWING STEPS TO COMPUTE THE NORMALIZED LAPLACIAN MATRIX:
    #         A. DEGREE MATRIX:
    #             - COMPUTE THE DEGREE MATRIX, WHICH IS A DIAGONAL MATRIX WHERE EACH DIAGONAL ELEMENT REPRESENTS THE SUM OF THE CORRESPONDING ROW OF THE AFFINITY MATRIX.
    #             - TO OBTAIN THE DEGREE MATRIX, SUM THE ROWS OF THE AFFINITY MATRIX USING `NP.SUM` ALONG `AXIS=1`.
    #             - CREATE A DIAGONAL MATRIX USING `NP.DIAG` WITH THE COMPUTED ROW SUMS AS THE DIAGONAL ELEMENTS.
    #         B. LAPLACIAN MATRIX:
    #             - CALCULATE THE LAPLACIAN MATRIX BY SUBTRACTING THE AFFINITY MATRIX FROM THE DEGREE MATRIX.
    #             - SUBTRACT THE AFFINITY MATRIX FROM THE DEGREE MATRIX USING THE `-` OPERATOR.
    #         C. SQUARE ROOT OF THE INVERSE DEGREE MATRIX:
    #             - COMPUTE THE SQUARE ROOT OF THE INVERSE OF THE DEGREE MATRIX.
    #             - COMPUTE THE INVERSE OF THE DEGREE MATRIX USING `NP.LINALG.INV`.
    #             - CALCULATE THE SQUARE ROOT OF THE INVERSE DEGREE MATRIX USING `NP.SQRT`.
    #         D. NORMALIZED LAPLACIAN MATRIX:
    #             - MULTIPLY THE SQUARE ROOT OF THE INVERSE DEGREE MATRIX BY THE LAPLACIAN MATRIX TWICE.
    #             - PERFORM MATRIX MULTIPLICATION USING `NP.DOT` TO OBTAIN THE NORMALIZED LAPLACIAN MATRIX.
    #             - THE MULTIPLICATION IS PERFORMED AS FOLLOWS: `NP.DOT(NP.DOT(SQRT_INVERSE_DEGREE_MATRIX, LAPLACIAN_MATRIX), SQRT_INVERSE_DEGREE_MATRIX)`.
    #     3. FINALLY, THE FUNCTION RETURNS THE COMPUTED NORMALIZED LAPLACIAN MATRIX.
    # IN SUMMARY, THE `__COMPUTE_NORMALIZED_LAPLACIAN__` FUNCTION CALCULATES THE NORMALIZED LAPLACIAN MATRIX BY SUBTRACTING THE AFFINITY MATRIX FROM THE DEGREE MATRIX AND THEN MULTIPLYING IT BY THE SQUARE ROOT OF THE INVERSE DEGREE MATRIX. THIS MATRIX TRANSFORMATION IS A CRUCIAL STEP IN SPECTRAL CLUSTERING AS IT CAPTURES THE UNDERLYING STRUCTURE OF THE DATA AND AIDS IN IDENTIFYING CLUSTERS.
    def __COMPUTE_NORMALIZED_LAPLACIAN__(AFFINITY_MATRIX):
        """THE `__COMPUTE_NORMALIZED_LAPLACIAN__` FUNCTION IS A HELPER FUNCTION USED IN THE SPECTRAL CLUSTERING ALGORITHM. IT PERFORMS THE FOLLOWING STEPS:
            1. DEGREE MATRIX: CALCULATE THE DEGREE MATRIX, WHICH SUMS UP THE ROWS OF THE AFFINITY MATRIX AND REPRESENTS THE CONNECTIVITY OF EACH DATA POINT.
            2. LAPLACIAN MATRIX: COMPUTE THE LAPLACIAN MATRIX BY SUBTRACTING THE AFFINITY MATRIX FROM THE DEGREE MATRIX. THIS MATRIX CAPTURES THE PAIRWISE RELATIONSHIPS BETWEEN DATA POINTS.
            3. SQUARE ROOT OF THE INVERSE DEGREE MATRIX: COMPUTE THE SQUARE ROOT OF THE INVERSE OF THE DEGREE MATRIX. THIS TRANSFORMATION NORMALIZES THE CONNECTIVITY INFORMATION.
            4. NORMALIZED LAPLACIAN MATRIX: MULTIPLY THE SQUARE ROOT OF THE INVERSE DEGREE MATRIX BY THE LAPLACIAN MATRIX TWICE. THIS OPERATION SCALES AND REWEIGHTS THE LAPLACIAN MATRIX TO OBTAIN THE NORMALIZED LAPLACIAN MATRIX.
            5. OUTPUT: RETURN THE COMPUTED NORMALIZED LAPLACIAN MATRIX.
        THE NORMALIZED LAPLACIAN MATRIX PLAYS A CRUCIAL ROLE IN SPECTRAL CLUSTERING AS IT ENCAPSULATES THE DATA'S STRUCTURE AND SIMILARITY RELATIONSHIPS. BY CALCULATING THE NORMALIZED LAPLACIAN, SPECTRAL CLUSTERING CAN EFFECTIVELY IDENTIFY CLUSTERS IN THE DATA BY PERFORMING EIGENVALUE DECOMPOSITION AND SUBSEQUENT CLUSTERING TECHNIQUES."""
        DEGREE_MATRIX = np.diag(np.sum(AFFINITY_MATRIX, axis=1))
        LAPLACIAN_MATRIX = DEGREE_MATRIX - AFFINITY_MATRIX
        SQRT_INVERSE_DEGREE_MATRIX = np.sqrt(np.linalg.inv(DEGREE_MATRIX))
        NORMALIZED_LAPLACIAN_MATRIX = np.dot(np.dot(SQRT_INVERSE_DEGREE_MATRIX, LAPLACIAN_MATRIX), SQRT_INVERSE_DEGREE_MATRIX)
        return NORMALIZED_LAPLACIAN_MATRIX

    # THE `__SELECT_EIGENVECTORS__` FUNCTION IS A HELPER FUNCTION USED IN THE `SPECTRAL_CLUSTERING` ALGORITHM TO SELECT A SUBSET OF EIGENVECTORS CORRESPONDING TO THE SMALLEST EIGENVALUES.
    #     1. THE FUNCTION TAKES THREE PARAMETERS AS INPUT:
    #         - `EIGENVALUES`: AN ARRAY CONTAINING THE EIGENVALUES OBTAINED FROM EIGENVALUE DECOMPOSITION.
    #         - `EIGENVECTORS`: A MATRIX WHERE EACH COLUMN REPRESENTS AN EIGENVECTOR OBTAINED FROM EIGENVALUE DECOMPOSITION.
    #         - `K`: THE NUMBER OF EIGENVECTORS TO BE SELECTED.
    #     2. THE FUNCTION PERFORMS THE FOLLOWING STEPS TO SELECT THE EIGENVECTORS:
    #         A. SORTING EIGENVALUES:
    #             - SORT THE EIGENVALUES IN ASCENDING ORDER.
    #             - OBTAIN THE INDICES THAT WOULD SORT THE EIGENVALUES USING `NP.ARGSORT`.
    #             - SORT THE EIGENVALUES USING THE OBTAINED INDICES.
    #         B. SORTING EIGENVECTORS:
    #             - SORT THE EIGENVECTORS ACCORDING TO THE SORTED EIGENVALUES.
    #             - REORDER THE COLUMNS OF THE EIGENVECTOR MATRIX BASED ON THE SORTED EIGENVALUES.
    #         C. SELECTING EIGENVECTORS:
    #             - CHOOSE THE FIRST `K` COLUMNS OF THE SORTED EIGENVECTOR MATRIX.
    #             - SLICE THE SORTED EIGENVECTOR MATRIX TO KEEP ONLY THE FIRST `K` COLUMNS.
    #     3. FINALLY, THE FUNCTION RETURNS THE SELECTED EIGENVECTORS AS THE OUTPUT.
    # IN SUMMARY, THE `__SELECT_EIGENVECTORS__` FUNCTION SORTS THE EIGENVALUES IN ASCENDING ORDER AND SORTS THE CORRESPONDING EIGENVECTORS ACCORDINGLY. IT THEN SELECTS THE TOP `K` EIGENVECTORS BY KEEPING THE FIRST `K` COLUMNS OF THE SORTED EIGENVECTOR MATRIX. THE SELECTED EIGENVECTORS ARE ESSENTIAL FOR CAPTURING THE RELEVANT INFORMATION ABOUT THE CLUSTERING STRUCTURE IN THE DATA AND ARE LATER USED IN THE SPECTRAL CLUSTERING ALGORITHM TO ASSIGN CLUSTER LABELS TO THE DATA POINTS.
    def __SELECT_EIGENVECTORS__(EIGENVALUES, EIGENVECTORS, K):
        """THE `__SELECT_EIGENVECTORS__` FUNCTION IS A HELPER FUNCTION USED IN THE SPECTRAL CLUSTERING ALGORITHM. IT PERFORMS THE FOLLOWING STEPS:
            1. SORTING EIGENVALUES:
            - SORT THE EIGENVALUES OBTAINED FROM EIGENVALUE DECOMPOSITION IN ASCENDING ORDER.
            - THIS STEP DETERMINES THE IMPORTANCE OR SIGNIFICANCE OF EACH EIGENVALUE.
            2. SORTING EIGENVECTORS:
            - SORT THE EIGENVECTORS CORRESPONDING TO THE EIGENVALUES.
            - THE SORTING ENSURES THAT THE EIGENVECTORS ALIGN WITH THEIR RESPECTIVE EIGENVALUES.
            3. SELECTING EIGENVECTORS:
            - CHOOSE THE TOP `K` EIGENVECTORS BASED ON THE SORTED EIGENVALUES.
            - BY SELECTING THE EIGENVECTORS ASSOCIATED WITH THE SMALLEST EIGENVALUES, WE CAPTURE THE ESSENTIAL INFORMATION FOR CLUSTERING.
            4. OUTPUT: RETURN THE SELECTED EIGENVECTORS AS THE RESULT OF THE FUNCTION.
        THE `__SELECT_EIGENVECTORS__` FUNCTION PLAYS A CRITICAL ROLE IN SPECTRAL CLUSTERING BY IDENTIFYING THE MOST INFORMATIVE EIGENVECTORS. BY CONSIDERING THE EIGENVALUES, WHICH INDICATE THE IMPORTANCE OF EACH EIGENVECTOR, THE FUNCTION ENABLES SPECTRAL CLUSTERING TO FOCUS ON THE EIGENVECTORS THAT BEST CAPTURE THE CLUSTERING STRUCTURE IN THE DATA."""
        SORTED_IDXS = np.argsort(EIGENVALUES)
        _ = EIGENVALUES[SORTED_IDXS]
        SORTED_EIGENVECTORS = EIGENVECTORS[:, SORTED_IDXS]
        SELECTED_EIGENVECTORS = SORTED_EIGENVECTORS[:, :K]
        return SELECTED_EIGENVECTORS

    AFFINITY_MATRIX = AFFINITY_MATRIX_FUNCTION(X)
    LAPLACIAN_MATRIX = __COMPUTE_NORMALIZED_LAPLACIAN__(AFFINITY_MATRIX)
    EIGENVALUES, EIGENVECTORS = np.linalg.eig(LAPLACIAN_MATRIX)
    EIGENVECTORS = __SELECT_EIGENVECTORS__(EIGENVALUES, EIGENVECTORS, K)
    LABELS = K_MEANS_CLUSTERING(EIGENVECTORS, K)
    return LABELS