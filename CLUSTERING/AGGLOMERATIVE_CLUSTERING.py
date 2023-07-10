import numpy as np

# THE `AGGLOMERATIVE_CLUSTERING` FUNCTION IS AN IMPLEMENTATION OF THE AGGLOMERATIVE CLUSTERING ALGORITHM.
#     1. IT TAKES TWO INPUT ARGUMENTS: `X`, WHICH IS THE INPUT DATA MATRIX CONTAINING SAMPLES AS ROWS AND FEATURES AS COLUMNS, AND `K`, WHICH IS THE DESIRED NUMBER OF CLUSTERS.
#     2. THE FUNCTION INITIALIZES SOME VARIABLES:
#         - `N_SAMPLES` REPRESENTS THE NUMBER OF SAMPLES IN THE INPUT DATA.
#         - `LABELS` IS AN ARRAY THAT ASSIGNS EACH SAMPLE A LABEL, INITIALLY SET AS THE INDEX OF THE SAMPLE.
#         - `N_SAMPLES_IN_CLUSTERS` IS AN ARRAY THAT KEEPS TRACK OF THE NUMBER OF SAMPLES IN EACH CLUSTER. IT IS INITIALLY SET TO AN ARRAY OF ONES, INDICATING THAT EACH SAMPLE IS ITS OWN CLUSTER.
#     3. THE FUNCTION DEFINES THE `__MERGE_CLUSTERS__` HELPER FUNCTION, WHICH TAKES A PAIR OF INDICES REPRESENTING THE SAMPLES TO MERGE. IT UPDATES THE `LABELS` ARRAY AND THE `N_SAMPLES_IN_CLUSTERS` ARRAY BY MERGING THE CLUSTERS.
#     4. THE FUNCTION ENTERS A WHILE LOOP, WHICH CONTINUES UNTIL THE NUMBER OF UNIQUE LABELS IN `LABELS` IS EQUAL TO THE DESIRED NUMBER OF CLUSTERS `K`. THIS LOOP PERFORMS THE CORE AGGLOMERATIVE CLUSTERING PROCESS.
#     5. WITHIN THE LOOP, THE FUNCTION INITIALIZES `MIN_DISTANCE` TO INFINITY AND `MERGE_IDXS` TO `NONE`. THESE VARIABLES WILL KEEP TRACK OF THE MINIMUM DISTANCE AND THE INDICES OF THE TWO CLUSTERS TO MERGE.
#     6. IT ITERATES OVER ALL PAIRS OF SAMPLES (I, J) IN THE INPUT DATA. FOR EACH PAIR:
#         - IT CHECKS IF THE SAMPLES BELONG TO DIFFERENT CLUSTERS BY COMPARING THEIR LABELS IN THE `LABELS` ARRAY.
#         - IF THE SAMPLES BELONG TO DIFFERENT CLUSTERS, IT COMPUTES THE DISTANCE BETWEEN THEM USING THE `__EUCLIDEAN_DISTANCE__` HELPER FUNCTION, WHICH CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS IN A MULTI-DIMENSIONAL SPACE.
#         - IF THE DISTANCE IS SMALLER THAN THE CURRENT `MIN_DISTANCE`, IT UPDATES `MIN_DISTANCE` AND `MERGE_IDXS` WITH THE NEW MINIMUM DISTANCE AND THE INDICES OF THE SAMPLES TO MERGE.
#     7. AFTER THE INNER LOOP COMPLETES, THE FUNCTION CHECKS IF A PAIR OF SAMPLES TO MERGE HAS BEEN FOUND (`MERGE_IDXS` IS NOT `NONE`). IF SO, IT CALLS THE `__MERGE_CLUSTERS__` HELPER FUNCTION TO MERGE THE CLUSTERS.
#     8. ONCE THE WHILE LOOP FINISHES AND THE DESIRED NUMBER OF CLUSTERS `K` IS REACHED, THE FUNCTION RETURNS THE `LABELS` ARRAY, WHICH CONTAINS THE CLUSTER ASSIGNMENTS FOR EACH SAMPLE.
# OVERALL, THE `AGGLOMERATIVE_CLUSTERING` FUNCTION PERFORMS AGGLOMERATIVE CLUSTERING BY ITERATIVELY MERGING THE CLOSEST PAIR OF CLUSTERS UNTIL THE DESIRED NUMBER OF CLUSTERS IS ACHIEVED. IT UTILIZES THE EUCLIDEAN DISTANCE METRIC TO MEASURE THE SIMILARITY BETWEEN SAMPLES.
def AGGLOMERATIVE_CLUSTERING(X, K):
    """THE `AGGLOMERATIVE_CLUSTERING` FUNCTION IMPLEMENTS THE AGGLOMERATIVE CLUSTERING ALGORITHM, WHICH IS A HIERARCHICAL CLUSTERING TECHNIQUE.
        1. INITIALIZATION: THE FUNCTION INITIALIZES VARIABLES, SUCH AS THE NUMBER OF SAMPLES, LABELS FOR EACH SAMPLE, AND THE NUMBER OF SAMPLES IN EACH CLUSTER.
        2. MERGE CLUSTERS: A HELPER FUNCTION, `__MERGE_CLUSTERS__`, IS DEFINED TO MERGE TWO CLUSTERS BY UPDATING THE LABELS AND THE COUNT OF SAMPLES IN EACH CLUSTER.
        3. CLUSTERING ITERATIONS:
            - THE FUNCTION ENTERS A WHILE LOOP UNTIL THE DESIRED NUMBER OF CLUSTERS IS ACHIEVED.
            - IT ITERATES THROUGH ALL PAIRS OF SAMPLES AND CALCULATES THE DISTANCE BETWEEN THEM USING A CHOSEN DISTANCE METRIC (IN THIS CASE, EUCLIDEAN DISTANCE).
            - THE CLOSEST PAIR OF SAMPLES FROM DIFFERENT CLUSTERS IS IDENTIFIED, AND THEIR CLUSTERS ARE MERGED USING THE `__MERGE_CLUSTERS__` FUNCTION.
        4. RESULT: AFTER THE DESIRED NUMBER OF CLUSTERS IS REACHED, THE FUNCTION RETURNS THE CLUSTER LABELS FOR EACH SAMPLE.
    THE AGGLOMERATIVE CLUSTERING ALGORITHM STARTS WITH EACH SAMPLE AS AN INDIVIDUAL CLUSTER AND ITERATIVELY MERGES THE CLOSEST CLUSTERS BASED ON THE CHOSEN DISTANCE METRIC. THIS PROCESS CONTINUES UNTIL THE DESIRED NUMBER OF CLUSTERS IS OBTAINED. THE ALGORITHM UTILIZES THE CONCEPT OF HIERARCHICAL CLUSTERING, WHERE CLUSTERS ARE HIERARCHICALLY NESTED BASED ON THEIR SIMILARITY OR DISTANCE.
    BY USING AGGLOMERATIVE CLUSTERING, THE FUNCTION CAN GROUP SIMILAR SAMPLES TOGETHER AND FORM A HIERARCHICAL STRUCTURE OF CLUSTERS. THE ALGORITHM IS USEFUL FOR EXPLORATORY DATA ANALYSIS, PATTERN RECOGNITION, AND FINDING NATURAL GROUPINGS WITHIN THE DATA. IT ALLOWS FOR FLEXIBILITY IN THE CHOICE OF DISTANCE METRICS AND CAN HANDLE DIFFERENT TYPES OF DATA."""
    N_SAMPLES = X.shape[0]
    LABELS = np.arange(N_SAMPLES)
    N_SAMPLES_IN_CLUSTERS = np.ones(N_SAMPLES)

    # THE `__MERGE_CLUSTERS__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE AGGLOMERATIVE CLUSTERING ALGORITHM TO MERGE TWO CLUSTERS.
    #     1. IT TAKES `MERGE_IDXS` AS INPUT, WHICH REPRESENTS THE INDICES OF THE TWO SAMPLES TO BE MERGED.
    #     2. THE FUNCTION RETRIEVES THE CLUSTER LABELS OF THE TWO SAMPLES TO BE MERGED, DENOTED AS `CLUSTER_1` AND `CLUSTER_2`, RESPECTIVELY.
    #     3. IT ITERATES OVER ALL THE ELEMENTS IN THE `LABELS` ARRAY, WHICH CONTAINS THE CURRENT CLUSTER ASSIGNMENTS FOR EACH SAMPLE.
    #     4. FOR EACH ELEMENT, IF THE CLUSTER LABEL MATCHES `CLUSTER_2`, IT UPDATES THE LABEL TO `CLUSTER_1`, EFFECTIVELY MERGING THE SAMPLES INTO THE SAME CLUSTER.
    #     5. ADDITIONALLY, IT INCREMENTS THE COUNT OF SAMPLES IN `N_SAMPLES_IN_CLUSTERS` FOR `CLUSTER_1` BY 1, INDICATING THE ADDITION OF THE MERGED SAMPLES.
    #     6. FINALLY, IT SETS THE COUNT OF SAMPLES IN `N_SAMPLES_IN_CLUSTERS` FOR `CLUSTER_2` TO 0, AS IT NO LONGER REPRESENTS AN INDIVIDUAL CLUSTER.
    # IN SUMMARY, THE `__MERGE_CLUSTERS__` FUNCTION FACILITATES THE MERGING OF TWO CLUSTERS BY UPDATING THE CLUSTER LABELS IN THE `LABELS` ARRAY AND ADJUSTING THE COUNT OF SAMPLES IN EACH CLUSTER IN THE `N_SAMPLES_IN_CLUSTERS` ARRAY. BY ITERATIVELY MERGING CLUSTERS BASED ON THEIR PROXIMITY, THE FUNCTION CONTRIBUTES TO THE HIERARCHICAL CLUSTERING PROCESS OF AGGLOMERATIVE CLUSTERING, ULTIMATELY LEADING TO THE FORMATION OF THE DESIRED NUMBER OF CLUSTERS.
    def __MERGE_CLUSTERS__(MERGE_IDXS):
        """THE `__MERGE_CLUSTERS__` FUNCTION IS A HELPER FUNCTION UTILIZED WITHIN THE AGGLOMERATIVE CLUSTERING ALGORITHM.
            1. CLUSTER IDENTIFICATION: THE FUNCTION TAKES THE INDICES OF TWO SAMPLES TO BE MERGED AS INPUT.
            2. CLUSTER LABEL UPDATE:
                - IT RETRIEVES THE CLUSTER LABELS OF THE TWO SAMPLES AND DESIGNATES ONE OF THEM AS THE REPRESENTATIVE CLUSTER.
                - IT ITERATES THROUGH THE LABELS OF ALL SAMPLES AND UPDATES ANY OCCURRENCES OF THE SECOND CLUSTER LABEL TO MATCH THE FIRST CLUSTER LABEL.
                - THIS UPDATE EFFECTIVELY MERGES THE SAMPLES INTO A SINGLE CLUSTER.
            3. SAMPLE COUNT ADJUSTMENT:
                - THE FUNCTION INCREMENTS THE COUNT OF SAMPLES IN THE REPRESENTATIVE CLUSTER BY THE NUMBER OF MERGED SAMPLES.
                - IT SETS THE COUNT OF SAMPLES IN THE SECOND CLUSTER TO 0 SINCE IT NO LONGER REPRESENTS AN INDIVIDUAL CLUSTER.
        THE `__MERGE_CLUSTERS__` FUNCTION PLAYS A CRUCIAL ROLE IN THE AGGLOMERATIVE CLUSTERING PROCESS BY ENABLING THE MERGING OF CLUSTERS BASED ON THEIR PROXIMITY. IT FACILITATES THE AGGREGATION OF SIMILAR SAMPLES, CONTRIBUTING TO THE FORMATION OF HIERARCHICAL STRUCTURES IN THE CLUSTERING ALGORITHM."""
        CLUSTER_1 = LABELS[MERGE_IDXS[0]]
        CLUSTER_2 = LABELS[MERGE_IDXS[1]]
        for i, _ in enumerate(LABELS):
            if LABELS[i] == CLUSTER_2:
                LABELS[i] = CLUSTER_1
                N_SAMPLES_IN_CLUSTERS[CLUSTER_1] += 1
        N_SAMPLES_IN_CLUSTERS[CLUSTER_2] = 0

    while len(np.unique(LABELS)) > K:
        MIN_DISTANCE = np.inf
        MERGE_IDXS = None
        for i in range(N_SAMPLES - 1):
            for j in range(i + 1, N_SAMPLES):
                if LABELS[i] != LABELS[j]:
                    DISTANCE = __EUCLIDEAN_DISTANCE__(X[i], X[j])
                    if DISTANCE < MIN_DISTANCE:
                        MIN_DISTANCE = DISTANCE
                        MERGE_IDXS = (i, j)
        if MERGE_IDXS is not None:
            __MERGE_CLUSTERS__(MERGE_IDXS)
    return LABELS

# THE `__EUCLIDEAN_DISTANCE__` FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS IN A MULTI-DIMENSIONAL SPACE. IT IS A COMMON DISTANCE METRIC USED IN VARIOUS MACHINE LEARNING ALGORITHMS, INCLUDING K-MEANS CLUSTERING.
#     1. THE FUNCTION TAKES TWO INPUT ARRAYS, `X_1` AND `X_2`, REPRESENTING THE COORDINATES OF TWO POINTS IN THE SAME DIMENSIONAL SPACE.
#     2. THE DIFFERENCE BETWEEN THE TWO INPUT POINTS IS COMPUTED BY SUBTRACTING `X_2` FROM `X_1`, RESULTING IN A NEW ARRAY REPRESENTING THE VECTOR BETWEEN THE TWO POINTS.
#     3. THE SQUARE OF EACH ELEMENT IN THE VECTOR IS CALCULATED USING THE `** 2` OPERATOR, EFFECTIVELY SQUARING THE DIFFERENCES ALONG EACH DIMENSION.
#     4. THE SQUARED DIFFERENCES ALONG EACH DIMENSION ARE SUMMED UP USING THE `NP.SUM` FUNCTION WITH THE `AXIS=0` ARGUMENT, RESULTING IN A SINGLE SCALAR VALUE REPRESENTING THE SUM OF SQUARED DIFFERENCES.
#     5. THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES IS TAKEN USING THE `NP.SQRT` FUNCTION, YIELDING THE EUCLIDEAN DISTANCE BETWEEN THE TWO POINTS.
#     6. THE EUCLIDEAN DISTANCE IS RETURNED AS THE OUTPUT OF THE FUNCTION.
# IN SUMMARY, THE `__EUCLIDEAN_DISTANCE__` FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS BY FINDING THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES ALONG EACH DIMENSION. IT PROVIDES A MEASURE OF SIMILARITY OR DISSIMILARITY BETWEEN POINTS IN A MULTI-DIMENSIONAL SPACE, WHICH IS USED IN VARIOUS ALGORITHMS FOR CLUSTERING, CLASSIFICATION, AND OTHER TASKS.
def __EUCLIDEAN_DISTANCE__(X_1, X_2):
    """THE `__EUCLIDEAN_DISTANCE__` FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS IN A MULTI-DIMENSIONAL SPACE. IT TAKES TWO INPUT ARRAYS REPRESENTING THE COORDINATES OF THE POINTS AND COMPUTES THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES ALONG EACH DIMENSION. THE RESULTING VALUE REPRESENTS THE DISTANCE BETWEEN THE TWO POINTS."""
    return np.sqrt(np.sum((X_1 - X_2) ** 2, axis=0))