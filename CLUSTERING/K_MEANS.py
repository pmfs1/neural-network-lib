import numpy as np

# THE `K_MEANS_CLUSTERING` FUNCTION IMPLEMENTS THE K-MEANS CLUSTERING ALGORITHM. GIVEN A SET OF DATA POINTS `X`, IT AIMS TO PARTITION THE DATA INTO `K` CLUSTERS BY ITERATIVELY ADJUSTING THE POSITIONS OF CENTROIDS.
#     1. THE `K_MEANS_CLUSTERING` FUNCTION TAKES THE INPUT DATA `X`, THE NUMBER OF CLUSTERS `K`, AND THE MAXIMUM NUMBER OF ITERATIONS `MAX_ITERATIONS` AS PARAMETERS.
#     2. INSIDE THE `K_MEANS_CLUSTERING` FUNCTION, THERE ARE SEVERAL HELPER FUNCTIONS DEFINED:
#         A. `__CREATE_CLUSTERS__` FUNCTION TAKES THE INPUT DATA `X` AND THE CURRENT CENTROIDS `CENTROIDS`. IT ASSIGNS EACH SAMPLE TO THE CLOSEST CENTROID AND RETURNS A LIST OF CLUSTERS, WHERE EACH CLUSTER IS A LIST OF INDICES OF THE SAMPLES BELONGING TO THAT CLUSTER.
#         B. `__CLOSEST_CENTROID__` FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN A SAMPLE `SAMPLE` AND EACH CENTROID IN `CENTROIDS`. IT RETURNS THE INDEX OF THE CLOSEST CENTROID.
#         C. `__GET_CENTROIDS__` FUNCTION TAKES THE LIST OF CLUSTERS AND CALCULATES THE MEAN OF EACH CLUSTER TO DETERMINE THE NEW CENTROID POSITIONS.
#         D. `__IS_CONVERGED__` FUNCTION CHECKS IF THE CENTROIDS HAVE CONVERGED BY COMPARING THE DISTANCES BETWEEN THE OLD CENTROIDS AND THE NEW CENTROIDS.
#         E. `__GET_CLUSTER_LABELS__` FUNCTION ASSIGNS CLUSTER LABELS TO EACH SAMPLE BASED ON THE INDICES IN THE LIST OF CLUSTERS.
#     3. THE CODE SETS THE RANDOM SEED TO 42 USING `NP.RANDOM.SEED(42)` FOR REPRODUCIBILITY.
#     4. THE NUMBER OF SAMPLES AND FEATURES IS EXTRACTED FROM THE SHAPE OF `X` USING `N_SAMPLES, N_FEATURES = X.SHAPE`.
#     5. INITIAL CENTROIDS ARE RANDOMLY SELECTED FROM THE SAMPLES WITHOUT REPLACEMENT USING `X[NP.RANDOM.CHOICE(N_SAMPLES, K, REPLACE=FALSE)]`.
#     6. THE CODE INITIALIZES AN EMPTY LIST OF CLUSTERS `CLUSTERS`.
#     7. THE MAIN LOOP RUNS FOR A MAXIMUM OF `MAX_ITERATIONS`.
#     8. IN EACH ITERATION, THE CODE CREATES NEW CLUSTERS BY CALLING `__CREATE_CLUSTERS__` WITH THE CURRENT DATA POINTS `X` AND CENTROIDS `CENTROIDS`.
#     9. THE NEW CENTROID POSITIONS ARE OBTAINED BY CALLING `__GET_CENTROIDS__` WITH THE NEW CLUSTERS.
#     10. THE CODE CHECKS IF THE CENTROIDS HAVE CONVERGED BY CALLING `__IS_CONVERGED__` WITH THE OLD AND NEW CENTROID POSITIONS. IF THEY HAVE CONVERGED, THE LOOP IS TERMINATED.
#     11. IF THE CENTROIDS HAVE NOT CONVERGED, THE NEW CLUSTERS BECOME THE CURRENT CLUSTERS, AND THE NEW CENTROID POSITIONS BECOME THE CURRENT CENTROIDS FOR THE NEXT ITERATION.
#     12. FINALLY, THE CLUSTER LABELS ARE OBTAINED BY CALLING `__GET_CLUSTER_LABELS__` WITH THE FINAL CLUSTERS.
#     13. THE FUNCTION RETURNS THE CLUSTER LABELS.
# IN SUMMARY, THIS CODE IMPLEMENTS THE K-MEANS CLUSTERING ALGORITHM BY ITERATIVELY ASSIGNING SAMPLES TO CLUSTERS, UPDATING CENTROID POSITIONS, AND CHECKING FOR CONVERGENCE. THE PROCESS CONTINUES UNTIL CONVERGENCE, OR THE MAXIMUM NUMBER OF ITERATIONS IS REACHED. THE FUNCTION RETURNS THE CLUSTER LABELS FOR EACH SAMPLE IN THE INPUT DATA.
def K_MEANS_CLUSTERING(X, K=3, MAX_ITERATIONS=100):
    """THE `K_MEANS_CLUSTERING` FUNCTION IMPLEMENTS THE K-MEANS CLUSTERING ALGORITHM. GIVEN A SET OF DATA POINTS `X`, IT AIMS TO PARTITION THE DATA INTO `K` CLUSTERS BY ITERATIVELY ADJUSTING THE POSITIONS OF CENTROIDS. HERE'S A BRIEF DESCRIPTION OF THE FUNCTION:
        1. RANDOMLY INITIALIZE `K` CENTROIDS FROM THE DATA POINTS.
        2. CREATE EMPTY CLUSTERS TO STORE THE INDICES OF SAMPLES BELONGING TO EACH CLUSTER.
        3. ITERATE UNTIL CONVERGENCE OR REACHING THE MAXIMUM NUMBER OF ITERATIONS:
            - ASSIGN EACH SAMPLE TO THE CLOSEST CENTROID, FORMING NEW CLUSTERS.
            - UPDATE THE CENTROID POSITIONS BY CALCULATING THE MEAN OF EACH CLUSTER.
            - CHECK FOR CONVERGENCE BY COMPARING THE DISTANCES BETWEEN OLD AND NEW CENTROIDS.
            - IF CONVERGED, EXIT THE LOOP.
            - OTHERWISE, UPDATE THE CURRENT CLUSTERS AND CENTROIDS FOR THE NEXT ITERATION.
        4. ASSIGN CLUSTER LABELS TO EACH SAMPLE BASED ON THE FINAL CLUSTERS.
        5. RETURN THE CLUSTER LABELS.
    THE K-MEANS ALGORITHM AIMS TO MINIMIZE THE WITHIN-CLUSTER SUM OF SQUARED DISTANCES, EFFECTIVELY GROUPING SIMILAR DATA POINTS TOGETHER. IT ITERATIVELY REFINES THE CLUSTER ASSIGNMENTS UNTIL CONVERGENCE, ENSURING THAT THE CENTROIDS REPRESENT THE CENTRES OF THEIR RESPECTIVE CLUSTERS. THE RESULT IS A SET OF CLUSTER LABELS THAT INDICATE WHICH CLUSTER EACH DATA POINT BELONGS TO."""

    # THE `__CREATE_CLUSTERS__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM. ITS PURPOSE IS TO ASSIGN DATA POINTS TO CLUSTERS BASED ON THEIR PROXIMITY TO CENTROIDS.
    #     1. THE FUNCTION `__CREATE_CLUSTERS__` TAKES TWO ARGUMENTS: `X` REPRESENTS THE DATA POINTS, AND `CENTROIDS` CONTAINS THE POSITIONS OF THE CENTROIDS.
    #     2. IT INITIALIZES AN EMPTY LIST CALLED `CLUSTERS` USING A LIST COMPREHENSION. THE LENGTH OF `CLUSTERS` IS EQUAL TO THE NUMBER OF CENTROIDS, WHICH MEANS THERE WILL BE ONE CLUSTER FOR EACH CENTROID.
    #     3. IT ITERATES OVER EACH DATA POINT IN `X` USING THE `ENUMERATE` FUNCTION, WHICH PROVIDES BOTH THE INDEX (`IDX`) AND THE CORRESPONDING DATA POINT (`SAMPLE`).
    #     4. INSIDE THE LOOP, IT CALLS THE `__CLOSEST_CENTROID__` HELPER FUNCTION TO DETERMINE THE INDEX OF THE CLOSEST CENTROID TO THE CURRENT DATA POINT. THIS FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN THE DATA POINT AND EACH CENTROID AND RETURNS THE INDEX OF THE CENTROID WITH THE SMALLEST DISTANCE.
    #     5. IT APPENDS THE INDEX (`IDX`) OF THE CURRENT DATA POINT TO THE APPROPRIATE CLUSTER IN THE `CLUSTERS` LIST. THE INDEX OF THE CLOSEST CENTROID (`CLOSEST_CENTROID_IDX`) DETERMINES WHICH CLUSTER THE DATA POINT BELONGS TO.
    #     6. AFTER ITERATING OVER ALL THE DATA POINTS, IT RETURNS THE `CLUSTERS` LIST, WHERE EACH ELEMENT REPRESENTS A CLUSTER AND CONTAINS THE INDICES OF THE DATA POINTS ASSIGNED TO THAT CLUSTER.
    # IN SUMMARY, THE `__CREATE_CLUSTERS__` FUNCTION ASSIGNS EACH DATA POINT IN `X` TO THE NEAREST CENTROID IN `CENTROIDS` AND ORGANIZES THE DATA POINTS INTO CLUSTERS BASED ON THESE ASSIGNMENTS. THE RESULTING `CLUSTERS` LIST IS A LIST OF LISTS, WHERE EACH INNER LIST REPRESENTS A CLUSTER AND CONTAINS THE INDICES OF THE DATA POINTS ASSIGNED TO THAT CLUSTER. THIS FUNCTION IS A KEY STEP IN THE K-MEANS CLUSTERING ALGORITHM, AS IT DETERMINES THE INITIAL ASSIGNMENT OF DATA POINTS TO CLUSTERS AND IS USED TO UPDATE THE CENTROIDS IN EACH ITERATION.
    def __CREATE_CLUSTERS__(X, CENTROIDS):
        """THE `__CREATE_CLUSTERS__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE `K_MEANS_CLUSTERING` FUNCTION TO ASSIGN DATA POINTS TO THEIR NEAREST CENTROIDS AND CREATE CLUSTERS BASED ON THESE ASSIGNMENTS. HERE'S AN EXPLANATION OF THE FUNCTION:
            1. IT TAKES TWO INPUTS: `X`, WHICH REPRESENTS THE DATA POINTS, AND `CENTROIDS`, WHICH CONTAINS THE POSITIONS OF THE CENTROIDS.
            2. IT INITIALIZES AN EMPTY LIST CALLED `CLUSTERS` TO STORE THE CLUSTERS. THE LENGTH OF `CLUSTERS` IS EQUAL TO THE NUMBER OF CENTROIDS.
            3. IT ITERATES OVER EACH DATA POINT IN `X` AND FINDS THE CLOSEST CENTROID USING THE `__CLOSEST_CENTROID__` HELPER FUNCTION.
            4. IT APPENDS THE INDEX OF THE CURRENT DATA POINT TO THE CORRESPONDING CLUSTER IN `CLUSTERS` BASED ON THE INDEX OF THE CLOSEST CENTROID.
            5. AFTER ITERATING OVER ALL DATA POINTS, IT RETURNS THE `CLUSTERS` LIST, WHERE EACH ELEMENT REPRESENTS A CLUSTER AND CONTAINS THE INDICES OF THE DATA POINTS ASSIGNED TO THAT CLUSTER.
        IN SUMMARY, THE `__CREATE_CLUSTERS__` FUNCTION ASSIGNS DATA POINTS TO THEIR NEAREST CENTROIDS AND ORGANIZES THEM INTO CLUSTERS BASED ON THESE ASSIGNMENTS. IT SERVES AS A CRUCIAL STEP IN THE K-MEANS ALGORITHM, ALLOWING THE ALGORITHM TO UPDATE THE CENTROID POSITIONS AND REFINE THE CLUSTERS IN SUBSEQUENT ITERATIONS."""
        CLUSTERS = [[] for _ in range(len(CENTROIDS))]
        for IDX, SAMPLE in enumerate(X):
            CLOSEST_CENTROID_IDX = __CLOSEST_CENTROID__(SAMPLE, CENTROIDS)
            CLUSTERS[CLOSEST_CENTROID_IDX].append(IDX)        
        return CLUSTERS

    # THE `__CLOSEST_CENTROID__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM. ITS PURPOSE IS TO FIND THE INDEX OF THE CLOSEST CENTROID TO A GIVEN DATA POINT. LET'S GO THROUGH ITS IMPLEMENTATION AND BEHAVIOR IN DETAIL:
    #     1. THE FUNCTION TAKES TWO ARGUMENTS: `SAMPLE`, REPRESENTING THE DATA POINT FOR WHICH WE WANT TO FIND THE CLOSEST CENTROID, AND `CENTROIDS`, WHICH IS AN ARRAY CONTAINING THE POSITIONS OF ALL THE CENTROIDS.
    #     2. IT INITIALIZES AN EMPTY LIST CALLED `DISTANCES`, WHICH WILL STORE THE EUCLIDEAN DISTANCES BETWEEN THE `SAMPLE` AND EACH CENTROID.
    #     3. THE FUNCTION THEN ITERATES OVER EACH CENTROID IN `CENTROIDS`. FOR EACH CENTROID, IT CALCULATES THE EUCLIDEAN DISTANCE TO THE `SAMPLE` USING THE `__EUCLIDEAN_DISTANCE__` HELPER FUNCTION.
    #     4. THE `__EUCLIDEAN_DISTANCE__` HELPER FUNCTION TAKES TWO DATA POINTS (IN THIS CASE, THE `SAMPLE` AND A CENTROID) AND COMPUTES THE EUCLIDEAN DISTANCE BETWEEN THEM. EUCLIDEAN DISTANCE IS A MEASURE OF THE STRAIGHT-LINE DISTANCE BETWEEN TWO POINTS IN A MULTI-DIMENSIONAL SPACE. IT IS COMPUTED AS THE SQUARE ROOT OF THE SUM OF THE SQUARED DIFFERENCES BETWEEN THE COORDINATES OF THE TWO POINTS.
    #     5. THE CALCULATED EUCLIDEAN DISTANCE FOR EACH CENTROID IS APPENDED TO THE `DISTANCES` LIST.
    #     6. AFTER CALCULATING THE DISTANCES TO ALL CENTROIDS, THE FUNCTION USES `NP.ARGMIN(DISTANCES)` TO FIND THE INDEX OF THE CENTROID WITH THE SMALLEST DISTANCE. THIS INDEX CORRESPONDS TO THE INDEX OF THE CLOSEST CENTROID IN THE `CENTROIDS` ARRAY.
    #     7. THE FUNCTION RETURNS THE INDEX OF THE CLOSEST CENTROID.
    # IN SUMMARY, THE `__CLOSEST_CENTROID__` FUNCTION COMPUTES THE EUCLIDEAN DISTANCES BETWEEN A GIVEN DATA POINT (`SAMPLE`) AND ALL CENTROIDS IN THE `CENTROIDS` ARRAY. IT THEN RETURNS THE INDEX OF THE CENTROID WITH THE SMALLEST DISTANCE, INDICATING WHICH CENTROID IS CLOSEST TO THE DATA POINT. THIS INFORMATION IS CRUCIAL IN THE K-MEANS ALGORITHM, AS IT DETERMINES THE CLUSTER ASSIGNMENT OF THE DATA POINT DURING THE CLUSTERING PROCESS.
    def __CLOSEST_CENTROID__(SAMPLE, CENTROIDS):
        """THE `__CLOSEST_CENTROID__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM. IT CALCULATES THE EUCLIDEAN DISTANCES BETWEEN A GIVEN DATA POINT AND ALL CENTROIDS AND RETURNS THE INDEX OF THE CENTROID WITH THE SMALLEST DISTANCE. THIS INDEX INDICATES WHICH CENTROID IS CLOSEST TO THE DATA POINT AND IS USED TO ASSIGN THE DATA POINT TO A CLUSTER DURING THE CLUSTERING PROCESS."""
        DISTANCES = [__EUCLIDEAN_DISTANCE__(SAMPLE, CENTROID) for CENTROID in CENTROIDS]
        CLOSEST_IDX = np.argmin(DISTANCES)
        return CLOSEST_IDX

    # THE `__GET_CENTROIDS__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM. IT TAKES A LIST OF CLUSTERS AS INPUT AND CALCULATES THE CENTROID FOR EACH CLUSTER.
    #     1. IT INITIALIZES AN ARRAY `CENTROIDS` OF ZEROS WITH DIMENSIONS `(LEN(CLUSTERS), LEN(CLUSTERS[0][0]))`. THIS ARRAY WILL STORE THE CENTROIDS FOR EACH CLUSTER.
    #         - `LEN(CLUSTERS)` REPRESENTS THE NUMBER OF CLUSTERS, AND `LEN(CLUSTERS[0][0])` REPRESENTS THE NUMBER OF FEATURES IN EACH DATA POINT.
    #     2. IT ITERATES OVER EACH CLUSTER IN THE LIST OF CLUSTERS USING THE `ENUMERATE` FUNCTION.
    #         - `IDX` IS THE INDEX OF THE CURRENT CLUSTER BEING PROCESSED, AND `CLUSTER` IS THE LIST OF DATA POINT INDICES BELONGING TO THAT CLUSTER.
    #     3. FOR EACH CLUSTER, IT CALCULATES THE MEAN OF THE DATA POINTS IN THAT CLUSTER ALONG THE FIRST AXIS (AXIS 0).
    #        - `NP.MEAN(CLUSTER, AXIS=0)` CALCULATES THE MEAN VALUE FOR EACH FEATURE ACROSS THE DATA POINTS IN THE CLUSTER.
    #     4. IT ASSIGNS THE CALCULATED MEAN AS THE CENTROID FOR THE CURRENT CLUSTER BY UPDATING THE CORRESPONDING ROW IN THE `CENTROIDS` ARRAY.
    #        - `CENTROIDS[IDX] = CLUSTER_MEAN` ASSIGNS THE CALCULATED MEAN TO THE `IDX`-TH ROW OF THE `CENTROIDS` ARRAY.
    #     5. AFTER ITERATING OVER ALL CLUSTERS, IT RETURNS THE `CENTROIDS` ARRAY CONTAINING THE CENTROIDS FOR EACH CLUSTER.
    # IN SUMMARY, THE `__GET_CENTROIDS__` FUNCTION CALCULATES THE CENTROID FOR EACH CLUSTER BY TAKING THE MEAN OF THE DATA POINTS IN THE CLUSTER ALONG THE FIRST AXIS AND RETURNS AN ARRAY OF CENTROIDS. THESE CENTROIDS REPRESENT THE CENTER POINTS OF THE CLUSTERS AND ARE USED IN THE K-MEANS ALGORITHM FOR ASSIGNING DATA POINTS TO THEIR NEAREST CLUSTERS.
    def __GET_CENTROIDS__(CLUSTERS):
        """THE `__GET_CENTROIDS__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM. IT TAKES A LIST OF CLUSTERS AS INPUT AND CALCULATES THE CENTROID FOR EACH CLUSTER. THE CENTROID IS THE MEAN VALUE OF THE DATA POINTS IN THE CLUSTER.
            1. INITIALIZE AN ARRAY `CENTROIDS` OF ZEROS WITH DIMENSIONS `(LEN(CLUSTERS), LEN(CLUSTERS[0][0]))`, WHERE `LEN(CLUSTERS)` REPRESENTS THE NUMBER OF CLUSTERS AND `LEN(CLUSTERS[0][0])` REPRESENTS THE NUMBER OF FEATURES IN EACH DATA POINT.
            2. ITERATE OVER EACH CLUSTER IN THE LIST OF CLUSTERS.
            3. FOR EACH CLUSTER, CALCULATE THE MEAN OF THE DATA POINTS IN THAT CLUSTER ALONG THE FIRST AXIS USING `NP.MEAN(CLUSTER, AXIS=0)`. THIS GIVES THE CENTROID OF THE CLUSTER.
            4. ASSIGN THE CALCULATED CENTROID AS THE VALUE FOR THE CORRESPONDING CLUSTER BY UPDATING THE `CENTROIDS` ARRAY.
            5. RETURN THE `CENTROIDS` ARRAY CONTAINING THE CENTROIDS FOR EACH CLUSTER.
        IN SUMMARY, THE `__GET_CENTROIDS__` FUNCTION CALCULATES THE CENTROID FOR EACH CLUSTER BY TAKING THE MEAN OF THE DATA POINTS IN THE CLUSTER ALONG THE FIRST AXIS. IT PROVIDES A WAY TO COMPUTE THE CENTRE POINT OF EACH CLUSTER, WHICH IS IMPORTANT FOR THE K-MEANS ALGORITHM IN ASSIGNING DATA POINTS TO THEIR NEAREST CLUSTERS."""
        CENTROIDS = np.zeros((len(CLUSTERS), len(CLUSTERS[0][0])))
        for IDX, CLUSTER in enumerate(CLUSTERS):
            CLUSTER_MEAN = np.mean(CLUSTER, axis=0)
            CENTROIDS[IDX] = CLUSTER_MEAN
        return CENTROIDS

    # THE `__IS_CONVERGED__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM TO DETERMINE IF THE ALGORITHM HAS CONVERGED. IT COMPARES THE OLD CENTROIDS WITH THE NEW CENTROIDS AND CHECKS IF THEY ARE CLOSE ENOUGH, INDICATING THAT THE ALGORITHM HAS REACHED A STABLE SOLUTION.
    #     1. THE FUNCTION TAKES TWO INPUT PARAMETERS: `OLD_CENTROIDS` AND `NEW_CENTROIDS`, WHICH REPRESENT THE CENTROIDS FROM THE PREVIOUS ITERATION AND THE UPDATED CENTROIDS FROM THE CURRENT ITERATION, RESPECTIVELY.
    #     2. CREATE AN EMPTY LIST `DISTANCES` TO STORE THE DISTANCES BETWEEN THE OLD AND NEW CENTROIDS FOR EACH CLUSTER.
    #     3. ITERATE OVER THE RANGE OF THE NUMBER OF CENTROIDS, WHICH IS THE LENGTH OF `OLD_CENTROIDS` OR `NEW_CENTROIDS`.
    #     4. FOR EACH CENTROID INDEX `IDX`, CALCULATE THE DISTANCE BETWEEN THE OLD AND NEW CENTROIDS USING THE `__EUCLIDEAN_DISTANCE__` FUNCTION. THIS FUNCTION COMPUTES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS.
    #     5. APPEND THE CALCULATED DISTANCE TO THE `DISTANCES` LIST.
    #     6. FINALLY, CHECK IF THE SUM OF ALL DISTANCES IN THE `DISTANCES` LIST IS EQUAL TO 0. IF THE SUM IS 0, IT MEANS THAT ALL THE DISTANCES BETWEEN THE OLD AND NEW CENTROIDS ARE CLOSE TO ZERO, INDICATING THAT THE ALGORITHM HAS CONVERGED. RETURN `TRUE` IN THIS CASE.
    #     7. IF THE SUM OF DISTANCES IS NOT EQUAL TO 0, IT MEANS THAT THE ALGORITHM HAS NOT YET CONVERGED. RETURN `FALSE` IN THIS CASE.
    # IN SUMMARY, THE `__IS_CONVERGED__` FUNCTION COMPARES THE DISTANCES BETWEEN THE OLD AND NEW CENTROIDS FOR EACH CLUSTER AND DETERMINES IF THE ALGORITHM HAS CONVERGED. IF THE DISTANCES ARE CLOSE TO ZERO, IT INDICATES CONVERGENCE, AND THE FUNCTION RETURNS `TRUE`. OTHERWISE, IT RETURNS `FALSE`, INDICATING THAT THE ALGORITHM NEEDS TO CONTINUE ITERATING TO FIND A STABLE SOLUTION.
    def __IS_CONVERGED__(OLD_CENTROIDS, NEW_CENTROIDS):
        """THE `__IS_CONVERGED__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM TO CHECK IF THE CENTROIDS HAVE CONVERGED, INDICATING THAT THE ALGORITHM HAS REACHED A STABLE SOLUTION. IT COMPARES THE DISTANCES BETWEEN THE OLD AND NEW CENTROIDS FOR EACH CLUSTER AND RETURNS `TRUE` IF THE DISTANCES ARE CLOSE TO ZERO, INDICATING CONVERGENCE. OTHERWISE, IT RETURNS `FALSE`, INDICATING THAT THE ALGORITHM SHOULD CONTINUE ITERATING TO FIND A STABLE SOLUTION."""
        DISTANCES = [__EUCLIDEAN_DISTANCE__(OLD_CENTROIDS[IDX], NEW_CENTROIDS[IDX]) for IDX in range(len(OLD_CENTROIDS))]
        return sum(DISTANCES) == 0

    # THE `__GET_CLUSTER_LABELS__` FUNCTION IS A HELPER FUNCTION USED IN THE K-MEANS CLUSTERING ALGORITHM TO ASSIGN CLUSTER LABELS TO EACH DATA SAMPLE IN THE DATASET BASED ON THEIR CLOSEST CENTROIDS. IT TAKES THE LIST OF CLUSTERS, WHERE EACH CLUSTER IS REPRESENTED AS A LIST OF SAMPLE INDICES, AND RETURNS A NUMPY ARRAY OF CLUSTER LABELS FOR EACH DATA SAMPLE IN THE DATASET.
    #     1. `LABELS = NP.EMPTY(SUM(LEN(_C) FOR _C IN CLUSTERS))`: THIS LINE INITIALIZES AN EMPTY NUMPY ARRAY TO STORE THE CLUSTER LABELS FOR ALL DATA SAMPLES. THE SIZE OF THIS ARRAY IS EQUAL TO THE TOTAL NUMBER OF DATA SAMPLES IN THE DATASET.
    #     2. `FOR IDX, CLUSTER IN ENUMERATE(CLUSTERS):`: THIS LINE INITIATES A LOOP THAT ITERATES OVER EACH CLUSTER IN THE LIST OF CLUSTERS.
    #     3. `FOR SAMPLE_IDX IN CLUSTER:`: THIS NESTED LOOP ITERATES OVER EACH SAMPLE INDEX IN THE CURRENT CLUSTER.
    #     4. `LABELS[SAMPLE_IDX] = IDX`: FOR EACH SAMPLE INDEX IN THE CURRENT CLUSTER, IT ASSIGNS THE CLUSTER INDEX (`IDX`) AS THE CLUSTER LABEL FOR THAT SAMPLE IN THE `LABELS` ARRAY. SINCE THE `LABELS` ARRAY IS ALIGNED WITH THE ORIGINAL DATASET, THE CLUSTER LABEL IS ASSIGNED TO THE CORRESPONDING DATA SAMPLE.
    #     5. AFTER PROCESSING ALL CLUSTERS AND THEIR SAMPLES, THE FUNCTION RETURNS THE `LABELS` ARRAY CONTAINING THE CLUSTER LABELS FOR EACH DATA SAMPLE IN THE DATASET.
    # IN SUMMARY, THE `__GET_CLUSTER_LABELS__` FUNCTION TAKES A LIST OF CLUSTERS, WHERE EACH CLUSTER CONTAINS THE INDICES OF SAMPLES BELONGING TO THAT CLUSTER, AND RETURNS A NUMPY ARRAY WHERE EACH ELEMENT REPRESENTS THE CLUSTER LABEL FOR THE CORRESPONDING DATA SAMPLE IN THE DATASET. THIS FUNCTION IS AN ESSENTIAL STEP IN THE K-MEANS ALGORITHM TO ASSOCIATE EACH DATA SAMPLE WITH ITS CORRESPONDING CLUSTER AFTER THE CENTROIDS HAVE BEEN DETERMINED.
    def __GET_CLUSTER_LABELS__(CLUSTERS):
        """THE `__GET_CLUSTER_LABELS__` FUNCTION IN THE K-MEANS CLUSTERING ALGORITHM IS RESPONSIBLE FOR ASSIGNING CLUSTER LABELS TO EACH DATA SAMPLE IN THE DATASET BASED ON THEIR CLOSEST CENTROIDS. IT TAKES A LIST OF CLUSTERS, WHERE EACH CLUSTER IS REPRESENTED AS A LIST OF SAMPLES' INDICES AND RETURNS A NUMPY ARRAY OF CLUSTER LABELS FOR EACH DATA SAMPLE.
            1. INITIALIZE AN EMPTY NUMPY ARRAY `LABELS` WITH A SIZE EQUAL TO THE TOTAL NUMBER OF DATA SAMPLES.
            2. ITERATE OVER EACH CLUSTER IN THE LIST OF CLUSTERS.
            3. FOR EACH CLUSTER, ITERATE OVER THE SAMPLE INDICES WITHIN THAT CLUSTER.
            4. ASSIGN THE CLUSTER INDEX AS THE LABEL FOR EACH DATA SAMPLE IN THE `LABELS` ARRAY. THE CLUSTER INDEX REPRESENTS THE CLOSEST CENTROID FOR THAT SAMPLE.
            5. AFTER PROCESSING ALL CLUSTERS AND THEIR SAMPLES, RETURN THE `LABELS` ARRAY CONTAINING THE CLUSTER LABELS FOR EACH DATA SAMPLE.
        IN SUMMARY, THE `__GET_CLUSTER_LABELS__` FUNCTION ASSIGNS CLUSTER LABELS TO EACH DATA SAMPLE BASED ON THEIR CLOSEST CENTROIDS. IT ENABLES THE ASSOCIATION OF DATA SAMPLES WITH THEIR RESPECTIVE CLUSTERS, PROVIDING VALUABLE INFORMATION ABOUT THE CLUSTER MEMBERSHIP OF EACH SAMPLE IN THE DATASET."""
        LABELS = np.empty(sum(len(_C) for _C in CLUSTERS))        
        for IDX, CLUSTER in enumerate(CLUSTERS):
            for SAMPLE_IDX in CLUSTER:
                LABELS[SAMPLE_IDX] = IDX
        return LABELS

    np.random.seed(42)
    N_SAMPLES, _ = X.shape
    CENTROIDS = X[np.random.choice(N_SAMPLES, K, replace=False)]
    CLUSTERS = [[] for _ in range(K)]
    for _ in range(MAX_ITERATIONS):
        NEW_CLUSTERS = __CREATE_CLUSTERS__(X, CENTROIDS)
        NEW_CENTROIDS = __GET_CENTROIDS__(NEW_CLUSTERS)
        if __IS_CONVERGED__(CENTROIDS, NEW_CENTROIDS):
            break
        CLUSTERS = NEW_CLUSTERS
        CENTROIDS = NEW_CENTROIDS
    LABELS = __GET_CLUSTER_LABELS__(CLUSTERS)
    return LABELS

# IMPLEMENTS THE BISECTING K-MEANS CLUSTERING ALGORITHM. THE `BISECTING_K_MEANS_CLUSTERING` FUNCTION TAKES THREE INPUT PARAMETERS:
#     - `X`: THE INPUT DATA MATRIX REPRESENTED AS A NUMPY ARRAY, WHERE EACH ROW REPRESENTS A DATA POINT AND EACH COLUMN REPRESENTS A FEATURE.
#     - `K`: THE DESIRED NUMBER OF CLUSTERS (DEFAULT IS SET TO 3).
#     - `MAX_ITERATIONS`: THE MAXIMUM NUMBER OF ITERATIONS FOR THE K-MEANS CLUSTERING ALGORITHM (DEFAULT IS SET TO 100).
# THE FUNCTION FIRST DEFINES AN INNER HELPER FUNCTION CALLED `__EUCLIDEAN_DISTANCE__`. THIS FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS IN A MULTI-DIMENSIONAL SPACE.
#     1. INITIALIZE `CLUSTERS` AS A LIST CONTAINING A SINGLE CLUSTER, WHICH INITIALLY INCLUDES ALL THE DATA POINTS.
#     2. WHILE THE NUMBER OF CLUSTERS IN `CLUSTERS` IS LESS THAN THE DESIRED `K` CLUSTERS, CONTINUE THE LOOP:
#         - COMPUTE THE SUM OF SQUARED ERRORS (SSE) VALUES FOR EACH CLUSTER IN `CLUSTERS`. FOR EACH CLUSTER:
#             - CALCULATE THE CENTROID BY TAKING THE MEAN OF THE DATA POINTS IN THE CLUSTER ALONG EACH FEATURE DIMENSION.
#             - CALCULATE THE SSE BY SUMMING THE SQUARED EUCLIDEAN DISTANCES BETWEEN EACH DATA POINT IN THE CLUSTER AND THE CENTROID.
#             - APPEND THE SSE VALUE TO THE `SSE_VALUES` LIST.
#         - SELECT THE CLUSTER WITH THE HIGHEST SSE VALUE TO SPLIT. THIS IS DONE BY FINDING THE INDEX OF THE MAXIMUM VALUE IN `SSE_VALUES` AND RETRIEVING THE CORRESPONDING CLUSTER FROM `CLUSTERS`.
#         - REMOVE THE SELECTED CLUSTER FROM `CLUSTERS`.
#         - APPLY THE K-MEANS CLUSTERING ALGORITHM (USING THE `K_MEANS_CLUSTERING` FUNCTION) TO THE SELECTED CLUSTER, SPLITTING IT INTO TWO NEW CLUSTERS.
#         - EXTEND `CLUSTERS` WITH THE RESULTING SPLIT CLUSTERS.
#     3. ONCE THE LOOP IS COMPLETED, THE FUNCTION RETURNS THE LIST OF CLUSTERS STORED IN `CLUSTERS`.
# OVERALL, THE `BISECTING_K_MEANS_CLUSTERING` FUNCTION ITERATIVELY APPLIES THE K-MEANS ALGORITHM TO CLUSTERS WITH THE HIGHEST SSE VALUE UNTIL THE DESIRED NUMBER OF CLUSTERS (`K`) IS REACHED. THIS APPROACH AIMS TO SPLIT THE CLUSTERS THAT EXHIBIT HIGHER VARIANCE OR DISPERSION, THEREBY POTENTIALLY CAPTURING MORE DISTINCT SUBGROUPS IN THE DATA.
def BISECTING_K_MEANS_CLUSTERING(X, K=3, MAX_ITERATIONS=100):
    """THE `BISECTING_K_MEANS_CLUSTERING` FUNCTION IS AN IMPLEMENTATION OF THE BISECTING K-MEANS CLUSTERING ALGORITHM. THIS ALGORITHM IS AN EXTENSION OF THE TRADITIONAL K-MEANS CLUSTERING ALGORITHM AND AIMS TO DIVIDE CLUSTERS INTO SMALLER SUBCLUSTERS ITERATIVELY.
        1. THE FUNCTION TAKES THE INPUT DATA `DATA`, THE DESIRED NUMBER OF CLUSTERS `K`, AND AN OPTIONAL PARAMETER `MAX_ITERATIONS` THAT SPECIFIES THE MAXIMUM NUMBER OF ITERATIONS FOR THE K-MEANS ALGORITHM.
        2. THE FUNCTION STARTS BY INITIALIZING A SINGLE CLUSTER CONTAINING ALL DATA POINTS.
        3. IT ENTERS A LOOP THAT CONTINUES UNTIL THE DESIRED NUMBER OF CLUSTERS IS REACHED.
        4. FOR EACH ITERATION, THE ALGORITHM SELECTS THE CLUSTER WITH THE LARGEST SSE (SUM OF SQUARED ERRORS) AS THE CANDIDATE FOR SPLITTING.
        5. THE SSE VALUES ARE CALCULATED FOR EACH CLUSTER, WHERE SSE REPRESENTS THE SUM OF SQUARED DISTANCES BETWEEN EACH DATA POINT IN THE CLUSTER AND ITS CENTROID.
        6. THE CLUSTER WITH THE LARGEST SSE IS CHOSEN FOR SPLITTING. IT IS REMOVED FROM THE LIST OF CLUSTERS.
        7. THE SELECTED CLUSTER IS SPLIT INTO TWO NEW CLUSTERS USING THE K-MEANS CLUSTERING ALGORITHM. THE K-MEANS ALGORITHM IS APPLIED WITH `K=2` TO SPLIT THE CLUSTER.
        8. THE RESULTING CLUSTERS FROM THE SPLIT ARE ADDED TO THE LIST OF CLUSTERS.
        9. STEPS 4-8 ARE REPEATED UNTIL THE DESIRED NUMBER OF CLUSTERS IS OBTAINED.
        10. FINALLY, THE FUNCTION RETURNS A LIST OF CLUSTERS, WHERE EACH CLUSTER IS REPRESENTED AS A COLLECTION OF DATA POINTS.
    THE BISECTING K-MEANS CLUSTERING ALGORITHM ITERATIVELY DIVIDES CLUSTERS BASED ON THEIR SSE VALUES, AIMING TO IDENTIFY MORE DISTINCT SUBCLUSTERS WITHIN THE DATA. BY SPLITTING CLUSTERS, THE ALGORITHM CAN CAPTURE FINER-GRAINED STRUCTURES IN THE DATA AND POTENTIALLY IMPROVE THE OVERALL CLUSTERING PERFORMANCE."""
    CLUSTERS = [X]
    while len(CLUSTERS) < K:
        SSE_VALUES = []
        for CLUSTER in CLUSTERS:
            CENTROID = np.mean(CLUSTER, axis=0)
            SSE = np.sum([__EUCLIDEAN_DISTANCE__(POINT, CENTROID) ** 2 for POINT in CLUSTER])
            SSE_VALUES.append(SSE)
        CLUSTER_TO_SPLIT = CLUSTERS[np.argmax(SSE_VALUES)]
        CLUSTERS.remove(CLUSTER_TO_SPLIT)
        SPLITTED_CLUSTERS = K_MEANS_CLUSTERING(CLUSTER_TO_SPLIT, 2, MAX_ITERATIONS)
        CLUSTERS.extend(SPLITTED_CLUSTERS)
    return CLUSTERS

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