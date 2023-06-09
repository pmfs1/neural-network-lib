import numpy as np

# THE `MEAN_SHIFT` FUNCTION IMPLEMENTS THE MEAN SHIFT CLUSTERING ALGORITHM. IT TAKES THREE MAIN PARAMETERS AS INPUT: `X`, REPRESENTING THE INPUT DATA POINTS, `BANDWIDTH`, WHICH CONTROLS THE SIZE OF THE NEIGHBOURHOOD, AND `MAX_ITERATIONS`, THE MAXIMUM NUMBER OF ITERATIONS THE ALGORITHM WILL PERFORM.
#     1. INITIALIZATION: THE FUNCTION STARTS BY CREATING A COPY OF THE INPUT DATA POINTS AND ASSIGNS IT TO THE `CENTROIDS` VARIABLE. THE CENTROIDS WILL BE ITERATIVELY UPDATED TO FIND THE FINAL CLUSTERS.
#     2. ITERATIVE MEAN SHIFT: THE FUNCTION ENTERS A LOOP THAT WILL ITERATE A MAXIMUM OF `MAX_ITERATIONS` TIMES. THIS LOOP PERFORMS THE ITERATIVE MEAN SHIFT PROCEDURE TO FIND THE CLUSTERS.
#     3. UPDATING CENTROIDS:
#         - WITHIN EACH ITERATION OF THE OUTER LOOP, A NEW LIST CALLED `NEW_CENTROIDS` IS CREATED TO STORE THE UPDATED CENTROIDS FOR THE CURRENT ITERATION.
#         - THE FUNCTION THEN ITERATES OVER EACH CENTROID IN THE `CENTROIDS` LIST.
#     4. NEIGHBOURHOOD CALCULATION:
#         - FOR EACH CENTROID, A NEW LIST CALLED `WITHIN_BANDWIDTH` IS CREATED TO STORE THE DATA POINTS WITHIN THE SPECIFIED `BANDWIDTH` DISTANCE FROM THE CENTROID.
#         - THE FUNCTION ITERATES OVER EACH DATA POINT `_X` IN THE ORIGINAL INPUT DATA `X` AND CHECKS IF THE EUCLIDEAN DISTANCE BETWEEN `_X` AND THE CURRENT CENTROID (`CENTROID`) IS LESS THAN OR EQUAL TO THE `BANDWIDTH` VALUE. IF IT IS, THE DATA POINT `_X` IS CONSIDERED WITHIN THE BANDWIDTH AND IS APPENDED TO THE `WITHIN_BANDWIDTH` LIST.
#     5. SHIFTING CENTROID:
#         - IF THERE ARE DATA POINTS WITHIN THE BANDWIDTH FOR THE CURRENT CENTROID, THE FUNCTION CALCULATES A NEW CENTROID (`NEW_CENTROID`) BY TAKING THE MEAN OF THE DATA POINTS WITHIN THE BANDWIDTH ALONG EACH DIMENSION. THIS NEW CENTROID REPRESENTS THE UPDATED POSITION OF THE ORIGINAL CENTROID.
#         - THE NEW CENTROID IS APPENDED TO THE `NEW_CENTROIDS` LIST.
#     6. CONVERGENCE CHECK: AFTER UPDATING ALL THE CENTROIDS, THE FUNCTION CHECKS FOR CONVERGENCE. IT COMPARES THE LENGTH OF THE `NEW_CENTROIDS` LIST WITH THE LENGTH OF THE `CENTROIDS` LIST FROM THE PREVIOUS ITERATION. IF THEY ARE EQUAL, IT FURTHER CHECKS IF ALL THE CENTROIDS IN THE `CENTROIDS` LIST AND THE `NEW_CENTROIDS` LIST ARE CLOSE (WITHIN A SMALL TOLERANCE) USING `NP.ALLCLOSE()`. IF THE CONDITIONS ARE SATISFIED, INDICATING CONVERGENCE, THE LOOP IS TERMINATED, AND THE FINAL CENTROIDS ARE RETURNED.
#     7. FINALIZING CLUSTERS: IF THE LOOP COMPLETES WITHOUT CONVERGING WITHIN THE MAXIMUM NUMBER OF ITERATIONS, THE FUNCTION RETURNS THE LATEST CENTROIDS FOUND, WHICH REPRESENT THE CLUSTERS.
# OVERALL, THE `MEAN_SHIFT` FUNCTION PERFORMS AN ITERATIVE PROCESS WHERE IT IDENTIFIES THE DATA POINTS WITHIN A SPECIFIED BANDWIDTH FOR EACH CENTROID AND UPDATES THE CENTROIDS BY SHIFTING THEM TOWARDS THE MEAN OF THE POINTS WITHIN THEIR RESPECTIVE BANDWIDTH. THE PROCESS CONTINUES UNTIL CONVERGENCE OR REACHING THE MAXIMUM NUMBER OF ITERATIONS. THE FUNCTION THEN RETURNS THE FINAL CENTROIDS, WHICH REPRESENT THE CLUSTERS IDENTIFIED BY THE MEAN SHIFT CLUSTERING ALGORITHM.
def MEAN_SHIFT(X: np.ndarray, BANDWIDTH: float = 0.5, MAX_ITERATIONS: int = 100) -> np.ndarray:
    """THE `MEAN_SHIFT` FUNCTION IMPLEMENTS THE MEAN SHIFT CLUSTERING ALGORITHM.
        1. IT TAKES INPUT DATA POINTS `X`, BANDWIDTH `BANDWIDTH`, AND MAXIMUM ITERATIONS `MAX_ITERATIONS`.
        2. THE FUNCTION INITIALIZES THE CENTROIDS AS A COPY OF THE INPUT DATA POINTS.
        3. IT ITERATIVELY PERFORMS MEAN SHIFT UNTIL CONVERGENCE OR REACHING THE MAXIMUM ITERATIONS.
        4. WITHIN EACH ITERATION:
            - IT CALCULATES A NEW LIST OF CENTROIDS `NEW_CENTROIDS`.
            - FOR EACH CENTROID, IT FINDS THE DATA POINTS WITHIN THE BANDWIDTH DISTANCE.
            - IT SHIFTS THE CENTROID TOWARDS THE MEAN OF THE DATA POINTS WITHIN THE BANDWIDTH.
        5. THE FUNCTION CHECKS FOR CONVERGENCE BY COMPARING CENTROID LENGTHS AND CLOSENESS OF CENTROIDS.
        6. IF CONVERGENCE IS ACHIEVED, THE FINAL CENTROIDS REPRESENTING THE CLUSTERS ARE RETURNED.
        7. IF THE MAXIMUM ITERATIONS ARE REACHED WITHOUT CONVERGENCE, THE LATEST CENTROIDS ARE RETURNED.
    THE MEAN SHIFT CLUSTERING ALGORITHM FINDS CLUSTERS BY ITERATIVELY SHIFTING CENTROIDS TOWARDS THE MEAN OF NEARBY POINTS WITHIN A SPECIFIED BANDWIDTH."""
    CENTROIDS = X.copy()  # INITIALIZE CENTROIDS AS A COPY OF THE INPUT DATA POINTS
    for _ in range(MAX_ITERATIONS):  # ITERATE UNTIL CONVERGENCE OR MAXIMUM ITERATIONS
        NEW_CENTROIDS = []  # INITIALIZE NEW CENTROIDS
        for CENTROID in CENTROIDS:  # FOR EACH CENTROID
            WITHIN_BANDWIDTH = []  # INITIALIZE DATA POINTS WITHIN BANDWIDTH
            for _X in X:  # FOR EACH DATA POINT
                # IF DATA POINT IS WITHIN BANDWIDTH
                if __EUCLIDEAN_DISTANCE__(_X, CENTROID) <= BANDWIDTH:
                    # ADD DATA POINT TO WITHIN BANDWIDTH
                    WITHIN_BANDWIDTH.append(_X)
            if WITHIN_BANDWIDTH:  # IF THERE ARE DATA POINTS WITHIN BANDWIDTH
                # CALCULATE NEW CENTROID
                NEW_CENTROID = np.mean(WITHIN_BANDWIDTH, axis=0)
                # ADD NEW CENTROID TO NEW CENTROIDS
                NEW_CENTROIDS.append(NEW_CENTROID)
        # IF CONVERGENCE IS ACHIEVED
        if len(NEW_CENTROIDS) == len(CENTROIDS) and np.allclose(CENTROIDS, NEW_CENTROIDS):
            break  # TERMINATE LOOP
        CENTROIDS = NEW_CENTROIDS  # UPDATE CENTROIDS
    return np.array(CENTROIDS)  # RETURN FINAL CENTROIDS

# THE `__EUCLIDEAN_DISTANCE__` FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS IN A MULTI-DIMENSIONAL SPACE. EUCLIDEAN DISTANCE IS A COMMON DISTANCE METRIC USED IN VARIOUS MACHINE LEARNING ALGORITHMS, INCLUDING K-MEANS CLUSTERING.
#     1. THE FUNCTION TAKES TWO INPUT ARRAYS, `X_1` AND `X_2`, WHICH REPRESENT THE COORDINATES OF TWO POINTS IN THE SAME DIMENSIONAL SPACE. EACH ARRAY REPRESENTS THE COORDINATES ALONG EACH DIMENSION OF THE SPACE.
#     2. THE DIFFERENCE BETWEEN THE TWO INPUT POINTS IS COMPUTED BY SUBTRACTING `X_2` FROM `X_1`. THIS OPERATION RESULTS IN A NEW ARRAY REPRESENTING THE VECTOR BETWEEN THE TWO POINTS. EACH ELEMENT OF THE VECTOR CORRESPONDS TO THE DIFFERENCE BETWEEN THE COORDINATES OF THE TWO POINTS ALONG A SPECIFIC DIMENSION.
#     3. THE SQUARE OF EACH ELEMENT IN THE VECTOR IS CALCULATED USING THE `** 2` OPERATOR. THIS OPERATION EFFECTIVELY SQUARES THE DIFFERENCES ALONG EACH DIMENSION. SQUARING THE DIFFERENCES ENSURES THAT NEGATIVE VALUES DO NOT CANCEL OUT POSITIVE VALUES WHEN CALCULATING THE DISTANCE.
#     4. THE SQUARED DIFFERENCES ALONG EACH DIMENSION ARE THEN SUMMED UP USING THE `NP.SUM` FUNCTION WITH THE `AXIS=0` ARGUMENT. THE `AXIS=0` ARGUMENT SPECIFIES THAT THE SUM SHOULD BE PERFORMED ALONG THE FIRST AXIS, WHICH CORRESPONDS TO SUMMING THE SQUARED DIFFERENCES ACROSS EACH DIMENSION. THE RESULT IS A SINGLE SCALAR VALUE REPRESENTING THE SUM OF SQUARED DIFFERENCES.
#     5. THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES IS TAKEN USING THE `NP.SQRT` FUNCTION. THIS STEP IS NECESSARY TO OBTAIN THE ACTUAL EUCLIDEAN DISTANCE. TAKING THE SQUARE ROOT ENSURES THAT THE DISTANCE IS IN THE SAME UNITS AS THE ORIGINAL COORDINATES AND PROVIDES A MEASURE OF THE MAGNITUDE OF THE VECTOR BETWEEN THE TWO POINTS.
#     6. FINALLY, THE EUCLIDEAN DISTANCE IS RETURNED AS THE OUTPUT OF THE FUNCTION.
# IN SUMMARY, THE `__EUCLIDEAN_DISTANCE__` FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS BY FINDING THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES ALONG EACH DIMENSION. THE RESULTING VALUE REPRESENTS THE DISTANCE BETWEEN THE TWO POINTS IN THE MULTI-DIMENSIONAL SPACE. EUCLIDEAN DISTANCE IS COMMONLY USED AS A MEASURE OF SIMILARITY OR DISSIMILARITY BETWEEN POINTS AND IS EMPLOYED IN VARIOUS ALGORITHMS FOR CLUSTERING, CLASSIFICATION, AND OTHER TASKS.
def __EUCLIDEAN_DISTANCE__(X_1: np.ndarray, X_2: np.ndarray) -> float:
    """THE `__EUCLIDEAN_DISTANCE__` FUNCTION IS USED TO CALCULATE THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS IN A MULTI-DIMENSIONAL SPACE. EUCLIDEAN DISTANCE IS A MEASURE OF SIMILARITY OR DISSIMILARITY BETWEEN POINTS, AND IT IS WIDELY EMPLOYED IN VARIOUS MACHINE LEARNING ALGORITHMS, PARTICULARLY IN CLUSTERING AND CLASSIFICATION TASKS.
        1. INPUT: THE FUNCTION TAKES TWO INPUT ARRAYS, `X_1` AND `X_2`, WHICH REPRESENT THE COORDINATES OF THE TWO POINTS IN THE SAME DIMENSIONAL SPACE.
        2. VECTOR CALCULATION: THE FUNCTION COMPUTES THE DIFFERENCE BETWEEN THE TWO INPUT POINTS BY SUBTRACTING `X_2` FROM `X_1`. THIS CREATES A NEW ARRAY THAT REPRESENTS THE VECTOR CONNECTING THE TWO POINTS.
        3. SQUARED DIFFERENCES: EACH ELEMENT IN THE VECTOR IS SQUARED USING THE `** 2` OPERATOR. THIS STEP EFFECTIVELY SQUARES THE DIFFERENCES ALONG EACH DIMENSION, ENSURING THAT NEGATIVE VALUES DO NOT CANCEL OUT POSITIVE VALUES.
        4. SUMMATION: THE SQUARED DIFFERENCES ALONG EACH DIMENSION ARE SUMMED UP USING THE `NP.SUM` FUNCTION WITH THE `AXIS=0` ARGUMENT. THIS RESULTS IN A SINGLE SCALAR VALUE, WHICH REPRESENTS THE SUM OF SQUARED DIFFERENCES.
        5. SQUARE ROOT: THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES IS TAKEN USING THE `NP.SQRT` FUNCTION. THIS YIELDS THE EUCLIDEAN DISTANCE BETWEEN THE TWO POINTS. THE SQUARE ROOT OPERATION ENSURES THAT THE DISTANCE IS IN THE SAME UNITS AS THE ORIGINAL COORDINATES AND PROVIDES A MEASURE OF THE MAGNITUDE OF THE VECTOR BETWEEN THE POINTS.
        6. OUTPUT: THE FUNCTION RETURNS THE EUCLIDEAN DISTANCE AS THE FINAL OUTPUT.
    OVERALL, THE `__EUCLIDEAN_DISTANCE__` FUNCTION ENCAPSULATES THE COMPUTATION OF EUCLIDEAN DISTANCE IN A CONCISE MANNER. IT OFFERS A CONVENIENT WAY TO MEASURE THE SIMILARITY OR DISSIMILARITY BETWEEN POINTS IN A MULTI-DIMENSIONAL SPACE, MAKING IT VALUABLE FOR A RANGE OF MACHINE LEARNING TASKS SUCH AS CLUSTERING, CLASSIFICATION, AND OTHER ALGORITHMS THAT RELY ON DISTANCE METRICS."""
    return np.sqrt(np.sum((X_1 - X_2) ** 2, axis=0))  # RETURN EUCLIDEAN DISTANCE BETWEEN TWO POINTS
