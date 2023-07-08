import numpy as np

# THE `MEAN_SHIFT_CLUSTERING` FUNCTION IMPLEMENTS THE MEAN SHIFT CLUSTERING ALGORITHM. IT TAKES THREE MAIN PARAMETERS AS INPUT: `X`, REPRESENTING THE INPUT DATA POINTS, `BANDWIDTH`, WHICH CONTROLS THE SIZE OF THE NEIGHBORHOOD, AND `MAX_ITERATIONS`, THE MAXIMUM NUMBER OF ITERATIONS THE ALGORITHM WILL PERFORM.
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
# OVERALL, THE `MEAN_SHIFT_CLUSTERING` FUNCTION PERFORMS AN ITERATIVE PROCESS WHERE IT IDENTIFIES THE DATA POINTS WITHIN A SPECIFIED BANDWIDTH FOR EACH CENTROID AND UPDATES THE CENTROIDS BY SHIFTING THEM TOWARDS THE MEAN OF THE POINTS WITHIN THEIR RESPECTIVE BANDWIDTH. THE PROCESS CONTINUES UNTIL CONVERGENCE OR REACHING THE MAXIMUM NUMBER OF ITERATIONS. THE FUNCTION THEN RETURNS THE FINAL CENTROIDS, WHICH REPRESENT THE CLUSTERS IDENTIFIED BY THE MEAN SHIFT CLUSTERING ALGORITHM.
def MEAN_SHIFT_CLUSTERING(X, BANDWIDTH=0.5, MAX_ITERATIONS=100):
    """THE `MEAN_SHIFT_CLUSTERING` FUNCTION IMPLEMENTS THE MEAN SHIFT CLUSTERING ALGORITHM.
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
    CENTROIDS = X.copy()
    for _ in range(MAX_ITERATIONS):
        NEW_CENTROIDS = []
        for CENTROID in CENTROIDS:
            WITHIN_BANDWIDTH = []
            for _X in X:
                if __EUCLIDEAN_DISTANCE__(_X, CENTROID) <= BANDWIDTH:
                    WITHIN_BANDWIDTH.append(_X)
            if WITHIN_BANDWIDTH:
                NEW_CENTROID = np.mean(WITHIN_BANDWIDTH, axis=0)
                NEW_CENTROIDS.append(NEW_CENTROID)
        if len(NEW_CENTROIDS) == len(CENTROIDS) and np.allclose(CENTROIDS, NEW_CENTROIDS):
            break
        CENTROIDS = NEW_CENTROIDS
    return np.array(CENTROIDS)

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