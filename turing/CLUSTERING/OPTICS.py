import numpy as np

# THE `OPTICS` FUNCTION IS AN IMPLEMENTATION OF THE OPTICS (ORDERING POINTS TO IDENTIFY THE CLUSTERING STRUCTURE) ALGORITHM. IT IS A DENSITY-BASED CLUSTERING ALGORITHM THAT AIMS TO DISCOVER CLUSTERS OF ARBITRARY SHAPE IN A DATASET.
#     1. THE FUNCTION TAKES THREE PARAMETERS:
#         - `X`: THE INPUT DATASET, REPRESENTED AS A NUMPY ARRAY, WHERE EACH ROW CORRESPONDS TO A DATA POINT.
#         - `MIN_SAMPLES`: THE MINIMUM NUMBER OF SAMPLES REQUIRED FOR A CLUSTER TO BE FORMED.
#         - `EPSILON`: THE MAXIMUM DISTANCE BETWEEN TWO SAMPLES FOR THEM TO BE CONSIDERED NEIGHBOURS.
#     2. THE `OPTICS` FUNCTION CONTAINS SEVERAL HELPER FUNCTIONS THAT ARE USED WITHIN THE MAIN FUNCTION:
#         - `__CALCULATE_DISTANCES__`: IT CALCULATES THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET USING THE `PDIST` FUNCTION.
#         - `__CALCULATE_CORE_DISTANCES__`: IT CALCULATES THE CORE DISTANCES FOR EACH POINT BASED ON THE MINIMUM NUMBER OF SAMPLES (`MIN_SAMPLES`). THE CORE DISTANCE IS DEFINED AS THE DISTANCE TO THE `MIN_SAMPLES`-TH NEAREST NEIGHBOUR.
#         - `__CALCULATE_REACHABILITY_DISTANCES__`: IT CALCULATES THE REACHABILITY DISTANCES FOR EACH POINT. THE REACHABILITY DISTANCE OF A POINT IS DEFINED AS THE MAXIMUM OF THE CORE DISTANCE OF THE POINT ITSELF AND THE DISTANCE TO ITS NEAREST NEIGHBOURS WITHIN A DISTANCE OF `EPSILON`.
#         - `__EXTRACT_CLUSTERS__`: IT EXTRACTS THE CLUSTERS FROM THE REACHABILITY DISTANCES. IT ITERATES THROUGH THE POINTS IN A SPECIFIC ORDER, STARTING FROM THE ONE WITH THE LOWEST REACHABILITY DISTANCE. IT FORMS A CLUSTER BY INCLUDING ALL POINTS THAT HAVE A REACHABILITY DISTANCE BELOW `EPSILON` AND ARE CONNECTED TO THE CURRENT POINT.
#     3. THE MAIN PART OF THE `OPTICS` FUNCTION STARTS BY CALCULATING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET USING THE `__CALCULATE_DISTANCES__` FUNCTION.
#     4. THE CORE DISTANCES ARE THEN COMPUTED FOR EACH POINT USING THE `__CALCULATE_CORE_DISTANCES__` FUNCTION.
#     5. NEXT, THE REACHABILITY DISTANCES ARE CALCULATED FOR EACH POINT USING THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION.
#     6. FINALLY, THE CLUSTERS ARE EXTRACTED FROM THE REACHABILITY DISTANCES USING THE `__EXTRACT_CLUSTERS__` FUNCTION.
#     7. THE RESULTING CLUSTERS ARE RETURNED AS A LIST OF LISTS, WHERE EACH INNER LIST REPRESENTS A CLUSTER AND CONTAINS THE INDICES OF THE DATA POINTS BELONGING TO THAT CLUSTER.
# IN SUMMARY, THE `OPTICS` FUNCTION TAKES A DATASET, MINIMUM SAMPLES, AND AN OPTIONAL EPSILON VALUE, AND PERFORMS THE OPTICS CLUSTERING ALGORITHM. IT CALCULATES PAIRWISE DISTANCES, CORE DISTANCES, REACHABILITY DISTANCES, AND EXTRACTS CLUSTERS BASED ON REACHABILITY DISTANCES. THE FUNCTION RETURNS A LIST OF CLUSTERS FOUND IN THE DATASET.
def OPTICS(X: np.ndarray, MIN_SAMPLES: int, EPSILON: float) -> list:
    """THE `OPTICS` FUNCTION IS AN IMPLEMENTATION OF THE OPTICS (ORDERING POINTS TO IDENTIFY THE CLUSTERING STRUCTURE) ALGORITHM. IT IS A DENSITY-BASED CLUSTERING ALGORITHM THAT AIMS TO DISCOVER CLUSTERS OF ARBITRARY SHAPE IN A DATASET. 
    THE FUNCTION TAKES A DATASET, MINIMUM SAMPLES, AND AN OPTIONAL EPSILON VALUE AS INPUTS. IT CALCULATES PAIRWISE DISTANCES, CORE DISTANCES, REACHABILITY DISTANCES, AND EXTRACTS CLUSTERS BASED ON REACHABILITY DISTANCES. 
        1. PAIRWISE DISTANCES: IT CALCULATES THE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
        2. CORE DISTANCES: IT DETERMINES THE CORE DISTANCE FOR EACH POINT, WHICH IS THE DISTANCE TO THE `MIN_SAMPLES`-TH NEAREST NEIGHBOUR.
        3. REACHABILITY DISTANCES: IT CALCULATES THE REACHABILITY DISTANCE FOR EACH POINT, CONSIDERING THE MAXIMUM OF THE CORE DISTANCE OF THE POINT ITSELF AND THE DISTANCE TO ITS NEAREST NEIGHBOURS WITHIN A DISTANCE OF `EPSILON`.
        4. EXTRACT CLUSTERS: IT EXTRACTS CLUSTERS FROM THE REACHABILITY DISTANCES BY ITERATING THROUGH THE POINTS IN A SPECIFIC ORDER AND INCLUDING ALL POINTS THAT HAVE A REACHABILITY DISTANCE BELOW `EPSILON` AND ARE CONNECTED TO THE CURRENT POINT.
        5. RETURN CLUSTERS: THE FUNCTION RETURNS A LIST OF CLUSTERS FOUND IN THE DATASET, WHERE EACH INNER LIST REPRESENTS A CLUSTER AND CONTAINS THE INDICES OF THE DATA POINTS BELONGING TO THAT CLUSTER.
    OVERALL, THE `OPTICS` FUNCTION PROVIDES A WAY TO PERFORM DENSITY-BASED CLUSTERING, ALLOWING FOR THE DISCOVERY OF CLUSTERS WITH VARYING DENSITIES AND SHAPES IN A DATASET."""

    # THE `__CALCULATE_DISTANCES__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE `OPTICS` FUNCTION TO COMPUTE THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
    #     1. THE FUNCTION TAKES A NUMPY ARRAY `X` AS INPUT, WHICH REPRESENTS THE DATASET. EACH ROW IN `X` CORRESPONDS TO A DATA POINT.
    #     2. THE `__CALCULATE_DISTANCES__` FUNCTION USES ANOTHER HELPER FUNCTION CALLED `PDIST` TO CALCULATE THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
    #     3. THE `PDIST` FUNCTION ITERATES OVER EACH PAIR OF POINTS IN THE DATASET. IT STARTS BY INITIALIZING AN EMPTY NUMPY ARRAY CALLED `PAIRWISE_DISTANCES` TO STORE THE DISTANCES.
    #     4. USING NESTED LOOPS, THE FUNCTION ITERATES OVER THE INDICES `I` AND `J`, WHERE `I` RANGES FROM 0 TO `N - 1` AND `J` RANGES FROM `I + 1` TO `N`, WHERE `N` IS THE NUMBER OF POINTS IN THE DATASET.
    #     5. FOR EACH PAIR OF POINTS `(X[I], X[J])`, THE EUCLIDEAN DISTANCE BETWEEN THEM IS CALCULATED USING THE FORMULA `DISTANCE = NP.SQRT(NP.SUM((X[I] - X[J]) ** 2))`. THIS FORMULA COMPUTES THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES BETWEEN THE CORRESPONDING ELEMENTS OF `X[I]` AND `X[J]`.
    #     6. THE CALCULATED DISTANCE `DISTANCE` IS THEN STORED IN THE `PAIRWISE_DISTANCES` ARRAY AT THE APPROPRIATE INDEX `IDX`. THE VARIABLE `IDX` KEEPS TRACK OF THE CURRENT POSITION IN THE ARRAY.
    #     7. AFTER ALL PAIRWISE DISTANCES HAVE BEEN CALCULATED AND STORED IN THE `PAIRWISE_DISTANCES` ARRAY, THE FUNCTION RETURNS THE ARRAY.
    # IN SUMMARY, THE `__CALCULATE_DISTANCES__` FUNCTION COMPUTES THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET BY ITERATING OVER EACH PAIR OF POINTS AND CALCULATING THE EUCLIDEAN DISTANCE BETWEEN THEM. THE RESULTING DISTANCES ARE STORED IN A NUMPY ARRAY, WHICH IS THEN RETURNED BY THE FUNCTION. THIS DISTANCE CALCULATION IS A CRUCIAL STEP IN VARIOUS CLUSTERING ALGORITHMS, INCLUDING OPTICS, AS IT PROVIDES THE BASIS FOR MEASURING THE SIMILARITY OR DISSIMILARITY BETWEEN DATA POINTS.
    def __CALCULATE_DISTANCES__(X: np.ndarray) -> np.ndarray:
        """THE `__CALCULATE_DISTANCES__` FUNCTION IS A HELPER FUNCTION USED IN THE `OPTICS` ALGORITHM TO COMPUTE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET. 
            1. INPUT: IT TAKES A NUMPY ARRAY `X` REPRESENTING THE DATASET AS INPUT.
            2. PAIRWISE DISTANCES: THE FUNCTION USES A NESTED LOOP TO ITERATE OVER ALL PAIRS OF POINTS IN THE DATASET. IT CALCULATES THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF POINTS USING THE FORMULA `DISTANCE = NP.SQRT(NP.SUM((X[I] - X[J]) ** 2))`.
            3. STORAGE: THE DISTANCES ARE STORED IN A NUMPY ARRAY CALLED `PAIRWISE_DISTANCES` AS THEY ARE COMPUTED.
            4. OUTPUT: ONCE ALL PAIRWISE DISTANCES HAVE BEEN CALCULATED, THE FUNCTION RETURNS THE `PAIRWISE_DISTANCES` ARRAY.
        IN SUMMARY, THE `__CALCULATE_DISTANCES__` FUNCTION COMPUTES PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET USING THE EUCLIDEAN DISTANCE FORMULA. THE RESULTING DISTANCES ARE STORED AND RETURNED AS A NUMPY ARRAY. THIS CALCULATION IS AN ESSENTIAL STEP IN THE OPTICS ALGORITHM AND PROVIDES A MEASURE OF DISSIMILARITY OR SIMILARITY BETWEEN DATA POINTS, ENABLING THE IDENTIFICATION OF CLUSTERS BASED ON THEIR DISTANCES FROM EACH OTHER."""
        return SQUAREFORM(PDIST(X))  # THE `SQUAREFORM` FUNCTION CONVERTS THE ARRAY OF PAIRWISE DISTANCES INTO A SQUARE MATRIX, WHERE EACH ROW AND COLUMN CORRESPONDS TO A DATA POINT AND EACH ELEMENT IN THE MATRIX CORRESPONDS TO THE DISTANCE BETWEEN THE POINTS REPRESENTED BY THE ROW AND COLUMN INDICES.

    # THE `__CALCULATE_CORE_DISTANCES__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE `OPTICS` ALGORITHM TO CALCULATE THE CORE DISTANCES FOR EACH DATA POINT.
    #     1. THE FUNCTION TAKES TWO PARAMETERS AS INPUT:
    #         - `DISTANCES`: A NUMPY ARRAY CONTAINING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
    #         - `MIN_SAMPLES`: THE MINIMUM NUMBER OF SAMPLES REQUIRED FOR A CLUSTER TO BE FORMED.
    #     2. THE `__CALCULATE_CORE_DISTANCES__` FUNCTION AIMS TO DETERMINE THE CORE DISTANCE FOR EACH DATA POINT. THE CORE DISTANCE IS DEFINED AS THE DISTANCE TO THE `MIN_SAMPLES`-TH NEAREST NEIGHBOUR.
    #     3. THE FUNCTION BEGINS BY SORTING THE `DISTANCES` ARRAY ALONG THE 0-AXIS, WHICH CORRESPONDS TO THE DISTANCES BETWEEN EACH POINT AND ITS NEIGHBOURS.
    #     4. THE SORTED DISTANCES ARE THEN ACCESSED USING INDEXING TO OBTAIN THE `MIN_SAMPLES`-TH DISTANCE FOR EACH POINT. THESE DISTANCES REPRESENT THE `MIN_SAMPLES`-TH NEAREST NEIGHBOURS.
    #     5. THE `__CALCULATE_CORE_DISTANCES__` FUNCTION RETURNS THE ARRAY OF `MIN_SAMPLES`-TH NEAREST NEIGHBOUR DISTANCES AS THE CORE DISTANCES FOR EACH POINT IN THE DATASET.
    # IN SUMMARY, THE `__CALCULATE_CORE_DISTANCES__` FUNCTION CALCULATES THE CORE DISTANCES FOR EACH DATA POINT BY SORTING THE PAIRWISE DISTANCES AND EXTRACTING THE `MIN_SAMPLES`-TH NEAREST NEIGHBOUR DISTANCE FOR EACH POINT. THE CORE DISTANCES REPRESENT THE MINIMUM DISTANCE REQUIRED FOR A POINT TO BE CONSIDERED A CORE POINT IN THE OPTICS ALGORITHM.
    def __CALCULATE_CORE_DISTANCES__(DISTANCES: np.ndarray, MIN_SAMPLES: int) -> np.ndarray:
        """THE `__CALCULATE_CORE_DISTANCES__` FUNCTION IS A HELPER FUNCTION USED IN THE `OPTICS` ALGORITHM TO DETERMINE THE CORE DISTANCES FOR EACH DATA POINT.
            1. INPUT: THE FUNCTION TAKES TWO PARAMETERS AS INPUTS:
                - `DISTANCES`: A NUMPY ARRAY CONTAINING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
                - `MIN_SAMPLES`: THE MINIMUM NUMBER OF SAMPLES REQUIRED FOR A CLUSTER TO BE FORMED.
            2. CORE DISTANCES: THE FUNCTION SORTS THE `DISTANCES` ARRAY TO OBTAIN THE DISTANCES BETWEEN EACH POINT AND ITS NEIGHBOURS. BY SORTING ALONG THE 0-AXIS, THE FUNCTION ENSURES THAT THE DISTANCES FOR EACH POINT ARE ARRANGED IN ASCENDING ORDER.
            3. EXTRACTING CORE DISTANCES: THE `MIN_SAMPLES`-TH DISTANCE FOR EACH POINT IS OBTAINED FROM THE SORTED DISTANCES. THESE DISTANCES REPRESENT THE `MIN_SAMPLES`-TH NEAREST NEIGHBOURS OF EACH POINT.
            4. OUTPUT: THE FUNCTION RETURNS THE ARRAY OF `MIN_SAMPLES`-TH NEAREST NEIGHBOUR DISTANCES, WHICH SERVE AS THE CORE DISTANCES FOR EACH DATA POINT.
        IN SUMMARY, THE `__CALCULATE_CORE_DISTANCES__` FUNCTION CALCULATES THE CORE DISTANCES FOR EACH DATA POINT BY SORTING THE PAIRWISE DISTANCES AND EXTRACTING THE `MIN_SAMPLES`-TH NEAREST NEIGHBOUR DISTANCE FOR EACH POINT. THE CORE DISTANCES PROVIDE A MEASURE OF HOW DENSE THE NEIGHBOURHOOD OF A POINT IS AND ARE USED IN THE OPTICS ALGORITHM TO IDENTIFY CORE POINTS AND CHARACTERIZE THE CLUSTERING STRUCTURE OF THE DATASET."""
        return np.sort(DISTANCES, axis=0)[MIN_SAMPLES]  # THE `NP.SORT` FUNCTION SORTS THE `DISTANCES` ARRAY ALONG THE 0-AXIS, WHICH CORRESPONDS TO THE DISTANCES BETWEEN EACH POINT AND ITS NEIGHBOURS. THE `MIN_SAMPLES`-TH NEAREST NEIGHBOUR DISTANCES ARE THEN EXTRACTED FROM THE SORTED ARRAY AND RETURNED AS THE CORE DISTANCES FOR EACH POINT.

    # THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE `OPTICS` ALGORITHM TO COMPUTE THE REACHABILITY DISTANCES FOR EACH DATA POINT.
    #     1. THE FUNCTION TAKES THREE PARAMETERS AS INPUTS:
    #         - `DISTANCES`: A NUMPY ARRAY CONTAINING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
    #         - `CORE_DISTANCES`: AN ARRAY OF CORE DISTANCES FOR EACH DATA POINT.
    #         - `EPSILON`: THE MAXIMUM DISTANCE BETWEEN TWO SAMPLES FOR THEM TO BE CONSIDERED NEIGHBOURS.
    #     2. THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION AIMS TO DETERMINE THE REACHABILITY DISTANCE FOR EACH DATA POINT. THE REACHABILITY DISTANCE IS DEFINED AS THE MAXIMUM OF THE CORE DISTANCE OF THE POINT ITSELF AND THE DISTANCE TO ITS NEAREST NEIGHBOURS WITHIN A DISTANCE OF `EPSILON`.
    #     3. THE FUNCTION STARTS BY OBTAINING THE NUMBER OF POINTS IN THE DATASET (`N_POINTS`) FROM THE SHAPE OF THE `DISTANCES` ARRAY.
    #     4. IT INITIALIZES AN ARRAY CALLED `REACHABILITY_DISTANCES` WITH THE SIZE OF `N_POINTS` AND FILLS IT WITH INFINITY VALUES.
    #     5. THE FUNCTION THEN ITERATES OVER EACH POINT IN THE DATASET USING A LOOP. FOR EACH POINT:
    #         - IT CHECKS IF THE CORE DISTANCE OF THE POINT (`CORE_DISTANCES[POINT]`) IS LESS THAN INFINITY. IF IT IS, IT MEANS THE POINT IS A CORE POINT.
    #         - IT IDENTIFIES THE NEIGHBOURHOOD OF THE CURRENT POINT BY FINDING THE INDICES OF THE POINTS WHOSE DISTANCE TO THE CURRENT POINT IS LESS THAN OR EQUAL TO `EPSILON`.
    #         - IT CALCULATES THE REACHABILITY DISTANCE FOR THE CURRENT POINT AS THE MAXIMUM VALUE BETWEEN THE CORE DISTANCE OF THE POINT ITSELF AND THE MAXIMUM DISTANCE TO ITS NEIGHBOURHOOD. THIS IS DONE USING THE `NP.MAX(NP.MAXIMUM(CORE_DISTANCES[POINT], DISTANCES[POINT][NEIGHBOURHOOD]))` EXPRESSION.
    #     6. FINALLY, THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION RETURNS THE ARRAY OF REACHABILITY DISTANCES.
    # IN SUMMARY, THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION COMPUTES THE REACHABILITY DISTANCES FOR EACH DATA POINT BASED ON THE CORE DISTANCES AND THE DISTANCES TO THE NEAREST NEIGHBOURS WITHIN A SPECIFIED `EPSILON` DISTANCE. THE REACHABILITY DISTANCE IS DEFINED AS THE MAXIMUM OF THE CORE DISTANCE AND THE DISTANCE TO THE NEAREST NEIGHBOURS. IT PROVIDES A MEASURE OF HOW CONNECTED OR REACHABLE A POINT IS WITHIN THE DATASET.
    def __CALCULATE_REACHABILITY_DISTANCES__(DISTANCES: np.ndarray, CORE_DISTANCES: np.ndarray, EPSILON: float) -> np.ndarray:
        """THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION IS A HELPER FUNCTION USED IN THE `OPTICS` ALGORITHM TO CALCULATE THE REACHABILITY DISTANCES FOR EACH DATA POINT.
            1. INPUT: THE FUNCTION TAKES THREE PARAMETERS AS INPUT:
                - `DISTANCES`: A NUMPY ARRAY CONTAINING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
                - `CORE_DISTANCES`: AN ARRAY OF CORE DISTANCES FOR EACH DATA POINT.
                - `EPSILON`: THE MAXIMUM DISTANCE BETWEEN TWO SAMPLES FOR THEM TO BE CONSIDERED NEIGHBOURS.
            2. REACHABILITY DISTANCES: THE FUNCTION ITERATES OVER EACH DATA POINT IN THE DATASET AND PERFORMS THE FOLLOWING STEPS:
                - CHECKS IF THE POINT IS A CORE POINT BY VERIFYING IF ITS CORE DISTANCE IS LESS THAN INFINITY.
                - IDENTIFIES THE NEIGHBOURHOOD OF THE CURRENT POINT BY FINDING THE INDICES OF POINTS WHOSE DISTANCE TO THE CURRENT POINT IS LESS THAN OR EQUAL TO `EPSILON`.
                - COMPUTES THE REACHABILITY DISTANCE FOR THE CURRENT POINT AS THE MAXIMUM VALUE BETWEEN ITS OWN CORE DISTANCE AND THE MAXIMUM DISTANCE TO ITS NEIGHBOURHOOD.
            3. OUTPUT: THE FUNCTION RETURNS AN ARRAY OF REACHABILITY DISTANCES, WHERE EACH DISTANCE CORRESPONDS TO A DATA POINT IN THE DATASET.
        IN SUMMARY, THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION CALCULATES THE REACHABILITY DISTANCES FOR EACH DATA POINT BASED ON THEIR CORE DISTANCES AND THE DISTANCES TO THEIR NEIGHBOURING POINTS WITHIN A SPECIFIED `EPSILON` DISTANCE. THE REACHABILITY DISTANCE PROVIDES INFORMATION ABOUT THE RELATIVE CONNECTIVITY OR REACHABILITY OF A POINT WITHIN THE DATASET, CONTRIBUTING TO THE CLUSTERING ANALYSIS PERFORMED BY THE OPTICS ALGORITHM."""
        N_POINTS = DISTANCES.shape[0]  # THE NUMBER OF POINTS IN THE DATASET IS OBTAINED FROM THE SHAPE OF THE `DISTANCES` ARRAY.
        # THE `REACHABILITY_DISTANCES` ARRAY IS INITIALIZED WITH THE SIZE OF `N_POINTS` AND FILLED WITH INFINITY VALUES.
        REACHABILITY_DISTANCES = np.full(N_POINTS, np.inf)
        # THE FUNCTION ITERATES OVER EACH POINT IN THE DATASET USING A LOOP.
        for POINT in range(N_POINTS):
            # FOR EACH POINT, IT CHECKS IF THE CORE DISTANCE OF THE POINT IS LESS THAN INFINITY. IF IT IS, IT MEANS THE POINT IS A CORE POINT.
            if CORE_DISTANCES[POINT] < np.inf:
                # THE NEIGHBOURHOOD OF THE CURRENT POINT IS IDENTIFIED BY FINDING THE INDICES OF THE POINTS WHOSE DISTANCE TO THE CURRENT POINT IS LESS THAN OR EQUAL TO `EPSILON`.
                NEIGHBOURHOOD = np.where(DISTANCES[POINT] <= EPSILON)[0]
                # THE REACHABILITY DISTANCE FOR THE CURRENT POINT IS CALCULATED AS THE MAXIMUM VALUE BETWEEN THE CORE DISTANCE OF THE POINT ITSELF AND THE MAXIMUM DISTANCE TO ITS NEIGHBOURHOOD.
                REACHABILITY_DISTANCES[POINT] = np.max(np.maximum(
                    CORE_DISTANCES[POINT], DISTANCES[POINT][NEIGHBOURHOOD]))
        # THE `__CALCULATE_REACHABILITY_DISTANCES__` FUNCTION RETURNS THE ARRAY OF REACHABILITY DISTANCES.
        return REACHABILITY_DISTANCES

    # THE `__EXTRACT_CLUSTERS__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE `OPTICS` ALGORITHM TO EXTRACT CLUSTERS FROM THE REACHABILITY DISTANCES.
    #     1. THE FUNCTION TAKES FOUR PARAMETERS AS INPUT:
    #         - `DISTANCES`: A NUMPY ARRAY CONTAINING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
    #         - `REACHABILITY_DISTANCES`: AN ARRAY OF REACHABILITY DISTANCES FOR EACH DATA POINT.
    #         - `EPSILON`: THE MAXIMUM DISTANCE BETWEEN TWO SAMPLES FOR THEM TO BE CONSIDERED NEIGHBOURS.
    #         - `CORE_DISTANCES`: AN ARRAY OF CORE DISTANCES FOR EACH DATA POINT.
    #     2. THE `__EXTRACT_CLUSTERS__` FUNCTION AIMS TO IDENTIFY CLUSTERS FROM THE REACHABILITY DISTANCES BY ITERATING THROUGH THE POINTS IN A SPECIFIC ORDER.
    #     3. THE FUNCTION STARTS BY OBTAINING THE NUMBER OF POINTS IN THE DATASET (`N_POINTS`) FROM THE SHAPE OF THE `DISTANCES` ARRAY.
    #     4. IT INITIALIZES AN EMPTY LIST CALLED `CLUSTERS` TO STORE THE CLUSTERS FOUND.
    #     5. THE FUNCTION ITERATES OVER THE ORDERED POINTS USING A LOOP. THE ORDERED POINTS ARE DETERMINED BY SORTING THE INDICES OF `REACHABILITY_DISTANCES` IN ASCENDING ORDER.
    #     6. FOR EACH POINT IN THE ORDERED POINTS:
    #         - IT CHECKS IF THE REACHABILITY DISTANCE OF THE POINT IS GREATER THAN `EPSILON`. IF IT IS, THE POINT IS NOT CONSIDERED PART OF A CLUSTER.
    #         - IF THE CORE DISTANCE OF THE POINT IS LESS THAN OR EQUAL TO `EPSILON`, IT SIGNIFIES THAT THE POINT IS A POTENTIAL SEED OF A NEW CLUSTER.
    #         - IT STARTS A NEW CLUSTER BY CREATING A LIST CALLED `CURRENT_CLUSTER` AND ADDS THE CURRENT POINT TO IT.
    #         - IT THEN FINDS THE INDEX OF THE CURRENT POINT IN THE ORDERED POINTS LIST AND ASSIGNS IT TO A VARIABLE CALLED `IDX`.
    #         - WHILE `IDX` IS LESS THAN THE TOTAL NUMBER OF POINTS AND THE REACHABILITY DISTANCE OF THE NEXT ORDERED POINT IS LESS THAN OR EQUAL TO `EPSILON`, IT ADDS THE NEXT ORDERED POINT TO THE `CURRENT_CLUSTER` AND INCREMENTS `IDX` BY 1.
    #         - AFTER COMPLETING THE INNER WHILE LOOP, THE `CURRENT_CLUSTER` REPRESENTS A COMPLETE CLUSTER, AND IT IS APPENDED TO THE `CLUSTERS` LIST.
    #     7. FINALLY, THE `__EXTRACT_CLUSTERS__` FUNCTION RETURNS THE `CLUSTERS` LIST CONTAINING THE IDENTIFIED CLUSTERS.
    # IN SUMMARY, THE `__EXTRACT_CLUSTERS__` FUNCTION EXTRACTS CLUSTERS FROM THE REACHABILITY DISTANCES BY ITERATING THROUGH THE ORDERED POINTS AND IDENTIFYING CONTIGUOUS POINTS WITH REACHABILITY DISTANCES BELOW `EPSILON`. IT FORMS CLUSTERS BY INCLUDING ALL SUCH POINTS AND RETURNS A LIST OF CLUSTERS FOUND IN THE DATASET, WHERE EACH INNER LIST REPRESENTS A CLUSTER AND CONTAINS THE INDICES OF THE DATA POINTS BELONGING TO THAT CLUSTER.
    def __EXTRACT_CLUSTERS__(DISTANCES: np.ndarray, REACHABILITY_DISTANCES: np.ndarray, EPSILON: float, CORE_DISTANCES: np.ndarray) -> list:
        """THE `__EXTRACT_CLUSTERS__` FUNCTION IS A HELPER FUNCTION USED IN THE `OPTICS` ALGORITHM TO EXTRACT CLUSTERS FROM THE REACHABILITY DISTANCES.
            1. INPUT: THE FUNCTION TAKES FOUR PARAMETERS AS INPUT:
                - `DISTANCES`: A NUMPY ARRAY CONTAINING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
                - `REACHABILITY_DISTANCES`: AN ARRAY OF REACHABILITY DISTANCES FOR EACH DATA POINT.
                - `EPSILON`: THE MAXIMUM DISTANCE BETWEEN TWO SAMPLES FOR THEM TO BE CONSIDERED NEIGHBOURS.
                - `CORE_DISTANCES`: AN ARRAY OF CORE DISTANCES FOR EACH DATA POINT.
            2. CLUSTER EXTRACTION: THE FUNCTION ITERATES OVER THE ORDERED POINTS, WHICH ARE DETERMINED BY SORTING THE INDICES OF THE `REACHABILITY_DISTANCES` ARRAY IN ASCENDING ORDER.
            3. FOR EACH ORDERED POINT:
                - IT CHECKS IF THE REACHABILITY DISTANCE OF THE POINT EXCEEDS `EPSILON`, INDICATING IT IS NOT PART OF A CLUSTER.
                - IF THE CORE DISTANCE OF THE POINT IS LESS THAN OR EQUAL TO `EPSILON`, IT STARTS A NEW CLUSTER BY CREATING A LIST AND ADDING THE CURRENT POINT TO IT.
                - IT CONTINUES TO ADD SUBSEQUENT POINTS TO THE CLUSTER AS LONG AS THEIR REACHABILITY DISTANCES ARE LESS THAN OR EQUAL TO `EPSILON`.
                - ONCE ALL POINTS CONNECTED TO THE CURRENT CLUSTER HAVE BEEN ADDED, THE CLUSTER IS CONSIDERED COMPLETE AND APPENDED TO THE LIST OF CLUSTERS.
            4. OUTPUT: THE FUNCTION RETURNS A LIST OF CLUSTERS FOUND IN THE DATASET, WHERE EACH INNER LIST REPRESENTS A CLUSTER AND CONTAINS THE INDICES OF THE DATA POINTS BELONGING TO THAT CLUSTER.
        IN SUMMARY, THE `__EXTRACT_CLUSTERS__` FUNCTION IDENTIFIES CLUSTERS BY ITERATIVELY EXAMINING THE REACHABILITY DISTANCES OF ORDERED POINTS. IT FORMS CLUSTERS BY INCLUDING POINTS WITH REACHABILITY DISTANCES BELOW `EPSILON` AND RETURNS A LIST OF THESE CLUSTERS. THIS STEP IS CRUCIAL IN THE OPTICS ALGORITHM AS IT ORGANIZES THE DATA POINTS INTO COHERENT CLUSTERS BASED ON THEIR CONNECTIVITY AND DENSITY."""
        N_POINTS = DISTANCES.shape[0]  # NUMBER OF POINTS IN THE DATASET
        # ORDERED POINTS BY REACHABILITY DISTANCE
        ORDERED_POINTS = np.argsort(REACHABILITY_DISTANCES)
        CLUSTERS = []  # LIST OF CLUSTERS: EACH CLUSTER IS A LIST OF POINTS
        for POINT in ORDERED_POINTS:  # ITERATE OVER THE ORDERED POINTS
            # IF REACHABILITY DISTANCE > EPSILON, POINT IS NOT PART OF A CLUSTER
            if REACHABILITY_DISTANCES[POINT] > EPSILON:
                # IF CORE DISTANCE <= EPSILON, POINT IS A POTENTIAL SEED OF A NEW CLUSTER
                if CORE_DISTANCES[POINT] <= EPSILON:
                    CURRENT_CLUSTER = [POINT]  # START A NEW CLUSTER
                    # FIND THE INDEX OF THE CURRENT POINT IN THE ORDERED POINTS LIST
                    IDX = np.where(ORDERED_POINTS == POINT)[0][0] + 1
                    # WHILE REACHABILITY DISTANCE <= EPSILON
                    while IDX < N_POINTS and REACHABILITY_DISTANCES[ORDERED_POINTS[IDX]] <= EPSILON:
                        # ADD THE NEXT ORDERED POINT TO THE CURRENT CLUSTER
                        CURRENT_CLUSTER.append(ORDERED_POINTS[IDX])
                        IDX += 1  # INCREMENT IDX
                    # APPEND THE CURRENT CLUSTER TO THE LIST OF CLUSTERS
                    CLUSTERS.append(CURRENT_CLUSTER)
        return CLUSTERS  # RETURN THE LIST OF CLUSTERS

    # CALCULATE THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET
    DISTANCES = __CALCULATE_DISTANCES__(X)
    # CALCULATE THE CORE DISTANCES FOR ALL POINTS
    CORE_DISTANCES = __CALCULATE_CORE_DISTANCES__(DISTANCES, MIN_SAMPLES)
    REACHABILITY_DISTANCES = __CALCULATE_REACHABILITY_DISTANCES__(
        DISTANCES, CORE_DISTANCES, EPSILON)  # CALCULATE THE REACHABILITY DISTANCES FOR ALL POINTS
    # EXTRACT CLUSTERS FROM THE REACHABILITY DISTANCES
    CLUSTERS = __EXTRACT_CLUSTERS__(
        DISTANCES, REACHABILITY_DISTANCES, EPSILON, CORE_DISTANCES)
    return CLUSTERS  # RETURN THE LIST OF CLUSTERS

# THE `PDIST` FUNCTION IS USED TO CALCULATE THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
#     1. THE FUNCTION TAKES A NUMPY ARRAY `X` AS INPUT, REPRESENTING THE DATASET. EACH ROW IN `X` CORRESPONDS TO A DATA POINT.
#     2. THE `PDIST` FUNCTION CALCULATES THE PAIRWISE DISTANCES BY ITERATING OVER EACH PAIR OF POINTS IN THE DATASET.
#     3. IT STARTS BY INITIALIZING A VARIABLE `N` WITH THE NUMBER OF POINTS IN THE DATASET, OBTAINED FROM THE SHAPE OF `X`.
#     4. THE FUNCTION ALSO INITIALIZES AN EMPTY NUMPY ARRAY CALLED `PAIRWISE_DISTANCES` TO STORE THE CALCULATED DISTANCES. THE SIZE OF THIS ARRAY IS DETERMINED BY THE TOTAL NUMBER OF PAIRWISE DISTANCES, WHICH CAN BE CALCULATED USING THE FORMULA `(N * (N - 1)) // 2`.
#     5. USING NESTED LOOPS, THE FUNCTION ITERATES OVER THE INDICES `I` AND `J`, WHERE `I` RANGES FROM 0 TO `N - 1` AND `J` RANGES FROM `I + 1` TO `N`. THIS ENSURES THAT EACH PAIR OF POINTS IS CONSIDERED ONLY ONCE, WITHOUT REPETITION.
#     6. FOR EACH PAIR OF POINTS `(X[I], X[J])`, THE FUNCTION CALCULATES THE EUCLIDEAN DISTANCE BETWEEN THEM. IT USES THE FORMULA `DISTANCE = NP.SQRT(NP.SUM((X[I] - X[J]) ** 2))`. THIS FORMULA CALCULATES THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES BETWEEN THE CORRESPONDING ELEMENTS OF `X[I]` AND `X[J]`, RESULTING IN THE EUCLIDEAN DISTANCE.
#     7. THE CALCULATED DISTANCE `DISTANCE` IS THEN STORED IN THE `PAIRWISE_DISTANCES` ARRAY AT THE APPROPRIATE INDEX. THE VARIABLE `IDX` IS USED TO KEEP TRACK OF THE CURRENT POSITION IN THE ARRAY.
#     8. AFTER ALL PAIRWISE DISTANCES HAVE BEEN CALCULATED AND STORED IN THE `PAIRWISE_DISTANCES` ARRAY, THE FUNCTION RETURNS THIS ARRAY.
# IN SUMMARY, THE `PDIST` FUNCTION COMPUTES THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET BY ITERATING OVER EACH PAIR OF POINTS AND CALCULATING THE EUCLIDEAN DISTANCE BETWEEN THEM. THE RESULTING DISTANCES ARE STORED IN A NUMPY ARRAY AND RETURNED AS THE OUTPUT. THIS DISTANCE CALCULATION IS A FUNDAMENTAL STEP IN VARIOUS CLUSTERING AND DISTANCE-BASED ALGORITHMS, PROVIDING A MEASURE OF DISSIMILARITY OR SIMILARITY BETWEEN DATA POINTS.
def PDIST(X: np.ndarray) -> np.ndarray:
    """THE `PDIST` FUNCTION IS USED TO COMPUTE THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
        1. INPUT: THE FUNCTION TAKES A NUMPY ARRAY `X` REPRESENTING THE DATASET AS INPUT.
        2. PAIRWISE DISTANCES: THE FUNCTION CALCULATES THE PAIRWISE DISTANCES BY ITERATING OVER EACH PAIR OF POINTS IN THE DATASET.
        3. STORAGE: IT INITIALIZES AN EMPTY NUMPY ARRAY CALLED `PAIRWISE_DISTANCES` TO STORE THE CALCULATED DISTANCES.
        4. DISTANCE CALCULATION: USING NESTED LOOPS, THE FUNCTION ITERATES OVER THE INDICES OF THE POINTS AND CALCULATES THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF POINTS. IT EMPLOYS THE EUCLIDEAN DISTANCE FORMULA, WHICH COMPUTES THE SQUARE ROOT OF THE SUM OF SQUARED DIFFERENCES BETWEEN THE CORRESPONDING ELEMENTS OF THE POINTS' COORDINATES.
        5. STORING DISTANCES: THE CALCULATED PAIRWISE DISTANCES ARE STORED IN THE `PAIRWISE_DISTANCES` ARRAY.
        6. OUTPUT: ONCE ALL PAIRWISE DISTANCES HAVE BEEN CALCULATED AND STORED, THE FUNCTION RETURNS THE `PAIRWISE_DISTANCES` ARRAY.
    IN SUMMARY, THE `PDIST` FUNCTION COMPUTES THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET USING THE EUCLIDEAN DISTANCE FORMULA. THE RESULTING DISTANCES ARE STORED AND RETURNED AS A NUMPY ARRAY. THIS CALCULATION IS A FUNDAMENTAL STEP IN VARIOUS ALGORITHMS THAT RELY ON DISTANCE-BASED MEASURES, ALLOWING FOR THE ASSESSMENT OF SIMILARITIES OR DISSIMILARITIES BETWEEN DATA POINTS."""
    N = X.shape[0]  # NUMBER OF POINTS IN THE DATASET
    # ARRAY TO STORE THE PAIRWISE DISTANCES
    PAIRWISE_DISTANCES = np.zeros((N * (N - 1)) // 2)
    IDX = 0  # INDEX TO KEEP TRACK OF THE CURRENT POSITION IN THE ARRAY
    for i in range(N - 1):  # ITERATE OVER THE INDICES OF THE POINTS
        for j in range(i + 1, N):  # ITERATE OVER THE INDICES OF THE POINTS
            # CALCULATE THE EUCLIDEAN DISTANCE BETWEEN THE POINTS
            DISTANCE = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            # STORE THE DISTANCE IN THE ARRAY
            PAIRWISE_DISTANCES[IDX] = DISTANCE
            IDX += 1  # INCREMENT THE INDEX
    return PAIRWISE_DISTANCES  # RETURN THE ARRAY OF PAIRWISE DISTANCES

# THE `SQUAREFORM` FUNCTION IS USED TO CONVERT PAIRWISE DISTANCES INTO A SQUARE DISTANCE MATRIX.
#     1. THE FUNCTION TAKES A NUMPY ARRAY `PAIRWISE_DISTANCES` AS INPUT, WHICH REPRESENTS THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
#     2. THE `SQUAREFORM` FUNCTION AIMS TO CONVERT THE 1-DIMENSIONAL ARRAY OF PAIRWISE DISTANCES INTO A SQUARE MATRIX OF DISTANCES.
#     3. THE FUNCTION BEGINS BY DETERMINING THE SIZE OF THE SQUARE MATRIX BASED ON THE LENGTH OF THE `PAIRWISE_DISTANCES` ARRAY. THE NUMBER OF POINTS (`N`) IN THE DATASET CAN BE CALCULATED USING THE FORMULA `N = INT(NP.SQRT(2 * LEN(PAIRWISE_DISTANCES)) + 0.5)`.
#     4. IT INITIALIZES AN EMPTY SQUARE DISTANCE MATRIX CALLED `SQUARE_DISTANCES` OF SIZE `N X N`.
#     5. THE FUNCTION USES NESTED LOOPS TO ASSIGN THE PAIRWISE DISTANCES TO THE APPROPRIATE POSITIONS IN THE SQUARE DISTANCE MATRIX.
#     6. FOR EACH PAIR OF POINTS `(I, J)` WHERE `I` RANGES FROM 0 TO `N - 1` AND `J` RANGES FROM `I + 1` TO `N`, THE FUNCTION ASSIGNS THE CORRESPONDING PAIRWISE DISTANCE FROM THE `PAIRWISE_DISTANCES` ARRAY TO `SQUARE_DISTANCES`. IT ENSURES THAT THE DISTANCES ARE ASSIGNED SYMMETRICALLY SINCE THE DISTANCE MATRIX IS SYMMETRIC.
#     7. FINALLY, THE `SQUAREFORM` FUNCTION RETURNS THE SQUARE DISTANCE MATRIX `SQUARE_DISTANCES`.
# IN SUMMARY, THE `SQUAREFORM` FUNCTION CONVERTS THE 1-DIMENSIONAL ARRAY OF PAIRWISE DISTANCES INTO A SQUARE DISTANCE MATRIX. IT DETERMINES THE SIZE OF THE SQUARE MATRIX BASED ON THE LENGTH OF THE PAIRWISE DISTANCES ARRAY, INITIALIZES AN EMPTY SQUARE MATRIX, AND ASSIGNS THE DISTANCES SYMMETRICALLY. THIS CONVERSION IS IMPORTANT FOR SUBSEQUENT CALCULATIONS AND ANALYSES THAT REQUIRE A SQUARE DISTANCE MATRIX REPRESENTATION OF THE PAIRWISE DISTANCES.
def SQUAREFORM(PAIRWISE_DISTANCES: np.ndarray) -> np.ndarray:
    """THE `SQUAREFORM` FUNCTION IS USED TO CONVERT PAIRWISE DISTANCES INTO A SQUARE DISTANCE MATRIX.
        1. INPUT: THE FUNCTION TAKES A NUMPY ARRAY `PAIRWISE_DISTANCES` AS INPUT, REPRESENTING THE PAIRWISE DISTANCES BETWEEN ALL POINTS IN THE DATASET.
        2. SQUARE DISTANCE MATRIX: THE FUNCTION CONVERTS THE 1-DIMENSIONAL ARRAY OF PAIRWISE DISTANCES INTO A SQUARE MATRIX OF DISTANCES.
        3. DETERMINING MATRIX SIZE: IT DETERMINES THE SIZE OF THE SQUARE DISTANCE MATRIX BASED ON THE LENGTH OF THE `PAIRWISE_DISTANCES` ARRAY. THE NUMBER OF POINTS (`N`) IN THE DATASET IS CALCULATED TO ENSURE A SQUARE MATRIX OF APPROPRIATE DIMENSIONS.
        4. MATRIX INITIALIZATION: THE FUNCTION INITIALIZES AN EMPTY SQUARE DISTANCE MATRIX CALLED `SQUARE_DISTANCES` OF SIZE `N X N`.
        5. ASSIGNING DISTANCES: IT USES NESTED LOOPS TO ASSIGN THE PAIRWISE DISTANCES TO THE CORRESPONDING POSITIONS IN THE SQUARE DISTANCE MATRIX. THE DISTANCES ARE ASSIGNED SYMMETRICALLY TO ENSURE THE MATRIX REPRESENTS A VALID DISTANCE MEASURE.
        6. OUTPUT: FINALLY, THE `SQUAREFORM` FUNCTION RETURNS THE SQUARE DISTANCE MATRIX `SQUARE_DISTANCES`.
    IN SUMMARY, THE `SQUAREFORM` FUNCTION CONVERTS THE 1-DIMENSIONAL ARRAY OF PAIRWISE DISTANCES INTO A SQUARE DISTANCE MATRIX. IT DETERMINES THE MATRIX SIZE BASED ON THE LENGTH OF THE PAIRWISE DISTANCES ARRAY, INITIALIZES THE SQUARE MATRIX, AND ASSIGNS THE DISTANCES SYMMETRICALLY. THIS CONVERSION ALLOWS FOR FURTHER ANALYSES AND CALCULATIONS THAT REQUIRE A SQUARE DISTANCE MATRIX REPRESENTATION OF THE PAIRWISE DISTANCES."""
    N = int(np.sqrt(2 * len(PAIRWISE_DISTANCES)) +
            0.5)  # NUMBER OF POINTS IN THE DATASET
    SQUARE_DISTANCES = np.zeros((N, N))  # ARRAY TO STORE THE SQUARE DISTANCES
    IDX = 0  # INDEX TO KEEP TRACK OF THE CURRENT POSITION IN THE ARRAY
    for i in range(N - 1):  # ITERATE OVER THE INDICES OF THE POINTS
        for j in range(i + 1, N):  # ITERATE OVER THE INDICES OF THE POINTS
            # ASSIGN THE DISTANCE TO THE CORRESPONDING POSITION IN THE MATRIX
            SQUARE_DISTANCES[i, j] = PAIRWISE_DISTANCES[IDX]
            # ASSIGN THE DISTANCE TO THE CORRESPONDING POSITION IN THE MATRIX
            SQUARE_DISTANCES[j, i] = PAIRWISE_DISTANCES[IDX]
            IDX += 1  # INCREMENT THE INDEX
    return SQUARE_DISTANCES  # RETURN THE SQUARE DISTANCE MATRIX
