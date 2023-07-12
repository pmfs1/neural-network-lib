import numpy as np

# THE DBSCAN (DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS WITH NOISE) FUNCTION IS A PYTHON IMPLEMENTATION OF THE DBSCAN ALGORITHM, WHICH IS A DENSITY-BASED CLUSTERING ALGORITHM. THE FUNCTION TAKES A DATASET `X` AS INPUT AND PERFORMS CLUSTERING BASED ON TWO PARAMETERS: `MIN_SAMPLES` AND `EPSILON`.
#     1. THE FUNCTION INITIALIZES SOME VARIABLES: `N_SAMPLES` STORES THE NUMBER OF SAMPLES IN THE DATASET, `LABELS` IS AN ARRAY TO STORE THE CLUSTER LABELS FOR EACH SAMPLE, AND `CLUSTER_ID` KEEPS TRACK OF THE CURRENT CLUSTER IDENTIFIER.
#     2. THE `__EXPAND_CLUSTER__` FUNCTION IS A HELPER FUNCTION USED TO EXPAND A CLUSTER BY RECURSIVELY ADDING CONNECTED SAMPLES TO IT. IT TAKES SEVERAL PARAMETERS: `VISITED` IS A BOOLEAN ARRAY INDICATING WHETHER A SAMPLE HAS BEEN VISITED OR NOT, `DISTANCES` IS THE PAIRWISE DISTANCE MATRIX BETWEEN SAMPLES, `NEIGHBOURS` IS A LIST OF INDICES OF NEIGHBOURING SAMPLES, AND `CLUSTER_ID` IS THE IDENTIFIER OF THE CURRENT CLUSTER BEING EXPANDED.
#     3. THE FUNCTION CALCULATES THE PAIRWISE DISTANCES BETWEEN SAMPLES USING THE `PAIRWISE_DISTANCES` FUNCTION. THIS FUNCTION TAKES THE DATASET `X` AS INPUT AND RETURNS A DISTANCE MATRIX, WHERE EACH ELEMENT REPRESENTS THE DISTANCE BETWEEN TWO SAMPLES.
#     4. AN ARRAY `VISITED` IS INITIALIZED TO KEEP TRACK OF VISITED SAMPLES. INITIALLY, ALL SAMPLES ARE MARKED AS NOT VISITED.
#     5. THE FUNCTION ENTERS A LOOP THAT ITERATES OVER EACH SAMPLE IN THE DATASET. IF A SAMPLE HAS ALREADY BEEN VISITED, IT IS SKIPPED.
#     6. FOR EACH UNVISITED SAMPLE, THE FUNCTION MARKS IT AS VISITED AND FINDS ITS NEIGHBOURING SAMPLES WITHIN A DISTANCE OF `EPSILON` BY QUERYING THE DISTANCE MATRIX. THE INDICES OF THE NEIGHBOURING SAMPLES ARE STORED IN THE `NEIGHBOURS` ARRAY, WHICH IS INITIALLY POPULATED WITH THE DIRECT NEIGHBOURS OF THE CURRENT SAMPLE.
#     7. THE NUMBER OF NEIGHBOURS (`NUM_NEIGHBOURS`) IS CALCULATED BASED ON THE LENGTH OF THE `NEIGHBOURS` ARRAY.
#     8. IF THE NUMBER OF NEIGHBOURS IS LESS THAN `MIN_SAMPLES`, THE CURRENT SAMPLE IS LABELED AS NOISE (CLUSTER LABEL -1) SINCE IT DOES NOT HAVE ENOUGH NEIGHBOURS TO FORM A DENSE REGION.
#     9. IF THE NUMBER OF NEIGHBOURS IS GREATER THAN OR EQUAL TO `MIN_SAMPLES`, A NEW CLUSTER IS CREATED. THE `CLUSTER_ID` IS INCREMENTED, AND THE CURRENT SAMPLE IS ASSIGNED THE NEW CLUSTER LABEL.
#     10. THE `__EXPAND_CLUSTER__` FUNCTION IS CALLED WITH THE CURRENT SAMPLE'S NEIGHBOURS, AND IT RECURSIVELY EXPANDS THE CLUSTER BY ADDING CONNECTED SAMPLES. THE `__EXPAND_CLUSTER__` FUNCTION CHECKS IF A NEIGHBOUR HAS BEEN VISITED AND EXPANDS THE CLUSTER IF IT HAS ENOUGH NEIGHBOURS.
#     11. AFTER PROCESSING ALL SAMPLES, THE FUNCTION RETURNS THE `LABELS` ARRAY, WHICH CONTAINS THE CLUSTER LABELS FOR EACH SAMPLE IN THE DATASET.
# IN SUMMARY, THE DBSCAN FUNCTION PERFORMS DENSITY-BASED CLUSTERING BY ITERATIVELY EXPLORING THE DATASET AND EXPANDING CLUSTERS BASED ON DENSITY AND PROXIMITY. IT USES A DISTANCE MATRIX TO EFFICIENTLY CALCULATE PAIRWISE DISTANCES BETWEEN SAMPLES AND ASSIGNS CLUSTER LABELS BASED ON THE MINIMUM NUMBER OF NEIGHBOURS (`MIN_SAMPLES`) AND A DISTANCE THRESHOLD (`EPSILON`). THE ALGORITHM IS CAPABLE OF DISCOVERING CLUSTERS OF ARBITRARY SHAPE AND IDENTIFYING NOISE POINTS THAT DO NOT BELONG TO ANY CLUSTER.


def DBSCAN(X, MIN_SAMPLES=5, EPSILON=0.5):
    """THE DBSCAN (DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS WITH NOISE) FUNCTION IS A PYTHON IMPLEMENTATION OF THE DBSCAN ALGORITHM, A DENSITY-BASED CLUSTERING ALGORITHM. IT TAKES A DATASET `X` AS INPUT AND PERFORMS CLUSTERING BASED ON TWO PARAMETERS: `MIN_SAMPLES` AND `EPSILON`.
        1. PAIRWISE DISTANCES CALCULATION:
            - THE FUNCTION FIRST CALCULATES THE PAIRWISE DISTANCES BETWEEN SAMPLES IN THE DATASET USING THE `PAIRWISE_DISTANCES` FUNCTION.
            - THIS STEP COMPUTES THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF SAMPLES AND STORES THE DISTANCES IN A DISTANCE MATRIX.
        2. CLUSTER EXPANSION:
            - THE FUNCTION INITIALIZES VARIABLES SUCH AS THE NUMBER OF SAMPLES, CLUSTER LABELS, AND A CLUSTER IDENTIFIER.
            - IT THEN ITERATES OVER EACH UNVISITED SAMPLE IN THE DATASET.
            - FOR EACH UNVISITED SAMPLE, IT MARKS IT AS VISITED AND IDENTIFIES ITS NEIGHBOURING SAMPLES WITHIN A DISTANCE OF `EPSILON`.
            - IF THE NUMBER OF NEIGHBOURS IS LESS THAN `MIN_SAMPLES`, THE SAMPLE IS LABELED AS NOISE.
            - IF THE NUMBER OF NEIGHBOURS IS GREATER THAN OR EQUAL TO `MIN_SAMPLES`, A NEW CLUSTER IS CREATED. THE SAMPLE IS ASSIGNED THE CLUSTER LABEL, AND THE CLUSTER IS EXPANDED BY RECURSIVELY ADDING CONNECTED SAMPLES.
            - THE EXPANSION CONTINUES UNTIL NO MORE NEIGHBOURS CAN BE ADDED TO THE CLUSTER.
        3. RESULT:
            - AFTER PROCESSING ALL SAMPLES, THE FUNCTION RETURNS AN ARRAY OF CLUSTER LABELS FOR EACH SAMPLE.
            - SAMPLES LABELED AS NOISE HAVE THE CLUSTER LABEL -1.
    THE DBSCAN ALGORITHM IS CAPABLE OF DISCOVERING CLUSTERS OF ARBITRARY SHAPE, AS IT IDENTIFIES DENSE REGIONS OF SAMPLES. IT IS PARTICULARLY USEFUL WHEN DEALING WITH DATASETS CONTAINING CLUSTERS OF DIFFERENT SIZES AND SHAPES, AND IT CAN EFFECTIVELY HANDLE OUTLIERS OR NOISE POINTS THAT DO NOT BELONG TO ANY CLUSTER.
    OVERALL, THE DBSCAN FUNCTION PERFORMS DENSITY-BASED CLUSTERING BY ITERATIVELY EXPANDING CLUSTERS BASED ON DENSITY AND PROXIMITY, RESULTING IN THE ASSIGNMENT OF CLUSTER LABELS TO THE SAMPLES IN THE DATASET."""
    N_SAMPLES = X.shape[0]  # NUMBER OF SAMPLES IN THE DATASET
    # ARRAY OF CLUSTER LABELS FOR EACH SAMPLE
    LABELS = np.zeros(N_SAMPLES, dtype=int)
    CLUSTER_ID = 0  # CLUSTER IDENTIFIER

    # THE `__EXPAND_CLUSTER__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE DBSCAN ALGORITHM TO EXPAND A CLUSTER BY RECURSIVELY ADDING CONNECTED SAMPLES TO IT.
    #     1. PARAMETERS:
    #         - `VISITED`: A BOOLEAN ARRAY INDICATING WHETHER A SAMPLE HAS BEEN VISITED OR NOT. IT HELPS TO KEEP TRACK OF WHICH SAMPLES HAVE BEEN PROCESSED.
    #         - `DISTANCES`: THE PAIRWISE DISTANCE MATRIX BETWEEN SAMPLES. IT PROVIDES INFORMATION ABOUT THE DISTANCES BETWEEN SAMPLES, WHICH IS USED TO IDENTIFY NEIGHBOURING SAMPLES.
    #         - `NEIGHBOURS`: A LIST OF INDICES REPRESENTING THE NEIGHBOURING SAMPLES OF THE CURRENT SAMPLE BEING PROCESSED.
    #         - `CLUSTER_ID`: THE IDENTIFIER OF THE CURRENT CLUSTER BEING EXPANDED. IT HELPS IN ASSIGNING THE APPROPRIATE CLUSTER LABEL TO NEWLY ADDED SAMPLES.
    #     2. ITERATING OVER NEIGHBOURS:
    #         - THE FUNCTION ENTERS A LOOP THAT ITERATES OVER EACH INDEX IN THE `NEIGHBOURS` LIST. THESE INDICES REPRESENT THE NEIGHBOURING SAMPLES OF THE CURRENT SAMPLE.
    #         - FOR EACH NEIGHBOUR, THE FUNCTION CHECKS WHETHER IT HAS BEEN VISITED OR NOT.
    #         - IF THE NEIGHBOUR HAS NOT BEEN VISITED (I.E., `VISITED[NEIGHBOUR_INDEX]` IS `FALSE`), IT PROCEEDS WITH THE EXPANSION PROCESS.
    #     3. UPDATING VISITED STATUS:
    #         - THE FUNCTION MARKS THE CURRENT NEIGHBOUR AS VISITED BY SETTING `VISITED[NEIGHBOUR_INDEX]` TO `TRUE`. THIS ENSURES THAT THE NEIGHBOUR IS NOT PROCESSED AGAIN IN SUBSEQUENT ITERATIONS.
    #     4. EXPLORING NEIGHBOUR'S NEIGHBOURS:
    #         - THE FUNCTION RETRIEVES THE NEIGHBOURS OF THE CURRENT NEIGHBOUR (`NEIGHBOUR_INDEX`) FROM THE `DISTANCES` MATRIX. THESE NEIGHBOURS REPRESENT POTENTIAL CANDIDATES TO BE ADDED TO THE CLUSTER.
    #         - IF THE NUMBER OF NEIGHBOURS IS GREATER THAN OR EQUAL TO THE `MIN_SAMPLES` THRESHOLD, IT INDICATES THAT THE NEIGHBOUR IS A CORE POINT (I.E., IT HAS SUFFICIENT NEIGHBOURING SAMPLES TO FORM A DENSE REGION).
    #         - IN THIS CASE, THE FUNCTION EXTENDS THE `NEIGHBOURS` LIST WITH THE NEIGHBOUR'S NEIGHBOURS (`NEIGHBOUR_NEIGHBOURS`). THESE NEIGHBOURS WILL BE CONSIDERED FOR CLUSTER EXPANSION IN SUBSEQUENT ITERATIONS.
    #     5. ASSIGNING CLUSTER LABELS:
    #         - IF THE CURRENT NEIGHBOUR HAS NOT BEEN ASSIGNED A CLUSTER LABEL (I.E., `LABELS[NEIGHBOUR_INDEX]` IS 0), IT IS ASSIGNED THE `CLUSTER_ID` OBTAINED FROM THE MAIN DBSCAN FUNCTION. THIS ENSURES THAT THE NEWLY ADDED SAMPLES ARE PART OF THE SAME CLUSTER.
    #     6. RECURSIVE EXPANSION:
    #         - THE FUNCTION RECURSIVELY CALLS ITSELF WITH THE UPDATED VISITED STATUS, DISTANCES, EXPANDED NEIGHBOURS, AND THE SAME CLUSTER IDENTIFIER.
    #         - THIS RECURSIVE CALL EXPANDS THE CLUSTER FURTHER BY ADDING CONNECTED SAMPLES UNTIL NO MORE NEIGHBOURS CAN BE ADDED TO THE CLUSTER.
    # THE `__EXPAND_CLUSTER__` FUNCTION PLAYS A CRUCIAL ROLE IN THE DBSCAN ALGORITHM AS IT ALLOWS FOR THE ITERATIVE EXPANSION OF CLUSTERS BASED ON DENSITY AND CONNECTIVITY. IT ENSURES THAT ALL SAMPLES WITHIN A CLUSTER ARE CONNECTED BY DENSE REGIONS AND ASSIGNS THE APPROPRIATE CLUSTER LABELS TO NEWLY ADDED SAMPLES.
    def __EXPAND_CLUSTER__(VISITED, DISTANCES, NEIGHBOURS, CLUSTER_ID):
        """THE `__EXPAND_CLUSTER__` FUNCTION IS A HELPER FUNCTION USED WITHIN THE DBSCAN ALGORITHM TO EXPAND CLUSTERS BY RECURSIVELY ADDING CONNECTED SAMPLES.
            1. VISITED STATUS:
                - THE FUNCTION KEEPS TRACK OF VISITED SAMPLES USING THE `VISITED` ARRAY, ENSURING THAT EACH SAMPLE IS PROCESSED ONLY ONCE.
            2. NEIGHBOURING SAMPLES:
                - IT ITERATES OVER THE NEIGHBOURS OF THE CURRENT SAMPLE TO IDENTIFY UNVISITED NEIGHBOURS THAT CAN BE ADDED TO THE CLUSTER.
                - IF A NEIGHBOUR HAS NOT BEEN VISITED, IT IS CONSIDERED FOR CLUSTER EXPANSION.
            3. UPDATING VISITED STATUS:
                - THE FUNCTION MARKS THE CURRENT NEIGHBOUR AS VISITED, PREVENTING IT FROM BEING PROCESSED AGAIN.
            4. EXPLORING NEIGHBOUR'S NEIGHBOURS:
                - IT RETRIEVES THE NEIGHBOURS OF THE CURRENT NEIGHBOUR AND DETERMINES IF IT HAS ENOUGH NEIGHBOURS TO BE A CORE POINT.
                - IF THE CURRENT NEIGHBOUR MEETS THE MINIMUM NEIGHBOUR THRESHOLD, ITS NEIGHBOURS ARE CONSIDERED FOR CLUSTER EXPANSION IN SUBSEQUENT ITERATIONS.
            5. ASSIGNING CLUSTER LABELS:
                - IF A NEIGHBOUR DOES NOT HAVE A CLUSTER LABEL ASSIGNED, IT IS ASSIGNED THE CURRENT CLUSTER IDENTIFIER.
                - THIS ENSURES THAT THE NEWLY ADDED SAMPLES ARE PART OF THE SAME CLUSTER.
            6. RECURSIVE EXPANSION:
                - THE FUNCTION RECURSIVELY CALLS ITSELF, CONTINUING THE CLUSTER EXPANSION PROCESS WITH THE UPDATED VISITED STATUS, DISTANCES, EXPANDED NEIGHBOURS, AND CLUSTER IDENTIFIER.
                - THIS RECURSIVE CALL EXPANDS THE CLUSTER BY ADDING CONNECTED SAMPLES UNTIL NO MORE NEIGHBOURS CAN BE ADDED.
        THE `__EXPAND_CLUSTER__` FUNCTION PLAYS A VITAL ROLE WITHIN DBSCAN, ENABLING THE ITERATIVE EXPANSION OF CLUSTERS BASED ON DENSITY AND CONNECTIVITY. BY EXPLORING NEIGHBOURING SAMPLES AND RECURSIVELY ADDING CONNECTED SAMPLES, IT FACILITATES THE FORMATION OF CLUSTERS IN THE DATASET."""
        for NEIGHBOUR_INDEX in NEIGHBOURS:  # ITERATE OVER THE NEIGHBOURS OF THE CURRENT SAMPLE
            if not VISITED[NEIGHBOUR_INDEX]:  # IF THE NEIGHBOUR HAS NOT BEEN VISITED
                # MARK THE NEIGHBOUR AS VISITED
                VISITED[NEIGHBOUR_INDEX] = True
                # GET THE NEIGHBOUR'S NEIGHBOURS
                NEIGHBOUR_NEIGHBOURS = DISTANCES[NEIGHBOUR_INDEX, 1:]
                # GET THE NUMBER OF NEIGHBOUR'S NEIGHBOURS
                NUM_NEIGHBOUR_NEIGHBOURS = len(NEIGHBOUR_NEIGHBOURS)
                if NUM_NEIGHBOUR_NEIGHBOURS >= MIN_SAMPLES:  # IF THE NEIGHBOUR HAS ENOUGH NEIGHBOURS TO BE A CORE POINT
                    # ADD THE NEIGHBOUR'S NEIGHBOURS TO THE LIST OF NEIGHBOURS
                    NEIGHBOURS.extend(NEIGHBOUR_NEIGHBOURS)
                # IF THE NEIGHBOUR DOES NOT HAVE A CLUSTER LABEL ASSIGNED
                if LABELS[NEIGHBOUR_INDEX] == 0:
                    # ASSIGN THE CURRENT CLUSTER IDENTIFIER
                    LABELS[NEIGHBOUR_INDEX] = CLUSTER_ID

    # CALCULATE THE PAIRWISE DISTANCES BETWEEN THE SAMPLES
    DISTANCES = PAIRWISE_DISTANCES(X)
    # INITIALISE THE VISITED STATUS OF ALL SAMPLES TO FALSE
    VISITED = np.zeros(N_SAMPLES, dtype=bool)
    for i in range(N_SAMPLES):  # ITERATE OVER THE SAMPLES
        if VISITED[i]:  # IF THE SAMPLE HAS BEEN VISITED
            continue  # SKIP THE SAMPLE
        VISITED[i] = True  # MARK THE SAMPLE AS VISITED
        # GET THE NEIGHBOURS OF THE CURRENT SAMPLE
        NEIGHBOURS = np.where(DISTANCES[i, :] <= EPSILON)[0]
        NUM_NEIGHBOURS = len(NEIGHBOURS)  # GET THE NUMBER OF NEIGHBOURS
        if NUM_NEIGHBOURS < MIN_SAMPLES:  # IF THE SAMPLE DOES NOT HAVE ENOUGH NEIGHBOURS TO BE A CORE POINT
            LABELS[i] = -1  # ASSIGN THE SAMPLE AS NOISE
        else:  # IF THE SAMPLE HAS ENOUGH NEIGHBOURS TO BE A CORE POINT
            CLUSTER_ID += 1  # INCREMENT THE CLUSTER IDENTIFIER
            LABELS[i] = CLUSTER_ID  # ASSIGN THE SAMPLE TO THE CURRENT CLUSTER
            # EXPAND THE CLUSTER BY ADDING CONNECTED SAMPLES
            __EXPAND_CLUSTER__(VISITED, DISTANCES,
                               NEIGHBOURS.tolist(), CLUSTER_ID)
    return LABELS  # RETURN THE CLUSTER LABELS

# THE `PAIRWISE_DISTANCES` FUNCTION CALCULATES THE PAIRWISE DISTANCES BETWEEN A SET OF SAMPLES REPRESENTED BY A MATRIX `X`.
#     1. THE FUNCTION TAKES A SINGLE PARAMETER `X`, WHICH IS EXPECTED TO BE A NUMPY ARRAY REPRESENTING THE SAMPLES. EACH ROW IN `X` CORRESPONDS TO A SINGLE SAMPLE, AND THE COLUMNS REPRESENT DIFFERENT FEATURES OR DIMENSIONS OF THE SAMPLES.
#     2. `N_SAMPLES = X.SHAPE[0]` ASSIGNS THE NUMBER OF SAMPLES IN `X` TO THE VARIABLE `N_SAMPLES`. THIS VALUE IS USED TO DETERMINE THE SIZE OF THE RESULTING DISTANCE MATRIX.
#     3. `DISTANCES = NP.ZEROS((N_SAMPLES, N_SAMPLES))` CREATES AN EMPTY SQUARE MATRIX OF SIZE `N_SAMPLES` BY `N_SAMPLES` USING NUMPY'S `ZEROS` FUNCTION. THIS MATRIX WILL STORE THE PAIRWISE DISTANCES BETWEEN THE SAMPLES.
#     4. THE FUNCTION ENTERS A NESTED LOOP WITH THE OUTER LOOP ITERATING OVER THE RANGE OF `N_SAMPLES`. THE VARIABLE `I` REPRESENTS THE CURRENT ROW INDEX.
#     5. THE INNER LOOP ITERATES OVER THE RANGE FROM `I + 1` TO `N_SAMPLES`. THE VARIABLE `J` REPRESENTS THE CURRENT COLUMN INDEX.
#     6. INSIDE THE NESTED LOOP, `DISTANCES[I, J] = DISTANCES[J, I] = NP.LINALG.NORM(X[I] - X[J])` CALCULATES THE EUCLIDEAN DISTANCE BETWEEN THE `I`-TH AND `J`-TH SAMPLES. `NP.LINALG.NORM` COMPUTES THE NORM OF THE VECTOR DIFFERENCE `X[I] - X[J]`, WHICH REPRESENTS THE DISTANCE BETWEEN THE TWO SAMPLES. THE CALCULATED DISTANCE IS THEN ASSIGNED TO BOTH `DISTANCES[I, J]` AND `DISTANCES[J, I]` TO ENSURE THAT THE DISTANCE MATRIX IS SYMMETRIC.
#     7. AFTER THE NESTED LOOPS HAVE ITERATED OVER ALL PAIRS OF SAMPLES, THE FUNCTION RETURNS THE RESULTING DISTANCE MATRIX `DISTANCES`.
# IN SUMMARY, THE `PAIRWISE_DISTANCES` FUNCTION CALCULATES THE PAIRWISE DISTANCES BETWEEN SAMPLES BY COMPUTING THE EUCLIDEAN DISTANCE BETWEEN EACH PAIR OF SAMPLES AND STORING THE DISTANCES IN A SYMMETRIC MATRIX. THIS FUNCTION CAN BE USEFUL IN VARIOUS APPLICATIONS SUCH AS CLUSTERING, DIMENSIONALITY REDUCTION, OR SIMILARITY ANALYSIS.


def PAIRWISE_DISTANCES(X):
    """THE `PAIRWISE_DISTANCES` FUNCTION IS A PYTHON IMPLEMENTATION THAT CALCULATES THE PAIRWISE DISTANCES BETWEEN SAMPLES IN A GIVEN DATASET. IT UTILIZES THE NUMPY LIBRARY FOR EFFICIENT MATHEMATICAL OPERATIONS.
    THE FUNCTION TAKES A MATRIX `X` AS INPUT, WHERE EACH ROW REPRESENTS A SAMPLE AND THE COLUMNS CORRESPOND TO DIFFERENT FEATURES OR DIMENSIONS OF THE SAMPLES. IT INITIALIZES AN EMPTY DISTANCE MATRIX OF SIZE `N_SAMPLES` BY `N_SAMPLES`, WHERE `N_SAMPLES` IS THE NUMBER OF SAMPLES IN THE INPUT MATRIX.
    USING NESTED LOOPS, THE FUNCTION ITERATES OVER EACH PAIR OF SAMPLES, EXCLUDING SELF-PAIRS AND REDUNDANT COMPUTATIONS DUE TO SYMMETRY. FOR EACH PAIR, IT CALCULATES THE EUCLIDEAN DISTANCE BETWEEN THE SAMPLES USING THE `NP.LINALG.NORM` FUNCTION, WHICH CALCULATES THE NORM OF THE VECTOR DIFFERENCE. THE RESULTING DISTANCE IS STORED IN BOTH THE CORRESPONDING CELLS OF THE DISTANCE MATRIX TO ENSURE SYMMETRY.
    ONCE ALL PAIRWISE DISTANCES HAVE BEEN COMPUTED, THE FUNCTION RETURNS THE DISTANCE MATRIX, WHICH PROVIDES A COMPREHENSIVE REPRESENTATION OF THE DISTANCES BETWEEN ALL SAMPLES IN THE DATASET.
    THIS FUNCTION CAN BE VALUABLE IN VARIOUS APPLICATIONS SUCH AS CLUSTERING ALGORITHMS, WHERE KNOWING THE DISTANCES BETWEEN SAMPLES IS CRUCIAL FOR GROUPING SIMILAR SAMPLES TOGETHER. ADDITIONALLY, IT CAN BE USED IN DIMENSIONALITY REDUCTION TECHNIQUES OR SIMILARITY ANALYSIS, WHERE THE DISTANCES BETWEEN SAMPLES HELP QUANTIFY THE RELATIONSHIPS OR SIMILARITIES BETWEEN THEM."""
    N_SAMPLES = X.shape[0]  # ASSIGN THE NUMBER OF SAMPLES IN `X` TO THE VARIABLE `N_SAMPLES`.
    # INITIALIZE AN EMPTY DISTANCE MATRIX OF SIZE `N_SAMPLES` BY `N_SAMPLES`.
    DISTANCES = np.zeros((N_SAMPLES, N_SAMPLES))
    for i in range(N_SAMPLES):  # ITERATE OVER THE SAMPLES IN `X`.
        # ITERATE OVER THE SAMPLES IN `X` STARTING FROM THE CURRENT SAMPLE.
        for j in range(i + 1, N_SAMPLES):
            # CALCULATE THE EUCLIDEAN DISTANCE BETWEEN THE CURRENT SAMPLE AND THE `J`-TH SAMPLE.
            DISTANCES[i, j] = DISTANCES[j, i] = np.linalg.norm(X[i] - X[j])
    return DISTANCES  # RETURN THE DISTANCE MATRIX.
