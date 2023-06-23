import math

# DISCRETE_UNIFORM_DISTRIBUTION: IMPLEMENTATION OF DISCRETE UNIFORM DISTRIBUTION. THE DISCRETE UNIFORM DISTRIBUTION, WHERE ALL ELEMENTS OF A FINITE SET ARE EQUALLY LIKELY. THIS IS THE THEORETICAL DISTRIBUTION MODEL FOR A BALANCED COIN, AN UNBIASED DIE, A CASINO ROULETTE, OR THE FIRST CARD OF A WELL-SHUFFLED DECK.
class DISCRETE_UNIFORM_DISTRIBUTION:
    # PROBABILITY_MASS_FUNCTION [STATIC]: IN PROBABILITY AND STATISTICS, A PROBABILITY MASS FUNCTION IS A FUNCTION THAT GIVES THE PROBABILITY THAT A DISCRETE RANDOM VARIABLE IS EXACTLY EQUAL TO SOME VALUE. THE PROBABILITY MASS FUNCTION IS OFTEN THE PRIMARY MEANS OF DEFINING A DISCRETE PROBABILITY DISTRIBUTION, AND SUCH FUNCTIONS EXIST FOR EITHER DISCRETE RANDOM VARIABLES OR CONTINUOUS RANDOM VARIABLES WHOSE CUMULATIVE DISTRIBUTION FUNCTION IS DISCONTINUOUS. THE PROBABILITY MASS FUNCTION IS SOMETIMES ALSO CALLED THE PROBABILITY FUNCTION, OR THE PROBABILITY DISTRIBUTION FUNCTION.
    @staticmethod
    def PROBABILITY_MASS_FUNCTION(K: int, A: int, B: int) -> float:
        assert A <= B, "A MUST BE LESS THAN OR EQUAL TO B" # ASSERT A IS LESS THAN OR EQUAL TO B
        assert K >= A and K <= B, "K MUST BE GREATER THAN OR EQUAL TO A AND LESS THAN OR EQUAL TO B" # ASSERT K IS GREATER THAN OR EQUAL TO A AND LESS THAN OR EQUAL TO B
        return 1 / (B - A + 1) # RETURN 1 / (B - A + 1)
    
    # CUMULATIVE_DISTRIBUTION_FUNCTION [STATIC]: IN PROBABILITY THEORY AND STATISTICS, THE CUMULATIVE DISTRIBUTION FUNCTION (CDF) OF A REAL-VALUED RANDOM VARIABLE X, OR JUST DISTRIBUTION FUNCTION OF X, EVALUATED AT X, IS THE PROBABILITY THAT X WILL TAKE A VALUE LESS THAN OR EQUAL TO X.
    @staticmethod
    def CUMULATIVE_DISTRIBUTION_FUNCTION(K: int, A: int, B: int) -> float:
        assert A <= B, "A MUST BE LESS THAN OR EQUAL TO B" # ASSERT A IS LESS THAN OR EQUAL TO B
        assert K >= A and K <= B, "K MUST BE GREATER THAN OR EQUAL TO A AND LESS THAN OR EQUAL TO B" # ASSERT K IS GREATER THAN OR EQUAL TO A AND LESS THAN OR EQUAL TO B
        return (math.floor(K) - A + 1) / (B - A + 1) # RETURN (K - A + 1) / (B - A + 1)