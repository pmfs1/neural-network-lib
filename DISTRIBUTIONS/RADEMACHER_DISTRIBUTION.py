# RADEMACHER_DISTRIBUTION [STATIC]: IMPLEMENTATION OF RADEMACHER DISTRIBUTION. THE RADEMACHER DISTRIBUTION, WHICH TAKES VALUE 1 WITH PROBABILITY 1/2 AND VALUE −1 WITH PROBABILITY 1/2.
class RADEMACHER_DISTRIBUTION:
    # PROBABILITY_MASS_FUNCTION [STATIC]: PROBABILITY MASS FUNCTION OF RADEMACHER DISTRIBUTION
    @staticmethod
    def PROBABILITY_MASS_FUNCTION(K: int) -> float:
        if K == 1:  # IF K IS EQUAL TO 1
            return 1 / 2  # RETURN 1 / 2
        elif K == -1:  # IF K IS EQUAL TO -1
            return 1 / 2  # RETURN 1 / 2
        else:  # ELSE
            return 0  # RETURN 0

    # CUMULATIVE_DISTRIBUTION_FUNCTION [STATIC]: CUMULATIVE DISTRIBUTION FUNCTION OF RADEMACHER DISTRIBUTION
    @staticmethod
    def CUMULATIVE_DISTRIBUTION_FUNCTION(K: int) -> float:
        if K < -1:  # IF K IS LESS THAN -1
            return 0  # RETURN 0
        elif K >= -1 and K < 1:  # IF K IS GREATER THAN OR EQUAL TO -1 AND LESS THAN 1
            return 1 / 2  # RETURN 1 / 2
        elif K >= 1:  # IF K IS GREATER THAN OR EQUAL TO 1
            return 1  # RETURN 1