"""
Different types of column players.

These adversaries define the game that the players play, since
they pick the losses.
"""

import numpy as np

class Adversary:
    def reveal_losses(self, t, rng, strategy):
        """
        t - round number
        rng - random number generator
        strategy - array of size n for the player strategy

        Returns:
          (array of size n with losses for this round, realized loss)
        """
        raise NotImplementedError

class OracleAdversary(Adversary):
    def __init__(self, A):
        self.A = A

    def reveal_losses(self, t, rng, strategy):
        n, m = self.A.shape
        adversary_choice = np.argmax(strategy.dot(self.A))
        player_choice = rng.choice(n, p=strategy)
        return self.A[:, adversary_choice], self.A[
            player_choice, adversary_choice]
