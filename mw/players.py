"""
Game-playing algorithms.
"""
import numpy as np

class Player:
    def strategy(self, t, rng, losses):
        """
        t - round number
        rng - random number generator
        losses - array of size n for the losses from last round

        Returns: array of size n with weights for next round's strategy
        """
        raise NotImplementedError

class MW(Player):
    def __init__(self, n, eps, decay):
        super().__init__()
        self.eps = eps
        self.weights = np.ones(n) / n
        self.decay = decay

    def strategy(self, t, rng, losses):
        if self.decay:
            eps = self.eps / (t + 1) ** self.decay
        else:
            eps = self.eps
        self.weights *= 1 - eps * losses
        self.weights /= self.weights.sum()
        return self.weights
