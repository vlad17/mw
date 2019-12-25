"""
Different types of observers, which record events
as they occur in games.
"""
from .utils import RollingAverageWindow
import numpy as np

class Observer:
    def __init__(self, nobservations):
        self.nobservations = nobservations

    def should_emit(self, t, nrounds):
        """
        Useful utility method which uniformly spaces out
        observations for t ranging from 0 to nrounds,
        and includes the endpoints.
        """
        if self.nobservations >= nrounds:
            return True
        lo = t * self.nobservations // nrounds
        hi = (t + 1) * self.nobservations // nrounds
        return lo < hi or t == 0 or (t + 1) == nrounds

    # observations go from t = 0 to nrounds - 1
    def observe(self, t, nrounds, player, adversary, losses, loss):
        raise NotImplementedError

    def results(self):
        """
        Should return a list of numpy arrays.
        """
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


class MAObserver:
    """
    Track moving average of loss.

    Uses window size max(1, nrecords // nobservations)
    """
    def __init__(self, nobservations, nrecords):
        super().__init__(nobservations)
        self.window_size = max(1, nrecords // nobservations)
        self.raw = RollingAverageWindow(self.window_size)
        self.losses = []
        self.timestamps = []

    def observe(self, t, nrounds, player, adversary, losses, loss):
        self.raw.update(loss)
        if self.should_emit(t, nrounds):
            self.timestamps.append(t)
            self.losses.append(self.raw.value())

    def results(self):
        return [np.array(self.timestamps), np.array(self.losses)]


    def summary(self):
        return "MA(loss) = {:.3f}".format(self.raw.value())

class ErgodicObserver(Observer):
    """
    Track time average of loss.
    """
    def __init__(self, nobservations):
        super().__init__(nobservations)
        self.net_loss = 0
        self.time = 0
        self.losses = []
        self.timestamps = []

    def observe(self, t, nrounds, player, adversary, losses, loss):
        assert self.time == t
        self.time += 1
        self.net_loss += loss
        if self.should_emit(t, nrounds):
            self.timestamps.append(t)
            self.losses.append(self.net_loss / self.time)

    def results(self):
        return [np.array(self.timestamps), np.array(self.losses)]


    def summary(self):
        return "time-avg loss = {:.3f}".format(self.net_loss / max(self.time, 1))

class AveragedErgodicObserver(Observer):
    """
    Track true average of loss.

    Assumes that the expert algorithm was a function of losses, not loss.

    Assumes that the player has `weights` for their most recent strategy.
    """
    def __init__(self, nobservations):
        super().__init__(nobservations)
        self.net_loss = 0
        self.time = 0
        self.losses = []
        self.timestamps = []
        self.nrounds = {}


    def should_emit(self, t, nrounds):
        """
        Logscale version of super().should_emit
        """
        if nrounds not in self.nrounds:
            self.nrounds[nrounds] = np.unique(np.round(np.logspace(
                0, np.log2(nrounds), num=nrounds, base=2))).astype(int)

        arr = self.nrounds[nrounds]
        return arr[np.searchsorted(arr, t)] == t or t == 0 or (t + 1) == nrounds

    def observe(self, t, nrounds, player, adversary, losses, loss):
        assert self.time == t
        self.time += 1
        self.net_loss += np.dot(losses, player.weights)
        if self.should_emit(t, nrounds):
            self.timestamps.append(t)
            self.losses.append(self.net_loss / self.time)

    def results(self):
        return [np.array(self.timestamps), np.array(self.losses)]

    def summary(self):
        return "time-avg loss = {:.3f}".format(self.net_loss / max(self.time, 1))
