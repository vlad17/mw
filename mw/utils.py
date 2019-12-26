"""
Various utility functions used across several files.
"""

import collections
import hashlib
import itertools
import random
import time
from contextlib import contextmanager
import sys

import numpy as np


class _timeit:
    def __init__(self):
        self.seconds = 0

    def set_seconds(self, x):
        self.seconds = x


@contextmanager
def timeit(name):
    """
    Enclose a with-block with to print out block runtime.
    """
    print(name, end='')
    sys.stdout.flush()
    x = _timeit()
    t = time.time()
    yield x
    x.set_seconds(time.time() - t)
    print("  ...took {:10.2f} sec ".format(x.seconds))

class RollingAverageWindow:
    """Creates an automatically windowed rolling average."""

    def __init__(self, window_size):
        self._window_size = window_size
        self._items = collections.deque([], window_size)
        self._total = 0

    def update(self, value):
        """updates the rolling window"""
        if len(self._items) < self._window_size:
            self._total += value
            self._items.append(value)
        else:
            self._total -= self._items.popleft()
            self._total += value
            self._items.append(value)

    def value(self):
        """returns the current windowed avg"""
        if not self._items:
            return 0
        return self._total / len(self._items)


def import_matplotlib():
    """import and return the matplotlib module in a way that uses
    a display-independent backend (import when generating images on
    servers"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def intfmt(maxval, fill=" "):
    """
    returns the appropriate format string for integers that can go up to
    maximum value maxvalue, inclusive.
    """
    vallen = len(str(maxval))
    return "{:" + fill + str(vallen) + "d}"


def chunkify(iterable, n):
    """
    Break up an iterable into chunks of size n, except for the last chunk,
    if the iterable does not divide evenly.
    """
    # https://stackoverflow.com/questions/8991506
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


class OnlineSampler:
    """
    Online sampling algorithm. Given an arbitrary stream of data, this online
    sampler maintains a set of a pre-determined size k that is a simple random
    sample, without replacement, from all observed data in the stream so far.

    In other words, if this sampler has seen n > k data points so far then
    its sample member is a uniformly selected set of k data points among
    those seen.

    Note that observing sample repeatedly does NOT give iid samples between
    updates. Depends on random seed.
    """

    def __init__(self, k):
        self.k = k
        self.n = 0
        self.sample = []

    def update(self, example):
        """
        Observe a datapoint from the incoming stream and possibly include it
        in the sampled set.
        """
        self.n += 1

        if len(self.sample) < self.k:
            self.sample.append(example)
            return

        # We wish to show by induction that the sample list will always be a
        # uniform k-sample without replacement of the n items seen so far
        # through all update calls. When n == k in the base case the unique
        # set of all observed points is vacuously uniformly randomly selected.
        # Now assume the inductive hypothesis holds for n > k.
        # The current set of k samples is a uniform selection without
        # replacement from the n previously observed data points.
        # Now we observe the next example e.
        #
        # Let S be a k-sized set uniformly selected without replacement from
        # our n + 1 points.
        #
        # P{e in S} = Binomial(n, k-1) / Binomial(n, k) = k / (n+1)
        #
        # Then if we include e with the above probability the distribution of S
        # remains uniform (there's a conditioning argument here that I'm too
        # lazy to make).
        #
        # Note we incremented n already at the beginning of this method.

        if random.random() < self.k / self.n:
            i = random.randrange(self.k)
            self.sample[i] = example

class OnlineAverage:
    """
    Online, numerically stable, averaging algorithm.

    Vectorized.
    """

    def __init__(self, init=0):
        self.avg = None
        self.n = 0

    def update(self, example):
        """
        Observe a datapoint from the incoming stream and
        add its effect on the average.

        Returns updated self.value.
        """
        self.n += 1
        if self.avg is None:
            self.avg = np.zeros_like(example)

        self.avg += (example - self.avg) / self.n

        return self.value()

    def value(self):
        """
        Return the current sample average.
        """
        return self.avg
