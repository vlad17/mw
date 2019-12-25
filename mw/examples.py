import numpy as np

def random_beta(n, m):
    """
    Creates a random beta(0.5, 0.5) matrix of size n x m.
    """
    return np.random.beta(0.5, 0.5, size=(n * m)).reshape(n, m)
