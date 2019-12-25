"""
Fully solve a zero-sum game.
"""

import numpy as np
import scipy.optimize

def solve(A):
    """
    For an n x m game matrix, finds the optimal mixed strategy over the
    n rows for minimizing the losses in the entries A and
    the optimal value of the game when played against the column-maximizing
    adversary.
    """

    n, m = A.shape
    A = np.copy(A)
    mm = A.min()
    A -= mm

    res = scipy.optimize.linprog(
        c=np.ones(m),
        A_ub=-A,
        b_ub=-np.ones(n),
        bounds=(0, None))
    assert res.success, res
    opt_value = 1 / res.fun
    opt_probs = res.x / res.fun

    return opt_probs, opt_value - mm
