"""
Records the online performance (realized random losses over time)
for an incarnation of the mw.py algorithm.

Uses ray.
"""

from absl import app, flags
import numpy as np
import ray

flags.DEFINE_integer("seed", 1234, "random seed")

flags.DEFINE_string(
    "outfile", None,
    "generates an outfile.npz with "
    "x - T array of running average losses, "
    "t - T array of timestamps, "
    "A - n x m game matrix, "
    " and outfile.json with parameters."
)
flags.mark_flag_as_required("outfile")
flags.DEFINE_string(
    "reuse_game_matrix", None, "if set, load the game matrix A from the "
    "given archive. overrides n,m settings.")

flags.DEFINE_string(
    "ray_address", None, "address for ray cluster to use")

flags.DEFINE_integer("n", 10, "number of experts")
flags.DEFINE_integer("m", 100, "number of chance events")
flags.DEFINE_integer("T", 1000, "number of rounds")

flags.DEFINE_float("eps", 0.1, "initial_rate")
# flags.DEFINE_integer("oracle_rate_horizon", None,
#                      "if set, use the oracle fixed learning rate for the "
#                      "given time horizon")
flags.DEFINE_float("decay", None,
                   "learning rate decay exponent")
flags.DEFINE_integer("nobs", 1000, "number of observations")

from mw.examples import random_beta
from mw.solve import solve
from mw.players import MW
from mw.adversaries import OracleAdversary
from mw.observers import AveragedErgodicObserver
from mw.simulate import play_many

import multiprocessing
import json

def _main(_argv):
    seed = abs(int(flags.FLAGS.seed)) % (1 << 15)
    np.random.seed(seed)
    n = flags.FLAGS.n
    m = flags.FLAGS.m
    if flags.FLAGS.reuse_game_matrix:
        A = np.load(flags.FLAGS.reuse_game_matrix)['A']
        n, m = A.shape
    else:
        A = random_beta(n, m)

    print('created {}x{} example'.format(n, m))

    T = flags.FLAGS.T

    eps = flags.FLAGS.eps
    print('fixed learning rate', eps)

    if flags.FLAGS.ray_address:
        ray.init(address=flags.FLAGS.ray_address)
    else:
        ray.init(num_cpus=(multiprocessing.cpu_count() - 1))

    timestamps, time_averages = play_many(
        seed,
        1,
        n,
        lambda: MW(n, eps, flags.FLAGS.decay),
        lambda: OracleAdversary(A),
        lambda: AveragedErgodicObserver(flags.FLAGS.nobs),
        T)

    assert np.all(timestamps == timestamps[0])

    with open(flags.FLAGS.outfile + ".json", "w") as ff:
        fflags = 'eps,decay'.split(',')
        fflags = {f: getattr(flags.FLAGS, f) for f in fflags}
        fflags['eps'] = eps
        json.dump(fflags, ff)

    ff = flags.FLAGS.outfile + '.npz'
    np.savez(ff, t=timestamps[0], x=time_averages[0], A=A)


if __name__ == "__main__":
    app.run(_main)
