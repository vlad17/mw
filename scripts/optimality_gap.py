"""
Plots the optimality gap on a log scale over time.

Base file names from --runs with .npz appended should contain
"t" and "x" arrays, where t is a timestamp and x is a T x trials
2D array of time-average losses.

Assumes that the first run's game matrix is used throughout all the
other runs.
"""

from absl import app, flags
import numpy as np

flags.DEFINE_multi_string("runs", [], "base file names for plotting")

flags.DEFINE_string(
    "outfile", None,
    "output PDF file"
)
flags.mark_flag_as_required("outfile")

from mw.utils import timeit, import_matplotlib
from mw.solve import solve

import json
from itertools import cycle

def l10print(x):
    from math import log10, floor
    return "{}e{}".format(
        floor(x / (10**floor(log10(x)))),
        floor(log10(x)))

def name(settings):
    e = 'eps={}'.format(l10print(settings['eps']))
    if 'decay' in settings and settings['decay']:
        return e + ',decay={}'.format(settings['decay'])
    return e

def _main(_argv):
    assert flags.FLAGS.runs

    A = np.load(flags.FLAGS.runs[0] + '.npz')['A']
    n, m = A.shape
    print('loaded {}x{} game matrix'.format(*A.shape))

    with timeit('solving system'):
        optimal_strategy, opt = solve(A)

    plt = import_matplotlib()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    top = -np.inf
    for run, c in zip(flags.FLAGS.runs, cycle(colors)):
        with open(run + '.json', 'r') as f:
            settings = json.load(f)
        d = np.load(run + '.npz')
        t, x = [d[xx] for xx in ['t', 'x']]

        mid = x - opt
        decay = 'decay' in settings and settings['decay']
        ls = ':' if decay else None
        label = name(settings)
        plt.loglog(t, mid, label=label, color=c, ls=ls)

        if decay:
            continue

        lt = np.log(n) / (settings['eps'] ** 2 * opt)
        plt.axvline(lt, color=c, ls='--',
                    label='T*(eps={})={}'.format(
                        l10print(settings['eps']), l10print(lt)))

    plt.title('{}x{} game opt = {:.5f}'.format(n, m, opt))
    plt.ylabel('optimality gap for time average')
    plt.xlabel('time')

    plt.legend(loc="upper center",bbox_to_anchor=(0.5, -0.12))
    plt.savefig(
        flags.FLAGS.outfile,
        format="pdf",
        bbox_inches="tight")

if __name__ == "__main__":
    app.run(_main)
