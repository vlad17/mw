"""
Solution evaluator.
"""

import multiprocessing
import os
import ray
import random
import sys
import pandas as pd
from time import time
import numpy as np
import scipy
import scipy.optimize

def create_random(n):
    def solution(rankings):
        return np.ones(n) / n
    return solution

def sample_ints(n):
    return np.round(4.5 * (2 * np.random.beta(0.5, 0.5, size=n) - 1) + 5.5)

def make_example(n):
    a = sample_ints(n ** 2)
    while True:
        selector = (a < 1) | (a > 10)
        if not np.any(selector):
            break
        a[selector] = sample_ints(np.sum(selector))

    A = a.reshape(n, n)

    return A

def main(argv):
    if len(argv) not in [3, 4, 5]:
        raise ValueError('usage: python evaluate.py n runs [max-steps] [seed]')

    n = int(argv[1])
    runs = int(argv[2])
    max_steps = int(argv[3]) if len(argv) > 3 else None
    seed = int(argv[4]) if len(argv) > 4 else 1234
    ray.init(num_cpus=(multiprocessing.cpu_count() - 1))

    np.random.seed(seed)

    A = make_example(n)

    print('created {}x{} example'.format(n, n))
    res = scipy.optimize.linprog(
        c=np.ones(n),
        A_ub=-A,
        b_ub=-np.ones(n),
        bounds=(0, None))
    assert res.success, res
    opt_value = 1 / res.fun
    opt_probs = res.x / res.fun

    print('optimal perf', opt_value)

    # https://www.cs.princeton.edu/~arora/pubs/MWsurvey.pdf
    # in positive matrix regime
    # expert with eps learning rate and delta additive error
    # eps is also the multiplicative error
    # number of steps T is proportional to 1/(eps*delta)
    #
    # we consider regimes for total error = x = 1, 0.1
    # an optimal allocation gives
    # epsilon = x / (opt * 2) and delta = x / 2

    # compute the highest T we need
    xs = [1, 0.1, 0.01]
    ts = []
    epss = []

    for x in xs:
        eps = x / (opt_value * 2)
        delta = x / 2
        rho = 9 # upper bound on ratings once shifted
        T = max(int(max(np.log(n), 1) * rho * 2 / (delta * eps)), 1)

        ts.append(T)
        epss.append(eps)

    max_steps = max_steps or max(ts)
    print('running to', max_steps, 'steps')

    perf_acc_futs = {}
    for x, eps, T in zip(xs, epss, ts):
        name = 'MW(err={},T=1e{})'.format(x, max(int(np.round(np.log10(T))), 1))
        score_at_t = evaluate.remote(A, lambda n: make_expert(eps, n), runs, max_steps)
        perf_acc_futs[name] = score_at_t

    perf_acc_futs['random'] = evaluate.remote(A, create_random, runs, max_steps)

    perf_acc = {k: ray.get(v) for k, v in perf_acc_futs.items()}

    for i, (name, score_at_t) in enumerate(perf_acc.items(), 1):
        os.makedirs('saved/' + str(i), exist_ok=True)
        os.system(f'echo "{name}" > saved/{i}/txt')
        np.save(f'saved/{i}/score.npy', score_at_t)

    os.system(f'echo "{opt_value}" > saved/opt')
    os.system(f'echo "{n}" > saved/n')
    np.save('saved/t.npy', 1 + np.arange(len(score_at_t)))

    print()
    d = {k: [v[t] if t < len(v) else None for t in ts] for k, v in perf_acc.items()}
    print(pd.DataFrame(d, index=ts))
    print('opt', opt_value)

def make_expert(eps, n):

    experts = np.ones(n) / n

    def algo(ratings):
        nonlocal experts
        if ratings is None:
            return experts

        cost = - ( (ratings - 1) / 9 )
        assert np.all(cost <= 0)
        experts *= (1 + eps) ** (-cost)
        experts /= experts.sum()
        return experts

    return algo

@ray.remote
def evaluate(A, algo_generator, num_runs, max_steps):
    # algo_generator(n) should create a (stateful) function
    # f(rankings) -> next probabilities

    seeds = np.random.randint(2**15, size=num_runs)
    runs = [eval_run.remote(A, algo_generator, max_steps, seed)
            for seed in seeds]

    runs = ray.get(runs)

    score_at_time = np.cumsum(sum(runs))
    score_at_time /= np.arange(1, 1 + max_steps)
    return score_at_time / num_runs

@ray.remote
def eval_run(A, algo_generator, max_steps, seed):

    np.random.seed(seed)
    scores = np.zeros(max_steps)
    n = len(A)

    algo = algo_generator(n)
    ratings = None

    for i in range(max_steps):

        if max_steps > 10 ** 6:
            if (i + 1) % 10 ** 6 == 0:
                print(i+1, 'of', max_steps, 'done')

        probs = np.copy(algo(ratings))

        if np.abs(1 - probs.sum()) > 1e-2:
            raise ValueError(
                'Probabilities should sum to 1 but sum to {} instead'.format(
                    probs.sum()))

        if len(probs) != n:
            raise ValueError(
                'Expecting {} probabilities, got {}'.format(
                    n, len(probs)))

        probs /= probs.sum()
        selection = np.random.choice(n, p=probs)

        # solution is column player, max
        # adversary is row player, min
        adversary = np.argmin(A.dot(probs))

        scores[i] = A[adversary, selection]
        ratings = A[adversary]

    return scores


if __name__ == '__main__':
    main(sys.argv)
