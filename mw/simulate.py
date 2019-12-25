"""
Play a zero-sum game.

The player always goes first with a uniform strategy.
"""

import numpy as np
from numpy.random import SeedSequence, default_rng
import ray

from absl import flags

flags.DEFINE_integer("update_every", 10 ** 5,
                     "If >0, print an update on progress after this "
                     "many steps (max'd with what 1% of steps is).")

def play(seed, n, player, adversary, observer, nrounds, update_every=0):
    rng = default_rng(seed)
    strategy = np.ones(n) / n

    if update_every > 0:
        update_every = max(update_every, nrounds // 100)

    for i in range(nrounds):
        losses, realized = adversary.reveal_losses(i, rng, strategy)
        observer.observe(i, nrounds, player, adversary, losses, realized)
        strategy = player.strategy(i, rng, losses)

        if update_every <= 0:
            continue

        t = i + 1
        if t not in [1, nrounds] and t % update_every:
            continue
        if nrounds < update_every:
            continue

        print('{:10d} of {:10d} done ({:4.0%}) - {}'.format(
            t, nrounds, t / nrounds, observer.summary()))

    return observer.results()

ray_play = ray.remote(play)

def play_many(seed, runs, n, player_gen, adversary_gen, observer_gen, nrounds):
    seeds = SeedSequence(seed)
    runs = [
        ray_play.remote(
            s, n, player_gen(), adversary_gen(), observer_gen(), nrounds
            , flags.FLAGS.update_every)
        for i, s in enumerate(seeds.spawn(runs))]
    runs = ray.get(runs)
    stacked = []
    for i in range(len(runs[0])):
        stacked.append(np.stack([r[i] for r in runs]))
    return stacked
