import numpy as np
import os
import sys

def import_matplotlib():
    """import and return the matplotlib module in a way that uses
    a display-independent backend (import when generating images on
    servers"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

def ma(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if True:

    desired_num_points = 100
    t = np.load('saved/t.npy')
    sample = max(len(t) // desired_num_points, 1)

    ys = {}

    for x in os.listdir('saved'):
        if os.path.isfile(f'saved/{x}/txt'):
            name = open(f'saved/{x}/txt').read()
            vals = np.load(f'saved/{x}/score.npy')
            ys[name] = vals

    n = int(open('saved/n').read())
    opt = float(open('saved/opt').read())


    plt = import_matplotlib()

    ma_window = sample
    ts = t[ma_window - 1::][::sample]
    for name in sorted(list(ys)):
        y = ys[name]
        yy = opt - ma(y, ma_window)
        kwargs = {}
        name = name.strip()
        if name == 'random':
            kwargs['linestyle'] = 'None'
            kwargs['marker'] = 'x'
            # kwargs['markersize'] = 10
        plt.semilogy(ts, yy[::sample], label=name, **kwargs)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('t')
    plt.ylabel('gap (opt - avg score over time)')
    plt.title('n = {}, opt = {:.2f}, MA(1e{})'.format(n, opt, int(max(1, np.log10(sample)))))

    print('saving to out.pdf')
    plt.savefig(
        'out.pdf',
        format="pdf",
        bbox_inches="tight",
    )
