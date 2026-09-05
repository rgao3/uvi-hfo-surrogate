"""
Does the resolution criterion depend on the surrogate model class?

The concern this script addresses: the staircase behaviour that makes the
resolution criterion so visible in the tree-ensemble surrogate is a consequence
of trees being piecewise constant between splits. It is therefore necessary to test whether a smooth surrogate on the same design
shows the effect at all.

We isolate the model-class effect exactly. Because the simulation database is a
complete factorial grid, two interpolators can be built on the *identical*
coarsened design and evaluated on the *identical* held-out levels:

    A. nearest-node      - piecewise constant; the idealised tree ensemble
    B. multilinear       - piecewise linear and continuous; the smooth comparator

Everything else - design, held-out set, reference values, applicability-domain
definition - is held fixed, so any difference in the behaviour of the diagnostic
is attributable to the model class alone.

Two forms of the resolution score are computed:

    R1 = max_a | f(x | a -> u_a) - f(x | a -> l_a) |          (first difference;
                                                               the definition in
                                                               the manuscript)
    R2 = max_a | f(x | a -> u_a) + f(x | a -> l_a) - 2 f(x) | (second difference)

R1 measures the slope the design leaves unresolved; R2 measures the curvature.
For a piecewise-constant model the interpolation error is governed by the slope,
for a piecewise-linear model by the curvature, so the two forms are expected to
behave differently across model classes. Establishing that is the point.

Pure numpy; no scikit-learn, scipy or xgboost dependency, so the result is
reproducible from the released data alone.
"""

import json
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'data', 'U_HFO_ML_Dataset_Final.csv')
AXES = ['Input_pH', 'U_initial', 'Carbonate', 'NaCl', 'Ca', 'Hfo_s']
LOG_AXES = {'U_initial', 'Carbonate', 'NaCl', 'Ca', 'Hfo_s'}
TARGET = 'Ads_%'
N_TEST = 15000          # matches the size of the existing coarsening experiment
EPS = 1e-12


# ----------------------------------------------------------------- utilities
def avg_rank(a):
    """Average ranks, ties shared (Spearman needs this)."""
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = avg_rank(np.asarray(x, float)), avg_rank(np.asarray(y, float))
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else np.nan


# ------------------------------------------------------------- grid assembly
def build_grid():
    df = pd.read_csv(DATA)
    levels, coords = {}, {}
    for ax in AXES:
        lv = np.sort(df[ax].unique())
        levels[ax] = lv
        # transformed coordinate used for interpolation and for the AD metric
        coords[ax] = np.log10(np.clip(lv, EPS, None)) if ax in LOG_AXES else lv.astype(float)
    shape = tuple(len(levels[ax]) for ax in AXES)
    Y = np.full(shape, np.nan)
    idx = tuple(np.searchsorted(levels[ax], df[ax].values) for ax in AXES)
    Y[idx] = df[TARGET].values
    return levels, coords, Y, shape, df


# ------------------------------------------------------- the two interpolators
def interp_multilinear(Yc, cc, query):
    """Multilinear interpolation on the coarse grid. query: (n, 6) in coord space."""
    n, D = query.shape
    lo = np.empty((n, D), int)
    w = np.empty((n, D))
    for d in range(D):
        g = cc[d]
        i = np.clip(np.searchsorted(g, query[:, d], side='right') - 1, 0, len(g) - 2)
        lo[:, d] = i
        span = g[i + 1] - g[i]
        w[:, d] = np.where(span > 0, (query[:, d] - g[i]) / np.where(span > 0, span, 1), 0.0)
    out = np.zeros(n)
    for corner in range(1 << D):
        bits = np.array([(corner >> d) & 1 for d in range(D)])
        wt = np.prod(np.where(bits == 1, w, 1.0 - w), axis=1)
        sel = tuple((lo[:, d] + bits[d]) for d in range(D))
        out += wt * Yc[sel]
    return out


def interp_nearest(Yc, cc, query):
    """Nearest-node interpolation: the idealised piecewise-constant surrogate."""
    n, D = query.shape
    idx = []
    for d in range(D):
        g = cc[d]
        i = np.clip(np.searchsorted(g, query[:, d], side='right') - 1, 0, len(g) - 2)
        # snap to whichever bracketing node is closer
        take_upper = (query[:, d] - g[i]) > (g[i + 1] - query[:, d])
        idx.append(i + take_upper.astype(int))
    return Yc[tuple(idx)]


MODELS = {'nearest (piecewise constant)': interp_nearest,
          'multilinear (smooth)': interp_multilinear}


# ------------------------------------------------------------ resolution score
def resolution_scores(predict, Yc, cc, query):
    """R1 (first difference) and R2 (second difference), max over all six axes."""
    n, D = query.shape
    base = predict(Yc, cc, query)
    r1 = np.zeros(n)
    r2 = np.zeros(n)
    for d in range(D):
        g = cc[d]
        i = np.clip(np.searchsorted(g, query[:, d], side='right') - 1, 0, len(g) - 2)
        ql = query.copy(); ql[:, d] = g[i]
        qu = query.copy(); qu[:, d] = g[i + 1]
        fl = predict(Yc, cc, ql)
        fu = predict(Yc, cc, qu)
        r1 = np.maximum(r1, np.abs(fu - fl))
        r2 = np.maximum(r2, np.abs(fu + fl - 2.0 * base))
    return r1, r2


# ------------------------------------------------------- applicability domain
def ad_distance(train_pts, test_pts, k=5, chunk=1000):
    """Mean Euclidean distance to the k nearest training points, standardised."""
    mu, sd = train_pts.mean(0), train_pts.std(0)
    sd[sd == 0] = 1.0
    A = ((train_pts - mu) / sd).astype(np.float32)
    B = ((test_pts - mu) / sd).astype(np.float32)
    a2 = (A ** 2).sum(1)
    out = np.empty(len(B))
    for s in range(0, len(B), chunk):
        e = min(s + chunk, len(B))
        d2 = a2[None, :] - 2.0 * B[s:e] @ A.T + (B[s:e] ** 2).sum(1)[:, None]
        np.maximum(d2, 0, out=d2)
        part = np.partition(d2, k, axis=1)[:, :k]
        out[s:e] = np.sqrt(part).mean(1)
    return out


# ------------------------------------------------------------------ experiment
def run_axis(axis, levels, coords, Y, keep_idx, test_idx):
    d_ax = AXES.index(axis)
    coarse = [np.arange(len(levels[a])) for a in AXES]
    coarse[d_ax] = np.array(keep_idx)
    Yc = Y[np.ix_(*coarse)]
    cc = [coords[a][coarse[i]] for i, a in enumerate(AXES)]

    # held-out points: the omitted levels of the coarsened axis, full grid elsewhere
    full = [np.arange(len(levels[a])) for a in AXES]
    full[d_ax] = np.array(test_idx)
    mesh = np.meshgrid(*full, indexing='ij')
    flat = np.stack([m.ravel() for m in mesh], axis=1)
    truth = Y[tuple(flat[:, d] for d in range(len(AXES)))]

    ok = np.isfinite(truth)
    flat, truth = flat[ok], truth[ok]
    if len(flat) > N_TEST:
        sel = RNG.choice(len(flat), N_TEST, replace=False)
        flat, truth = flat[sel], truth[sel]

    query = np.stack([coords[a][flat[:, i]] for i, a in enumerate(AXES)], axis=1)

    train_mesh = np.meshgrid(*coarse, indexing='ij')
    train_flat = np.stack([m.ravel() for m in train_mesh], axis=1)
    train_pts = np.stack([coords[a][train_flat[:, i]] for i, a in enumerate(AXES)], axis=1)
    keep = np.isfinite(Y[tuple(train_flat[:, d] for d in range(len(AXES)))])
    train_pts = train_pts[keep]

    ad = ad_distance(train_pts, query)
    sub = train_pts[RNG.choice(len(train_pts), min(4000, len(train_pts)), replace=False)]
    ad_thr = float(np.quantile(ad_distance(train_pts, sub, k=6), 0.95))

    res = {}
    for name, fn in MODELS.items():
        pred = fn(Yc, cc, query)
        good = np.isfinite(pred)
        err = np.abs(pred[good] - truth[good])
        r1, r2 = resolution_scores(fn, Yc, cc, query)
        r1, r2, adg = r1[good], r2[good], ad[good]
        res[name] = dict(
            n=int(good.sum()),
            MAE=float(err.mean()),
            RMSE=float(np.sqrt((err ** 2).mean())),
            rho_R1=spearman(r1, err),
            rho_R2=spearman(r2, err),
            rho_AD=spearman(adg, err),
            R1_flag_frac=None, R2_flag_frac=None,
            _err=err, _r1=r1, _r2=r2, _ad=adg,
        )
    return res, ad_thr


def flag_stats(err, score, thr):
    f = score > thr
    return dict(flagged_pct=float(100 * f.mean()),
                MAE_flagged=float(err[f].mean()) if f.any() else None,
                MAE_unflagged=float(err[~f].mean()) if (~f).any() else None)


if __name__ == '__main__':
    levels, coords, Y, shape, df = build_grid()
    print('grid shape', shape, 'cells', int(np.prod(shape)),
          'missing', int(np.isnan(Y).sum()))

    experiments = {
        'pH (unit -> 2-unit spacing)':
            ('Input_pH', [0, 2, 4, 6, 8], [1, 3, 5, 7]),
        'Hfo_s (0.75 -> 1.5 decade spacing)':
            ('Hfo_s', [0, 2, 4], [1, 3]),
    }

    # conformal half-widths from the manuscript, used as the flagging thresholds
    HALF_WIDTH = {'Input_pH': 3.8404083251953125, 'Hfo_s': 3.724609375}

    summary = {}
    for label, (axis, keep, test) in experiments.items():
        print('\n' + '=' * 78)
        print(label)
        print('=' * 78)
        res, ad_thr = run_axis(axis, levels, coords, Y, keep, test)
        hw = HALF_WIDTH[axis]
        summary[label] = {}
        for name, r in res.items():
            fl1 = flag_stats(r['_err'], r['_r1'], hw)
            fl2 = flag_stats(r['_err'], r['_r2'], hw)
            fad = flag_stats(r['_err'], r['_ad'], ad_thr)
            print(f"\n  {name}")
            print(f"    n = {r['n']},  MAE = {r['MAE']:.3f},  RMSE = {r['RMSE']:.3f}")
            print(f"    Spearman rho vs |error|:  R1 = {r['rho_R1']:.3f}   "
                  f"R2 = {r['rho_R2']:.3f}   AD = {r['rho_AD']:.3f}")
            print(f"    R1 flag: {fl1['flagged_pct']:.1f}%  "
                  f"MAE flagged {fl1['MAE_flagged']}  unflagged {fl1['MAE_unflagged']}")
            print(f"    R2 flag: {fl2['flagged_pct']:.1f}%  "
                  f"MAE flagged {fl2['MAE_flagged']}  unflagged {fl2['MAE_unflagged']}")
            print(f"    AD flag: {fad['flagged_pct']:.1f}%  "
                  f"MAE flagged {fad['MAE_flagged']}  unflagged {fad['MAE_unflagged']}")
            summary[label][name] = {k: v for k, v in r.items() if not k.startswith('_')}
            summary[label][name].update(R1=fl1, R2=fl2, AD=fad, ad_threshold=ad_thr,
                                        flag_threshold=hw)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'model_class_experiment_results.json')
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('\nwritten:', out)
