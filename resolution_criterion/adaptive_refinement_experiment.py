"""
Does the resolution score allocate a simulation budget better than the alternatives?

The criterion is only useful for modelling practice if it answers the design
question: given budget for N more simulator runs, where should they go? This
script tests that directly, at equal simulator cost, against the two obvious
competitors.

Setup
-----
A deliberately coarse initial design is taken from the full factorial database
(pH at 2-unit spacing, Hfo_s at 1.5-decade spacing, all other axes complete). A
surrogate is fitted to it. Every remaining converged simulation is a *candidate*
that could be purchased. A fixed random 20% of the candidates is held out from
every strategy and used as the common evaluation set; the other 80% forms the
acquisition pool.

Each strategy then spends the same budget N on points from the acquisition pool,
the surrogate is refitted on (initial + purchased), and all strategies are scored
on the identical evaluation set.

Strategies
----------
    random           N points drawn uniformly - the standard space-filling baseline
    resolution_topN  the N points with the largest resolution score R1 (naive greedy)
    resolution_strat score-guided but coverage-preserving: candidates are grouped by
                     the four axes that were not coarsened, and the budget is spent
                     round-robin across groups, taking the highest-scoring candidate
                     from each in turn
    resolution_prob  sampled without replacement with probability proportional to R1
    ad               the N points with the largest applicability-domain distance

Greedy selection on any per-point score collapses onto a single region and loses
the coverage that a space-filling design provides; the two tempered variants test
whether the score still carries usable design information once that failure mode is
controlled for.

The third strategy matters: if the applicability domain guided design as well as
the resolution score does, the criterion would be redundant. It is included to
give the argument a chance to fail.

The surrogate is HistGradientBoostingRegressor - one of the three algorithms
compared in the manuscript - because it accepts scattered training data, which a
grid interpolator cannot. Feature space is the six design axes, log-transformed
where the axis is log-spaced.

No new PHREEQC runs are required: refinement is simulated by revealing points
that already exist in the database.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'U_HFO_ML_Dataset_Final.csv')
AXES = ['Input_pH', 'U_initial', 'Carbonate', 'NaCl', 'Ca', 'Hfo_s']
LOG_AXES = {'U_initial', 'Carbonate', 'NaCl', 'Ca', 'Hfo_s'}
TARGET = 'Ads_%'
EPS = 1e-12
SEED = 42
BUDGETS = [2000, 5000, 10000, 20000]
STRAT_AXES = [1, 2, 3, 4]        # U, Carbonate, NaCl, Ca - the axes not coarsened
N_RANDOM_SEEDS = 2

HGB = dict(max_iter=250, learning_rate=0.05, max_depth=None,
           min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def coords_of(df):
    X = np.empty((len(df), len(AXES)))
    for i, ax in enumerate(AXES):
        v = df[ax].values.astype(float)
        X[:, i] = np.log10(np.clip(v, EPS, None)) if ax in LOG_AXES else v
    return X


def fit(X, y):
    m = HistGradientBoostingRegressor(**HGB)
    m.fit(X, y)
    return m


def resolution_score(model, X, design_levels):
    """R1 = max_a |f(x | a -> u_a) - f(x | a -> l_a)| over the *design* levels."""
    n, D = X.shape
    r = np.zeros(n)
    for d in range(D):
        g = design_levels[d]
        if len(g) < 2:
            continue
        i = np.clip(np.searchsorted(g, X[:, d], side='right') - 1, 0, len(g) - 2)
        Xl = X.copy(); Xl[:, d] = g[i]
        Xu = X.copy(); Xu[:, d] = g[i + 1]
        r = np.maximum(r, np.abs(model.predict(Xu) - model.predict(Xl)))
    return r


def ad_distance(train_X, test_X, k=5, chunk=1000):
    mu, sd = train_X.mean(0), train_X.std(0)
    sd[sd == 0] = 1.0
    A = ((train_X - mu) / sd).astype(np.float32)
    B = ((test_X - mu) / sd).astype(np.float32)
    a2 = (A ** 2).sum(1)
    out = np.empty(len(B))
    for s in range(0, len(B), chunk):
        e = min(s + chunk, len(B))
        d2 = a2[None, :] - 2.0 * B[s:e] @ A.T + (B[s:e] ** 2).sum(1)[:, None]
        np.maximum(d2, 0, out=d2)
        out[s:e] = np.sqrt(np.partition(d2, k, axis=1)[:, :k]).mean(1)
    return out


def stratified_order(scores, strata):
    """Round-robin across strata, highest-scoring member of each stratum first."""
    order = np.lexsort((-scores, strata))
    s_sorted = strata[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    rank_in_group = np.arange(len(order)) - np.repeat(
        starts, np.diff(np.r_[starts, len(order)]))
    return order[np.argsort(rank_in_group, kind='stable')]


def probability_order(scores, rng):
    """Successive sampling without replacement, weight proportional to score."""
    w = np.clip(scores, 1e-9, None)
    keys = rng.random(len(w)) ** (1.0 / w)      # Efraimidis-Spirakis weighted sampling
    return np.argsort(-keys)


def score(model, X, y, hard=None):
    p = model.predict(X)
    out = dict(MAE=float(mean_absolute_error(y, p)),
               RMSE=float(np.sqrt(mean_squared_error(y, p))),
               R2=float(r2_score(y, p)))
    if hard is not None:
        out['MAE_steep'] = float(mean_absolute_error(y[hard], p[hard]))
        out['RMSE_steep'] = float(np.sqrt(mean_squared_error(y[hard], p[hard])))
    return out


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA).dropna(subset=[TARGET])

    ph_levels = np.sort(df['Input_pH'].unique())
    hfo_levels = np.sort(df['Hfo_s'].unique())
    ph_keep = ph_levels[::2]                 # 3, 5, 7, 9, 11
    hfo_keep = hfo_levels[::2]               # 1e-6, 3.16e-5, 1e-3

    in_initial = df['Input_pH'].isin(ph_keep) & df['Hfo_s'].isin(hfo_keep)
    init = df[in_initial]
    cand = df[~in_initial]
    print(f'initial design {len(init)}   candidates {len(cand)}')

    # common evaluation set, withheld from every strategy
    idx = rng.permutation(len(cand))
    n_eval = int(0.2 * len(cand))
    eval_df = cand.iloc[idx[:n_eval]]
    pool_df = cand.iloc[idx[n_eval:]]
    print(f'evaluation {len(eval_df)}   acquisition pool {len(pool_df)}')

    X_init, y_init = coords_of(init), init[TARGET].values
    X_pool, y_pool = coords_of(pool_df), pool_df[TARGET].values
    X_eval, y_eval = coords_of(eval_df), eval_df[TARGET].values

    design_levels = []
    for ax in AXES:
        lv = np.sort(init[ax].unique().astype(float))
        design_levels.append(np.log10(np.clip(lv, EPS, None)) if ax in LOG_AXES else lv)

    base = fit(X_init, y_init)
    r_eval = resolution_score(base, X_eval, design_levels)
    hard = r_eval >= np.quantile(r_eval, 0.75)
    print(f'steep subset of the evaluation set: {int(hard.sum())} of {len(hard)}')
    results = {'initial_design': dict(n=len(init),
                                      **score(base, X_eval, y_eval, hard))}
    print('initial design  ', results['initial_design'])

    print('scoring the acquisition pool ...')
    r_pool = resolution_score(base, X_pool, design_levels)
    ad_pool = ad_distance(X_init, X_pool)
    order_res = np.argsort(-r_pool)
    order_ad = np.argsort(-ad_pool)

    codes = np.zeros(len(X_pool), dtype=np.int64)
    for d in STRAT_AXES:
        lv = np.unique(X_pool[:, d])
        codes = codes * len(lv) + np.searchsorted(lv, X_pool[:, d])
    order_strat = stratified_order(r_pool, codes)
    order_prob = probability_order(r_pool, np.random.default_rng(SEED))
    print(f'strata used for the coverage-preserving variant: {len(np.unique(codes))}')

    CKPT = os.path.join(HERE, 'adaptive_refinement_results.json')
    if os.path.exists(CKPT):
        prev = json.load(open(CKPT))
        results['budgets'] = {int(k): v for k, v in prev.get('budgets', {}).items()}
    else:
        results['budgets'] = {}
    for N in BUDGETS:
        if N in results['budgets']:
            print(f'budget {N} already done, skipping'); continue
        entry = {}

        for name, order in (('resolution_topN', order_res),
                            ('resolution_strat', order_strat),
                            ('resolution_prob', order_prob),
                            ('ad', order_ad)):
            sel = order[:N]
            m = fit(np.vstack([X_init, X_pool[sel]]),
                    np.concatenate([y_init, y_pool[sel]]))
            entry[name] = score(m, X_eval, y_eval, hard)

        runs = []
        for s in range(N_RANDOM_SEEDS):
            sel = np.random.default_rng(SEED + s).choice(len(X_pool), N, replace=False)
            m = fit(np.vstack([X_init, X_pool[sel]]),
                    np.concatenate([y_init, y_pool[sel]]))
            runs.append(score(m, X_eval, y_eval, hard))
        entry['random'] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        entry['random_sd'] = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}

        base_mae = results['initial_design']['MAE']
        for k in ('resolution_topN', 'resolution_strat', 'resolution_prob',
                  'ad', 'random'):
            entry[k]['MAE_reduction_pct'] = float(
                100 * (base_mae - entry[k]['MAE']) / base_mae)
        results['budgets'][N] = entry
        json.dump(results, open(CKPT, 'w'), indent=2)

        print(f"\nbudget N = {N}  ({100*N/len(cand):.1f}% of the candidate pool)")
        for k in ('random', 'ad', 'resolution_topN', 'resolution_prob',
                  'resolution_strat'):
            e = entry[k]
            print(f"  {k:17s} MAE {e['MAE']:7.3f}  RMSE {e['RMSE']:7.3f}  "
                  f"R2 {e['R2']:7.4f}  | steep MAE {e['MAE_steep']:7.3f}  "
                  f"RMSE {e['RMSE_steep']:7.3f}")

    json.dump(results, open(CKPT, 'w'), indent=2)
    print('\nwritten:', CKPT)


if __name__ == '__main__':
    main()
