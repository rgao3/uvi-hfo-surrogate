# Resolution criterion — supporting experiments

Analyses that support the resolution-aware reliability criterion in the manuscript.
Kept separate from `predictive_modeling/` (surrogate development) and `external_validation/`
(comparison against published measurements) because this work concerns the
criterion itself rather than the surrogate or the literature comparison.

Everything here is intended to be part of the public code release.

---

## `model_class_experiment.py`

**Question.** The staircase behaviour that makes the resolution criterion visible
in the XGBoost surrogate is a consequence of trees being piecewise constant
between splits. Does the criterion still work for a smooth surrogate on the same
design, or is it a tree artefact?

**Design.** Because the simulation database is a complete factorial grid, two
interpolators can be built on the *identical* coarsened design and evaluated on the
*identical* held-out levels:

| | model | role |
|---|---|---|
| A | nearest-node | piecewise constant — the idealised tree ensemble |
| B | multilinear | piecewise linear and continuous — the smooth comparator |

Design, held-out set, reference values and applicability-domain definition are all
held fixed, so any difference is attributable to the model class alone. This is a
cleaner contrast than XGBoost-versus-Gaussian-process, where training procedure,
hyperparameters and feature representation would all differ simultaneously.

Two forms of the score are computed:

```
R1 = max_a | f(x | a -> u_a) - f(x | a -> l_a) |            first difference (manuscript Eq. 1)
R2 = max_a | f(x | a -> u_a) + f(x | a -> l_a) - 2 f(x) |   second difference
```

**Validity check on the idealisation.** On the coarsened pH axis the nearest-node
model reproduces the trained XGBoost surrogate closely — MAE 11.48 against 11.78,
Spearman rho 0.901 against 0.881 (`external_validation/resolution_criterion_validation.csv`).
The idealisation is therefore faithful and the comparison is meaningful.

**Results** (`model_class_experiment_results.json`):

| Coarsened axis | Model | MAE | rho(R1) | rho(R2) | rho(AD) |
|---|---|---|---|---|---|
| pH, unit → 2-unit | nearest | 11.48 | **0.901** | 0.901 | −0.001 |
| pH, unit → 2-unit | multilinear | 5.99 | **0.808** | 0.653 | −0.027 |
| Hfo_s, 0.75 → 1.5 decade | nearest | 17.68 | **0.889** | 0.889 | 0.012 |
| Hfo_s, 0.75 → 1.5 decade | multilinear | 7.59 | **0.737** | 0.515 | 0.001 |

**Four conclusions.**

1. **The criterion is not a tree artefact.** For the smooth interpolator the rank
   correlation falls from ~0.90 to 0.74–0.81, but remains strongly informative and
   two orders of magnitude above the applicability domain.
2. **The first-difference form is the right definition for both model classes.**
   R2 was included on the expectation that a piecewise-linear model's error would
   be governed by curvature rather than slope. It is not: R2 is identical to R1 for
   the piecewise-constant model and clearly *worse* for the smooth one (0.653 and
   0.515 against 0.808 and 0.737). Manuscript Eq. 1 needs no modification.
3. **Smoothing halves the error but does not remove it.** MAE falls from 11.5 to
   6.0 and from 17.7 to 7.6. A smooth surrogate on a coarse design is better but
   still badly wrong where the design is coarse, which is the point: design
   resolution, not model class, is the binding constraint.
4. **The applicability domain is uninformative for both model classes**, with rank
   correlations between −0.03 and +0.01. Because this uses a 6-feature
   log-transformed AD rather than the 13-feature version used elsewhere in the
   project, it also serves as a check that the AD result is not an artefact of one
   particular AD definition.

**Reproduce:**

```
python model_class_experiment.py
```

Pure numpy and pandas; no scikit-learn, scipy or xgboost dependency, so it runs
from the released dataset alone. Fixed seed (42). Runtime approximately one minute.

---

## `adaptive_refinement_experiment.py`

**Question.** Given budget for N more simulator runs, does the resolution score
allocate them better than the alternatives? Tested at equal simulator cost, with
no new PHREEQC runs — refinement is simulated by revealing points that already
exist in the database.

**Setup.** Initial design: pH at 2-unit spacing, Hfo_s at 1.5-decade spacing, other
axes complete (28,800 points). A fixed random 20% of the remaining candidates
(11,515 points) is the common evaluation set; the other 46,060 form the acquisition
pool. Surrogate: `HistGradientBoostingRegressor`, one of the three algorithms
compared in the manuscript, refitted from scratch for each strategy and budget.

Because the argument concerns *where the design is under-resolved*, results are
reported both over the whole evaluation set and over its steepest quartile
(the 2,879 points with the highest resolution score under the initial design).

**Results** (MAE in adsorption percentage points; `adaptive_refinement_results.json`):

| Budget | | random | ad | res_topN | res_prob | res_strat |
|---|---|---|---|---|---|---|
| — | initial design | 16.75 (steep 36.41) | | | | |
| 2,000 | all | **4.435** | 7.423 | 8.441 | 4.911 | 7.611 |
| | steep | 5.182 | 13.394 | 8.789 | **4.739** | 5.662 |
| 5,000 | all | **3.776** | 6.040 | 8.495 | 4.080 | 7.124 |
| | steep | 4.229 | 11.023 | 6.826 | **3.353** | 4.322 |
| 10,000 | all | **3.228** | 6.039 | 6.292 | 3.430 | 5.626 |
| | steep | 3.544 | 11.291 | **2.583** | 2.876 | 2.779 |
| 20,000 | all | **2.992** | 4.306 | 4.358 | 3.139 | 4.403 |
| | steep | 3.074 | 6.512 | **2.338** | 2.725 | 2.534 |

**Four conclusions.**

1. **Uniform refinement wins on the uniform average, at every budget.** Greedy
   top-N selection on the resolution score is much worse (8.4 against 4.4 at
   N = 2,000). A per-prediction diagnostic is not an acquisition function: greedy
   selection collapses onto the adsorption edge and forfeits the coverage that a
   space-filling design provides. This is the standard failure mode of pure
   uncertainty sampling in active learning, and it applies here too.
2. **On the region the criterion identifies as under-resolved, resolution-guided
   refinement wins at every budget.** At N = 10,000 the steep-quartile MAE is 2.58
   against 3.54 for uniform refinement, 27% lower; at N = 20,000, 2.34 against
   3.07. The score does know where the simulations are needed — it simply does not
   know that coverage elsewhere still has to be maintained.
3. **Score-weighted sampling is the usable compromise.** Sampling without
   replacement with probability proportional to R1 stays within 6% of uniform
   refinement on the overall MAE while improving the steep-region MAE by 19–23%,
   and gives the best RMSE and R² overall at N >= 10,000 (RMSE 4.79 against 4.92
   at N = 20,000).
4. **Applicability-domain-guided refinement is worse than random everywhere, and
   actively harmful where it matters.** Its steep-region MAE is 11.0–11.3 at
   N = 5,000–10,000 against 3.5–4.2 for uniform refinement — two to three times
   worse. Buying simulations at the points farthest from the existing design steers
   the budget towards the sparse corners of the domain and away from the response
   feature that limits accuracy.

**Caveat, stated because it determines the reading.** The overall evaluation set is
uniform over the candidate space. If the intended query distribution concentrates
near the adsorption edge — which is the usual situation, since that is where the
answer changes — the steep-region column is the relevant one and the ranking
reverses. Neither column is the "right" one in isolation; the design recommendation
depends on where predictions will actually be made, and the paper should say so.

**Reproduce:**

```
python adaptive_refinement_experiment.py
```

Checkpoints after each budget, so an interrupted run resumes. Fixed seed (42).
Requires scikit-learn. Runtime approximately five minutes.

---

## Not in this directory

Manuscript preparation scripts (redlining, comment insertion, document assembly)
are deliberately **not** part of the repository. They operate on Word files, not on
data, and produce no result that appears in the paper.
