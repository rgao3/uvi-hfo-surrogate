import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

T0=time.time(); BUDGET=8.0
CKPT=os.path.join(common.WORK,'nested_xgb.json')
df=common.load()
X=df[common.FEATURES].values.astype('float32')
N_OUTER=5
outer=KFold(n_splits=N_OUTER, shuffle=True, random_state=common.RNG)
folds={t:list(outer.split(X)) for t in common.TARGETS}  # same splits per target (seed fixed)

space=dict(n_estimators=[200,300,400,500], max_depth=[4,5,6,8],
    learning_rate=[0.03,0.05,0.08,0.1], subsample=[0.7,0.85,1.0],
    colsample_bytree=[0.7,0.85,1.0], min_child_weight=[1,3,5], reg_lambda=[0.5,1.0,2.0])

res=json.load(open(CKPT)) if os.path.exists(CKPT) else {}
def key(t,f): return f'{t}|{f}'
done=set(res.keys())
units=[(t,f) for t in common.TARGETS for f in range(N_OUTER) if key(t,f) not in done]
print(f'pending units: {len(units)}')
for t,f in units:
    if time.time()-T0 > BUDGET:
        print('time budget reached, exiting to checkpoint'); break
    y=df[t].values.astype('float32')
    tr,te=folds[t][f]
    base=XGBRegressor(tree_method='hist', n_jobs=1, random_state=common.RNG, verbosity=0)
    rs=RandomizedSearchCV(base, space, n_iter=8, cv=3, scoring='r2',
        n_jobs=2, random_state=common.RNG, refit=True)
    rs.fit(X[tr], y[tr])
    p=rs.best_estimator_.predict(X[te])
    res[key(t,f)]=dict(target=t, fold=f, best_params=rs.best_params_,
        inner_best_r2=float(rs.best_score_),
        outer_test_r2=float(r2_score(y[te],p)),
        outer_test_rmse=float(mean_squared_error(y[te],p)**0.5),
        outer_test_mae=float(mean_absolute_error(y[te],p)))
    json.dump(res, open(CKPT,'w'), indent=1)
    print(f'done {t} fold{f}: outer R2={res[key(t,f)]["outer_test_r2"]:.4f}  ({time.time()-T0:.0f}s)')
remaining=[(t,f) for t in common.TARGETS for f in range(N_OUTER) if key(t,f) not in res]
print('DONE' if not remaining else f'REMAINING {len(remaining)}')
