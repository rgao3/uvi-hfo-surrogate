import sys,os,json,time,pickle,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import common
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
T0=time.time()
os.makedirs(os.path.join(common.WORK,'models'),exist_ok=True)
CKPT=os.path.join(common.WORK,'final_xgb.json')
df=common.load()
X=df[common.FEATURES].values.astype('float32')
idx=np.arange(len(X))
tr,te=train_test_split(idx,test_size=0.2,random_state=common.RNG)
np.save(os.path.join(common.WORK,'test_idx.npy'),te)
np.save(os.path.join(common.WORK,'train_idx.npy'),tr)
space=dict(n_estimators=[300,400,500], max_depth=[4,5,6,8],
    learning_rate=[0.03,0.05,0.08], subsample=[0.7,0.85,1.0],
    colsample_bytree=[0.7,0.85,1.0], min_child_weight=[1,3,5], reg_lambda=[0.5,1.0,2.0])
inner=KFold(n_splits=3, shuffle=True, random_state=common.RNG)
res=json.load(open(CKPT)) if os.path.exists(CKPT) else {}
for t in common.TARGETS:
    if t in res: continue
    if time.time()-T0>6: print('budget, exit'); break
    y=df[t].values.astype('float32')
    base=XGBRegressor(tree_method='hist',n_jobs=1,random_state=common.RNG,verbosity=0)
    rs=RandomizedSearchCV(base,space,n_iter=6,cv=inner,scoring='r2',n_jobs=2,random_state=common.RNG,refit=True)
    rs.fit(X[tr],y[tr])
    m=rs.best_estimator_
    pte=m.predict(X[te]); ptr=m.predict(X[tr])
    res[t]=dict(best_params=rs.best_params_, cv_r2=float(rs.best_score_),
        train_r2=float(r2_score(y[tr],ptr)),
        test_r2=float(r2_score(y[te],pte)),
        test_rmse=float(mean_squared_error(y[te],pte)**0.5),
        test_mae=float(mean_absolute_error(y[te],pte)))
    pickle.dump(m,open(os.path.join(common.WORK,'models',f'xgb_{t.replace("%","pct")}.pkl'),'wb'))
    np.savez(os.path.join(common.WORK,f'holdout_{t.replace("%","pct")}.npz'),y_true=y[te],y_pred=pte)
    json.dump(res,open(CKPT,'w'),indent=1)
    print(f'{t}: test R2={res[t]["test_r2"]:.4f} rmse={res[t]["test_rmse"]:.4f} ({time.time()-T0:.0f}s)')
print('DONE' if all(t in res for t in common.TARGETS) else 'MORE')
