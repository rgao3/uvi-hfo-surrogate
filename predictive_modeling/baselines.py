import sys,os,json,time,pickle,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import common
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
T0=time.time()
X=np.load(os.path.join(common.WORK,'X.npy'))
tr=np.load(os.path.join(common.WORK,'train_idx.npy')); te=np.load(os.path.join(common.WORK,'test_idx.npy'))
CKPT=os.path.join(common.WORK,'baselines.json')
res=json.load(open(CKPT)) if os.path.exists(CKPT) else {}
def metr(y,p): return dict(r2=float(r2_score(y,p)),rmse=float(mean_squared_error(y,p)**0.5),mae=float(mean_absolute_error(y,p)))
for t in common.TARGETS:
    if t in res: continue
    if time.time()-T0>4: print('budget'); break
    y=np.load(os.path.join(common.WORK,'y_'+t.replace('%','pct')+'.npy'))
    rf=RandomForestRegressor(n_estimators=120,max_depth=20,max_samples=0.5,n_jobs=1,random_state=common.RNG).fit(X[tr],y[tr])
    hg=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=8,random_state=common.RNG).fit(X[tr],y[tr])
    res[t]=dict(RandomForest=dict(train=metr(y[tr],rf.predict(X[tr])),test=metr(y[te],rf.predict(X[te]))),
                HistGB=dict(train=metr(y[tr],hg.predict(X[tr])),test=metr(y[te],hg.predict(X[te]))))
    json.dump(res,open(CKPT,'w'),indent=1)
    print(f'{t}: RF test R2={res[t]["RandomForest"]["test"]["r2"]:.4f} | HistGB test R2={res[t]["HistGB"]["test"]["r2"]:.4f} ({time.time()-T0:.0f}s)')
print('DONE' if all(t in res for t in common.TARGETS) else 'MORE')
