import sys,os,json,time,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import common
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
T0=time.time(); BUDGET=35
X=np.load(os.path.join(common.WORK,'X.npy'))
finalx=json.load(open(os.path.join(common.WORK,'final_xgb.json')))
COLS={'pH':0,'carbonate':2,'HFO':5}
CKPT=os.path.join(common.WORK,'region_holdout.json')
res=json.load(open(CKPT)) if os.path.exists(CKPT) else {}
units=[]
for t in common.TARGETS:
    for typ,ci in COLS.items():
        for lv in np.unique(X[:,ci]):
            k=f'{t}|{typ}|{lv:.6g}'
            if k not in res: units.append((t,typ,ci,lv,k))
print('pending',len(units))
for t,typ,ci,lv,k in units:
    if time.time()-T0>BUDGET: print('budget'); break
    tag=t.replace('%','pct'); y=np.load(os.path.join(common.WORK,'y_'+tag+'.npy'))
    te=np.isclose(X[:,ci],lv); tr=~te
    p={kk:v for kk,v in finalx[t]['best_params'].items()}
    m=XGBRegressor(tree_method='hist',n_jobs=2,random_state=common.RNG,verbosity=0,**p).fit(X[tr],y[tr])
    pr=m.predict(X[te])
    res[k]=dict(target=t,type=typ,level=float(lv),n=int(te.sum()),
        r2=float(r2_score(y[te],pr)),rmse=float(mean_squared_error(y[te],pr)**0.5),
        mae=float(mean_absolute_error(y[te],pr)))
    json.dump(res,open(CKPT,'w'),indent=1)
rem=[1 for t in common.TARGETS for typ,ci in COLS.items() for lv in np.unique(X[:,ci]) if f'{t}|{typ}|{lv:.6g}' not in res]
print('DONE' if not rem else f'REMAINING {len(rem)}')
