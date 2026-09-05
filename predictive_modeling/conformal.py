import sys,os,json,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import common
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
X=np.load(os.path.join(common.WORK,'X.npy'))
finalx=json.load(open(os.path.join(common.WORK,'final_xgb.json')))
ipH=common.FEATURES.index('Input_pH')
idx=np.arange(len(X))
# 60/20/20 proper-train / calibration / test
tr,rest=train_test_split(idx,test_size=0.4,random_state=common.RNG)
cal,test=train_test_split(rest,test_size=0.5,random_state=common.RNG)
np.savez(os.path.join(common.WORK,'conformal_splits.npz'),tr=tr,cal=cal,test=test)
ALPHAS=[0.20,0.10,0.05]
res={}
for t in common.TARGETS:
    tag=t.replace('%','pct')
    y=np.load(os.path.join(common.WORK,'y_'+tag+'.npy'))
    p={k:v for k,v in finalx[t]['best_params'].items()}
    m=XGBRegressor(tree_method='hist',n_jobs=2,random_state=common.RNG,verbosity=0,**p).fit(X[tr],y[tr])
    pc=m.predict(X[cal]); pt=m.predict(X[test])
    scores=np.abs(y[cal]-pc)
    pH_cal=X[cal,ipH]; pH_test=X[test,ipH]
    tinfo={'alphas':{}}
    for a in ALPHAS:
        n=len(scores); q=np.quantile(scores, np.ceil((n+1)*(1-a))/n, method='higher')
        lo,hi=pt-q,pt+q
        cov=float(np.mean((y[test]>=lo)&(y[test]<=hi))); width=float(2*q)
        # Mondrian by pH band
        bands=[('<=5',pH_test<=5),('6-7',(pH_test>=6)&(pH_test<=7)),('>=8',pH_test>=8)]
        mond={}; lo_m=np.empty_like(pt); hi_m=np.empty_like(pt)
        for lab,_ in bands: pass
        qb={}
        for lab,mask_t in bands:
            if lab=='<=5': mcal=pH_cal<=5
            elif lab=='6-7': mcal=(pH_cal>=6)&(pH_cal<=7)
            else: mcal=pH_cal>=8
            sc=scores[mcal]; nb=len(sc); qq=np.quantile(sc,np.ceil((nb+1)*(1-a))/nb,method='higher')
            qb[lab]=float(qq); lo_m[mask_t]=pt[mask_t]-qq; hi_m[mask_t]=pt[mask_t]+qq
            mond[lab]=dict(coverage=float(np.mean((y[test][mask_t]>=pt[mask_t]-qq)&(y[test][mask_t]<=pt[mask_t]+qq))),
                           width=float(2*qq), n=int(mask_t.sum()))
        cov_m=float(np.mean((y[test]>=lo_m)&(y[test]<=hi_m)))
        tinfo['alphas'][str(a)]=dict(nominal=round(1-a,2), global_coverage=cov, global_width=width,
                                     mondrian_coverage=cov_m, mondrian_width=float(np.mean(hi_m-lo_m)),
                                     mondrian_by_band=mond)
    # quantile-GB comparison at 90%
    lo_gb=HistGradientBoostingRegressor(loss='quantile',quantile=0.05,max_iter=300,learning_rate=0.05,max_depth=8,random_state=common.RNG).fit(X[tr],y[tr]).predict(X[test])
    hi_gb=HistGradientBoostingRegressor(loss='quantile',quantile=0.95,max_iter=300,learning_rate=0.05,max_depth=8,random_state=common.RNG).fit(X[tr],y[tr]).predict(X[test])
    hi_gb=np.maximum(hi_gb,lo_gb)
    tinfo['quantile_gb_90']=dict(coverage=float(np.mean((y[test]>=lo_gb)&(y[test]<=hi_gb))), width=float(np.mean(hi_gb-lo_gb)))
    # save test arrays for plotting (alpha=0.10 global)
    n=len(scores); q=np.quantile(scores,np.ceil((n+1)*0.90)/n,method='higher')
    np.savez(os.path.join(common.WORK,f'conformal_arrays_{tag}.npz'),
             y=y[test],pred=pt,q=q,pH=pH_test,scores_cal=scores)
    res[t]=tinfo
    print(f'{t}: 90% global cov={tinfo["alphas"]["0.1"]["global_coverage"]:.3f} width={tinfo["alphas"]["0.1"]["global_width"]:.3f} | mondrian cov={tinfo["alphas"]["0.1"]["mondrian_coverage"]:.3f}')
json.dump(res,open(os.path.join(common.WORK,'conformal.json'),'w'),indent=1)
print('DONE')
