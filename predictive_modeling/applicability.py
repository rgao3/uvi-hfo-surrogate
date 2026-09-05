import sys,os,json,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import common
from sklearn.neighbors import NearestNeighbors
X=np.load(os.path.join(common.WORK,'X.npy'))
sp=np.load(os.path.join(common.WORK,'conformal_splits.npz'))
tr,test=sp['tr'],sp['test']
# standardize on training features
mu=X[tr].mean(0); sd=X[tr].std(0)+1e-9
Xs=(X-mu)/sd
nn=NearestNeighbors(n_neighbors=6).fit(Xs[tr])
# AD distance = mean distance to k nearest training points
dtest,_=nn.kneighbors(Xs[test], n_neighbors=5); ad_test=dtest.mean(1)
dtr,_=nn.kneighbors(Xs[tr], n_neighbors=6); ad_tr=dtr[:,1:].mean(1)  # exclude self
thr=np.quantile(ad_tr,0.95)
# failed 25 cases
FAILED=os.path.join(os.path.dirname(common.DATA),'U_HFO_ML_Dataset_Final_missing25_failed_cases.csv')
fdf=pd.read_csv(FAILED)
for c in common.LOG_COLS: fdf['log10_'+c]=np.log10(fdf[c].clip(lower=1e-12))
Xf=(fdf[common.FEATURES].values.astype('float64')-mu)/sd
df_,_=nn.kneighbors(Xf,n_neighbors=5); ad_fail=df_.mean(1)
# error vs AD decile (Ads_% + logKd), using conformal arrays
out={'threshold_p95':float(thr),
     'failed_ad_min':float(ad_fail.min()),'failed_ad_median':float(np.median(ad_fail)),'failed_ad_max':float(ad_fail.max()),
     'failed_pct_of_train_dist':[float((ad_tr<v).mean()*100) for v in [ad_fail.min(),np.median(ad_fail),ad_fail.max()]],
     'failed_frac_above_thr':float((ad_fail>thr).mean()),
     'test_frac_above_thr':float((ad_test>thr).mean())}
decile={}
for tag in ['Ads_pct','logKd']:
    a=np.load(os.path.join(common.WORK,f'conformal_arrays_{tag}.npz'))
    resid=np.abs(a['y']-a['pred'])
    q=np.quantile(ad_test,[0,.2,.4,.6,.8,1.0])
    rows=[]
    for i in range(5):
        m=(ad_test>=q[i])&(ad_test<=q[i+1]) if i==4 else (ad_test>=q[i])&(ad_test<q[i+1])
        rows.append(dict(bin=i+1,ad_lo=float(q[i]),ad_hi=float(q[i+1]),n=int(m.sum()),
            rmse=float(np.sqrt((resid[m]**2).mean())),mae=float(resid[m].mean())))
    decile[tag]=rows
    # in vs out of domain rmse
    ind=ad_test<=thr; ood=ad_test>thr
    out[tag+'_rmse_in_domain']=float(np.sqrt((resid[ind]**2).mean()))
    out[tag+'_rmse_out_domain']=float(np.sqrt((resid[ood]**2).mean())) if ood.sum()>0 else None
out['decile']=decile
np.savez(os.path.join(common.WORK,'ad_arrays.npz'),ad_test=ad_test,ad_tr=ad_tr,ad_fail=ad_fail,thr=thr)
json.dump(out,open(os.path.join(common.WORK,'applicability.json'),'w'),indent=1)
print('threshold p95=%.3f'%thr)
print('failed AD: min=%.3f median=%.3f max=%.3f'%(ad_fail.min(),np.median(ad_fail),ad_fail.max()))
print('failed frac above thr=%.2f | test frac above thr=%.3f'%(out['failed_frac_above_thr'],out['test_frac_above_thr']))
for tag in ['Ads_pct','logKd']:
    print(tag,'RMSE in-domain=%.3f out-domain=%.3f'%(out[tag+'_rmse_in_domain'],out[tag+'_rmse_out_domain']))
print('DONE')
