import sys,os,json,time,pickle,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import common, shap, xgboost as xgb
X=np.load(os.path.join(common.WORK,'X.npy'))
te=np.load(os.path.join(common.WORK,'test_idx.npy'))
rng=np.random.RandomState(common.RNG)
samp=rng.choice(te, size=900, replace=False)
np.save(os.path.join(common.WORK,'shap_sample_idx.npy'),samp)
for t in common.TARGETS:
    tag=t.replace('%','pct'); out=os.path.join(common.WORK,f'shap_{tag}.npz')
    if os.path.exists(out): continue
    m=pickle.load(open(os.path.join(common.WORK,'models',f'xgb_{tag}.pkl'),'rb'))
    expl=shap.TreeExplainer(m)
    sv=expl.shap_values(X[samp])
    np.savez(out, shap=sv, base=float(np.array(expl.expected_value).ravel()[0]),
             Xs=X[samp], feat=np.array(common.FEATURES))
    print(f'{t}: SHAP {sv.shape} base={np.array(expl.expected_value).ravel()[0]:.3f}')
print('DONE')
