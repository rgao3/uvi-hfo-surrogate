import sys,os,json,time,pickle,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import common, shap
T0=time.time()
X=np.load(os.path.join(common.WORK,'X.npy'))
te=np.load(os.path.join(common.WORK,'test_idx.npy'))
rng=np.random.RandomState(common.RNG)
samp=rng.choice(te, size=5000, replace=False)
np.save(os.path.join(common.WORK,'shap_sample_idx.npy'),samp)
done=[]
for t in common.TARGETS:
    tag=t.replace('%','pct')
    out=os.path.join(common.WORK,f'shap_{tag}.npz')
    if os.path.exists(out): done.append(t); continue
    if time.time()-T0>8: break
    m=pickle.load(open(os.path.join(common.WORK,'models',f'xgb_{tag}.pkl'),'rb'))
    expl=shap.TreeExplainer(m)
    sv=expl.shap_values(X[samp])
    # interaction values on a smaller subset (pH x carbonate focus) for speed
    inter=expl.shap_interaction_values(X[samp[:2500]])
    np.savez(out, shap=sv, base=np.array(expl.expected_value).ravel(),
             Xs=X[samp], inter=inter, feat=np.array(common.FEATURES))
    print(f'{t}: SHAP {sv.shape} inter {inter.shape} ({time.time()-T0:.0f}s)')
    done.append(t)
print('DONE' if len(done)==len(common.TARGETS) else 'MORE')
