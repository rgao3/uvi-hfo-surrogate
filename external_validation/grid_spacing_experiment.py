import pandas as pd, numpy as np, json, time
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
df=pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'U_HFO_ML_Dataset_Final.csv')
LOG=['U_initial','Carbonate','NaCl','Ca','Hfo_s','Hfo_w']
BASE=['Input_pH']+LOG
for c in LOG: df['log10_'+c]=np.log10(df[c].clip(lower=1e-12))
FEAT=BASE+['log10_'+c for c in LOG]
P=dict(n_estimators=500,max_depth=8,learning_rate=0.03,subsample=0.85,
       colsample_bytree=0.85,min_child_weight=5,reg_lambda=2.0,
       tree_method='hist',n_jobs=4,random_state=42,verbosity=0)
y=df['Ads_%'].values.astype('float32'); X=df[FEAT].values.astype('float32')

def run(name,train_mask,test_mask):
    t0=time.time()
    m=XGBRegressor(**P).fit(X[train_mask],y[train_mask])
    p=m.predict(X[test_mask])
    return dict(case=name,n_train=int(train_mask.sum()),n_test=int(test_mask.sum()),
                MAE=float(mean_absolute_error(y[test_mask],p)),
                RMSE=float(np.sqrt(((y[test_mask]-p)**2).mean())),
                R2=float(r2_score(y[test_mask],p)),secs=round(time.time()-t0,1))

res=[]
# reference: random 80/20 split
rng=np.random.RandomState(42); idx=rng.permutation(len(df)); te=np.zeros(len(df),bool); te[idx[:len(df)//5]]=True
res.append(run('reference: random 80/20 split',~te,te))

# pH spacing 2 -> predict the midpoints
ph=df['Input_pH'].values
tr=np.isin(ph,[3,5,7,9,11]); te=np.isin(ph,[4,6,8,10])
res.append(run('pH spacing 2.0, tested at midpoints',tr,te))

# pH: hold out only the steep part of the edge
tr2=np.isin(ph,[3,5,7,9,11]); te2=np.isin(ph,[4])
res.append(run('  ...of which pH 4 alone (steepest)',tr2,te2))

# Hfo_s spacing: 5 log levels -> train on 1,3,5 test on 2,4
hl=np.sort(df['Hfo_s'].unique()); hs=df['Hfo_s'].values
tr=np.isin(hs,hl[[0,2,4]]); te=np.isin(hs,hl[[1,3]])
res.append(run('Hfo_s spacing 1.5 decades, midpoints',tr,te))

# Carbonate: 8 levels -> train alternate
cl=np.sort(df['Carbonate'].unique()); cs=df['Carbonate'].values
tr=np.isin(cs,cl[[0,2,4,6]]); te=np.isin(cs,cl[[1,3,5,7]])
res.append(run('Carbonate spacing doubled, midpoints',tr,te))

# U: 8 levels
ul=np.sort(df['U_initial'].unique()); us=df['U_initial'].values
tr=np.isin(us,ul[[0,2,4,6]]); te=np.isin(us,ul[[1,3,5,7]])
res.append(run('U_initial spacing doubled, midpoints',tr,te))

out=pd.DataFrame(res)
print(out.to_string(index=False))
out.to_csv('modeling/external_validation/grid_spacing_experiment.csv',index=False)
