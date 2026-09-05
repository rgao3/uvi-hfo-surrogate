import numpy as np, pandas as pd, os

WORK=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(WORK, '..'))
DATA_CANDIDATES=[
    os.path.join(ROOT, 'data', 'U_HFO_ML_Dataset_Final.csv'),
    os.path.join(os.path.dirname(WORK), 'data', 'U_HFO_ML_Dataset_Final.csv'),
    os.path.join(os.getcwd(), 'data', 'U_HFO_ML_Dataset_Final.csv'),
    os.path.join(os.getcwd(), '..', 'data', 'U_HFO_ML_Dataset_Final.csv'),
]
DATA=next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[0])
LOG_COLS=['U_initial','Carbonate','NaCl','Ca','Hfo_s','Hfo_w']
BASE_NUM=['Input_pH','U_initial','Carbonate','NaCl','Ca','Hfo_s','Hfo_w']
LOG_FEATS=['log10_'+c for c in LOG_COLS]
FEATURES=BASE_NUM+LOG_FEATS            # Input_pe-free feature set (13 features)
TARGETS=['Ads_%','logKd']
RNG=42
def load():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f'Dataset not found. Update DATA_CANDIDATES in common.py. Tried: {DATA_CANDIDATES}')
    df=pd.read_csv(DATA)
    eps=1e-12
    for c in LOG_COLS:
        df['log10_'+c]=np.log10(df[c].clip(lower=eps))
    return df
