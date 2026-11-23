from pathlib import Path
import numpy as np,pandas as pd
o=Path(__file__).resolve().parent;d=pd.read_csv(o/'trial_results.csv')
assert len(d)==5849 and not d.duplicated(['key','trial']).any()
from population import ambiguous
d=d[~d.subject.isin(ambiguous)].copy()
print('nonfinite by column:',d.isna().sum().loc[lambda x:x>0].to_dict())
d['eye']=d.eye_after-d.eye_before;d['heart']=d.heart_after-d.heart_before
rows=[];participants=[]
for qc in ['all','screened','complete_screened']:
 for metric in ['beta','eye','heart']:
    t=d.copy()
    if qc!='all':
        t=t[~t.eog_flag]
        if metric=='heart':t=t[~t.ecg_flag]
    if qc=='complete_screened':
        t=t[t.groupby('key').eye.transform(lambda v:v.notna().all())]
        if metric=='heart':t=t[t.groupby('key').heart.transform(lambda v:v.notna().all())]
        cells=t.groupby(['key','condition','stratum']).size().groupby(level=0).size()
        t=t[t.key.isin(cells[cells==4].index)]
    if metric!='beta':t=t[np.isfinite(t[metric])]
    for strata in ['lower','higher']:
        g=t[t.stratum==strata]
        if metric=='beta':
            v=g.groupby(['subject','key','condition']).beta_shift.mean().unstack()
            val=(v.MI-v.rest).dropna()
        else:val=g.groupby(['subject','key'])[metric].mean()
        p=val.groupby('subject').mean()
        if not len(p):continue
        ci=np.quantile(np.random.default_rng(20261120).choice(p,size=(10000,len(p)),replace=True).mean(axis=1),[.025,.975])
        rows.append([qc,metric,strata,len(val),len(p),p.mean(),*ci])
        participants.extend([[qc,metric,strata,k,v] for k,v in p.items()])
r=pd.DataFrame(rows,columns=['subset','metric','stratum','runs','patients','mean','low','high']);r.to_csv(o/'summary.csv',index=False);print(r.round(4).to_string(index=False))
pd.DataFrame(participants,columns=['subset','metric','stratum','subject','value']).to_csv(o/'participants.csv',index=False)

