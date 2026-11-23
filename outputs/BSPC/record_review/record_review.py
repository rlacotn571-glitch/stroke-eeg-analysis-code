from pathlib import Path
import numpy as np,pandas as pd,mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
root=Path('outputs')
out=root/'BSPC/record_review'
rules=['Relaxed','Original','Strict']
terms=['MI_task','MI_baseline','rest_task','rest_baseline']
s=pd.read_csv(root/'BSPC/sensitivity/run_results.csv').set_index(['key','rule'])
from population import unique_keys
s=s[s.index.get_level_values('key').isin(unique_keys)]
w=s['shift'].unstack()[rules].loc[s.eligible.unstack().all(axis=1)]
meta=s.xs('Original',level='rule').loc[w.index,['subject','session']]
w['span']=w.max(axis=1)-w.min(axis=1)
w['switch']=s.increased.unstack().loc[w.index].nunique(axis=1)>1
w=w.join(meta)
prior=pd.read_csv(root/'stroke_review_four/run_review_metrics.csv').key
w['prior_review']=w.index.isin(prior)
for k,r in w.iterrows():
    lo=r[rules].astype(float).idxmin();hi=r[rules].astype(float).idxmax()
    v=s.loc[(k,hi),terms].astype(float)-s.loc[(k,lo),terms].astype(float)
    assert np.isclose(v.sum(),r.span)
    w.loc[k,'min_rule']=lo;w.loc[k,'max_rule']=hi
    for c in terms:w.loc[k,c+'_difference']=v[c]
assert len(w)==251 and w.switch.sum()==37
w.to_csv(out/'record_spans.csv')
print(w.groupby('switch').span.describe().to_string(),flush=True)
print('overlap',w.query('switch and prior_review').index.tolist(),flush=True)
# Match a typical non-switching recording within the participant and visit to each of the three largest switching cases.
cases=w[w.switch].nlargest(3,'span')
selected=set(cases.index);matches=[]
for k,r in cases.iterrows():
 pool=w[(~w.switch)&(w.subject==r.subject)&(w.session==r.session)]
 control=(pool.span-pool.span.median()).abs().idxmin() if len(pool) else None
 if control:selected.add(control)
 matches.append([k,control])
pd.DataFrame(matches,columns=['case','comparison']).to_csv(out/'matches.csv',index=False)
print(w.loc[sorted(selected),rules+['span']].to_string(),flush=True)
