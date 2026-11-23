from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
s=pd.read_csv(out/'run_results.csv')
rules=['Relaxed','Original','Strict']
assert len(s)==879 and not s.duplicated(['rule','key']).any()
from population import ambiguous, unique_keys
s=s[s.key.isin(unique_keys)].copy()
common=set(s.groupby('key').eligible.all().loc[lambda x:x].index)
persons=[];rows=[]
for i,rule in enumerate(rules):
 g=s[s.rule==rule];e=g[g.eligible]
 p_e=e[~e.subject.isin(ambiguous)]
 f=p_e.groupby('subject').increased.mean()
 n=p_e.groupby('subject').increased.agg(['sum','size'])
 for subject in sorted(set(s.subject)-ambiguous):
    persons.append([rule,subject,int(n.loc[subject,'sum']) if subject in n.index else 0,int(n.loc[subject,'size']) if subject in n.index else 0,f.get(subject,np.nan)])
 boot=np.random.default_rng(20261120).choice(f.values,size=(10000,len(f)),replace=True).mean(1)
 shared=g[g.key.isin(common)]
 rows.append([rule,len(g),g.components_removed.sum(),len(e),int(e.increased.sum()),e.increased.mean(),len(f),f.mean(),*np.quantile(boot,[.025,.975]),g['shift'].abs().median(),len(shared),int(shared.increased.sum())])
r=pd.DataFrame(rows,columns=['rule','runs','removed','eligible','increased','fraction','patients','equal_mean','low','high','median_abs_db','common_runs','common_increased'])
r.to_csv(out/'summary.csv',index=False)
p=pd.DataFrame(persons,columns=['rule','subject','increased','eligible','fraction']);p.to_csv(out/'participants.csv',index=False)
