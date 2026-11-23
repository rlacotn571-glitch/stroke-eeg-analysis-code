from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
base=out.parent/'stroke_analysis_full'
sys.path.insert(0,str(base))
from summarize_full import ci
t=pd.read_csv(base/'trial_metrics_all_stages.csv')
methods=['before','after','iclabel_after']
pairs=[('after','before'),('iclabel_after','before'),('iclabel_after','after')]
keys=['subject','session','key','trial_index','event_sample','condition','band']
w=t.pivot(index=keys,columns='stage',values='task_vs_baseline_db')[methods]
assert w.shape==(5849*2,3) and np.isfinite(w).all().all() and t.key.nunique()==293
def contrasts(group):
    means=w.groupby(group+['condition']).mean().unstack('condition')
    values=means.xs('MI',axis=1,level='condition')-means.xs('rest',axis=1,level='condition')
    for a,b in pairs:values[a+'-'+b]=values[a]-values[b]
    return values
subject=contrasts(['subject','session','band'])
record=contrasts(['key','band'])
subject.to_csv(out/'subject_contrasts.csv');record.to_csv(out/'record_contrasts.csv')
rows=[]
def summarize(scope,band,endpoint,frame):
    for method in frame:
        s=frame[method]
        mean,lo,hi,n=ci(s)
        rows.append([scope,band,endpoint,method,n,mean,lo,hi,s.min(),s.median(),s.max()])
for band in ['beta','mu']:
    subs=subject.xs(band,level='band')
    for sess,g in subs.groupby('session'):summarize('session',band,sess,g)
    for ep in ['ses-post','ses-follow']:
        change=subs.xs(ep,level='session').subtract(subs.xs('ses-pre',level='session')).dropna()
        assert len(change)==23
        summarize('longitudinal',band,ep,change)
stats=pd.DataFrame(rows,columns=['scope','band','endpoint','method','n','mean','lo','hi','min','median','max'])
stats.to_csv(out/'paired_estimates.csv',index=False)
beta=record.xs('beta',level='band')
target='sub-22_ses-pre_run-04'
for col,series in [('MI-rest',beta['iclabel_after-before']),('MI',w.xs(('MI','beta'),level=('condition','band')).groupby('key').mean().eval('iclabel_after-before'))]:
    print(col,series[target],series.abs().rank(ascending=False)[target],series.abs().quantile([.5,.9,.95]).to_list())
print(stats.query("band=='beta' and scope=='longitudinal'")[['endpoint','method','mean','lo','hi']].round(4).to_string(index=False))
print(stats.query("band=='beta' and method=='iclabel_after-before' and scope=='session'").round(3).to_string(index=False))
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
beta['iclabel_after-before'].hist(ax=ax[0],bins=35,color='#46818a',grid=False)
ax[0].axvline(beta.loc[target,'iclabel_after-before'],color='darkred',label='Participant 22, pre, run 4');ax[0].legend(fontsize=8);ax[0].set(xlabel='ICLabel minus filter only (dB)',ylabel='Recordings',title='Beta MI–rest contrast across 293 recordings')
for j,ep in enumerate(['ses-post','ses-follow']):
    g=stats.query("scope=='longitudinal' and band=='beta' and endpoint==@ep").set_index('method').loc[methods]
    ax[1].errorbar(np.arange(3)+(j-.5)*.12,g['mean'],yerr=[g['mean']-g.lo,g.hi-g['mean']],fmt='o',capsize=3,label=['Post minus pre','Follow-up minus pre'][j])
ax[1].axhline(0,color='gray',lw=1);ax[1].set(xticks=range(3),xticklabels=['Filter only','EOG/ECG ICA','ICLabel ICA'],ylabel='Change (dB), mean and 95% CI',title='Within-participant changes (n = 23)');ax[1].legend(fontsize=8)
fig.savefig(out/'beta_comparison.png',dpi=180)
