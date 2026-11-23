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
assert len(w)==255 and w.switch.sum()==37
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
trials=pd.read_csv(root/'stroke_analysis_full/trial_metrics_all_stages.csv').query("stage=='before' and band=='beta'")
mne.set_log_level('ERROR')
rows=[]
with threadpool_limits(limits=1):
 for k in sorted(selected):
    d=root/'stroke_analysis_full/runs'/k; g=trials[trials.key==k]
    raw=mne.io.read_raw_fif(d/(k+'_before_raw.fif'),preload=True)
    roi=[c for c in ['FCz','Cz','CPz','C1','C2'] if c in raw.ch_names]
    fit=mne.preprocessing.read_ica(d/'iclabel'/(k+'-ica.fif'))
    sc=pd.read_csv(d/'iclabel/component_scores.csv').set_index('component')
    art=sc[['eye','heart','muscle']].max(axis=1)
    def mask(t):return set(sc.index[(art>=t)&(sc.brain<=round(1-t,2))])
    relaxed,strict=mask(.8),mask(.95)
    def contrast(inds):
        fit.exclude=sorted(inds)
        data=raw.copy()
        if inds:fit.apply(data)
        x=data.pick(roi).filter(13,30,method='iir',iir_params=dict(order=4,ftype='butter')).get_data()
        ratios=[]
        for e in g.event_sample-raw.first_samp:
            ratios.append(10*np.log10(np.mean(x[:,e+125:e+1125]**2)/np.mean(x[:,e-875:e-375]**2)))
        v=np.asarray(ratios)
        return v[g.condition=='MI'].mean()-v[g.condition=='rest'].mean()
    full=contrast(relaxed); strict_db=contrast(strict)
    assert np.isclose(strict_db-full,w.loc[k,'Strict']-w.loc[k,'Relaxed'],atol=1e-4)
    for ic in sorted(relaxed-strict):
        change=contrast(relaxed-{ic})-full
        rows.append([k,k in cases.index,ic,art.loc[ic],sc.loc[ic,'predicted_class'],sc.loc[ic,'brain'],change,strict_db-full])
pd.DataFrame(rows,columns=['key','case','component_zero_based','artifact_score','label','brain_score','restore_change_db','restore_all_change_db']).to_csv(out/'restoration.csv',index=False)
fig,(a,b)=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
for flag,color,label in [(False,'#647483','Stable direction'),(True,'#A44627','Switched direction')]:
 vals=np.sort(w[w.switch==flag].span)
 a.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=color,label=f'{label} (n = {len(vals)})')
a.set(xlabel='Range across thresholds (dB)',ylabel='Cumulative fraction of recordings',title='A   All 255 eligible recordings');a.legend(frameon=False,fontsize=8)
for k,r in w[w.switch].iterrows():b.plot(range(3),r[rules].astype(float),color='#9CB7C1' if k not in cases.index else '#AA482B',alpha=.7,lw=.8 if k not in cases.index else 1.7)
b.axhline(0,color='#52626D',ls='--',lw=.8);b.set(xticks=range(3),xticklabels=rules,ylabel='Cleaned minus filter-only contrast (dB)',title='B   The 37 switching recordings')
for ax in (a,b):ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'Figure_S2.pdf',bbox_inches='tight');fig.savefig(out/'Figure_S2.png',dpi=320,bbox_inches='tight')

