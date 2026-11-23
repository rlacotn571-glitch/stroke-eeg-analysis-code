from pathlib import Path
import sys,json
import numpy as np,pandas as pd
out=Path(__file__).resolve().parent
base=out.parent/'stroke_analysis_full'
sys.path.insert(0,str(base))
import pipeline as p
from summarize_full import ci
t=pd.read_csv(base/'trial_metrics_all_stages.csv')
rows=[];counts=[]
with p.threadpool_limits(limits=2):
 for key,g in t.groupby('key'):
    d=base/'runs'/key
    raw=p.mne.io.read_raw_fif(d/(key+'_before_raw.fif'),preload=True)
    fit=p.mne.preprocessing.read_ica(d/'iclabel'/(key+'-ica.fif'))
    sources=fit.get_sources(raw).get_data();ref=p.references(raw)
    eye=np.abs(p.correlations(p.bandfilter(sources,1,15,250),p.bandfilter(ref[:2],1,15,250))).max(axis=1)
    heart=np.abs(p.correlations(p.bandfilter(sources,5,25,250),p.bandfilter(ref[2:],5,25,250)))[:,0]
    orig=fit.exclude.copy();fit.exclude=np.where((eye>=.4)|(heart>=.3))[0].tolist()
    counts.append([key,len(orig),len(fit.exclude),len(set(orig)&set(fit.exclude))])
    if fit.exclude:fit.apply(raw)
    roi=raw.pick([c for c in p.CFG['roi'] if c in raw.ch_names])
    for band,(lo,hi) in p.CFG['bands'].items():
        x=roi.copy().filter(lo,hi,method='iir',iir_params=dict(order=4,ftype='butter')).get_data()
        for e in g.query("stage=='before' and band==@band").itertuples():
            j=e.event_sample
            v=10*np.log10(np.mean(x[:,j+125:j+1125]**2)/np.mean(x[:,j-875:j-375]**2))
            if not fit.exclude:assert abs(v-e.task_vs_baseline_db)<1e-5
            rows.append([e.subject,e.session,key,e.trial_index,e.condition,band,v])
    if len(counts)%50==0:print(len(counts),flush=True)
new=pd.DataFrame(rows,columns=['subject','session','key','trial_index','condition','band','task_vs_baseline_db']).assign(stage='samefit_corr')
assert len(new)==11698 and len(counts)==293
new.to_csv(out/'samefit_trials.csv',index=False)
pd.DataFrame(counts,columns=['key','iclabel','correlation','overlap']).to_csv(out/'selection.csv',index=False)
t=pd.concat([t,new]);ids=['subject','session','key','trial_index','condition','band']
w=t.pivot(index=ids,columns='stage',values='task_vs_baseline_db')
assert np.isfinite(w).all().all()
g=w.groupby(['subject','session','band','condition']).mean().unstack()
g=g.xs('MI',axis=1,level='condition')-g.xs('rest',axis=1,level='condition')
g['selection_difference']=g.iclabel_after-g.samefit_corr
g['fit_difference']=g.samefit_corr-g.after
g.to_csv(out/'subject_estimates.csv')
results=[]
for b in ['mu','beta']:
    z=g.xs(b,level='band')
    for ep in ['ses-post','ses-follow']:
        delta=(z.xs(ep,level='session')-z.xs('ses-pre',level='session')).dropna()
        assert len(delta)==23
        for c in delta:results.append([b,ep,c,*ci(delta[c])])
r=pd.DataFrame(results,columns=['band','endpoint','method','mean','low','high','n'])
r.to_csv(out/'longitudinal.csv',index=False);print(r.query("band=='beta'").round(4).to_string(index=False))
