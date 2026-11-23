from pathlib import Path
import numpy as np
import pandas as pd
import mne
from threadpoolctl import threadpool_limits
mne.set_log_level('ERROR')
base=Path('outputs/stroke_analysis_full')
out=Path('outputs/BSPC/sensitivity')
rules=[('Relaxed',.80,.20),('Original',.90,.10),('Strict',.95,.05)]
cols=['MI_task','MI_baseline','rest_task','rest_baseline']
ref=pd.read_csv('outputs/BSPC/Figure_1_source.csv').set_index('key')
trials=pd.read_csv(base/'trial_metrics_all_stages.csv').query("stage=='before' and band=='beta'")
assert len(trials)==5849
def powers(raw,roi,events):
    x=raw.copy().pick(roi).filter(13,30,method='iir',iir_params=dict(order=4,ftype='butter')).get_data()
    return np.array([[np.mean(x[:,j+l:j+h]**2) for l,h in [(125,1125),(-875,-375)]] for j in events])
rows=[]
with threadpool_limits(limits=1):
 for key,g in trials.groupby('key'):
    folder=base/'runs'/key
    raw=mne.io.read_raw_fif(folder/(key+'_before_raw.fif'),preload=True)
    roi=[c for c in ['FCz','Cz','CPz','C1','C2'] if c in raw.ch_names]
    fit=mne.preprocessing.read_ica(folder/'iclabel'/(key+'-ica.fif'))
    sc=pd.read_csv(folder/'iclabel/component_scores.csv')
    assert sc.component.tolist()==list(range(fit.n_components_))
    original=tuple(fit.exclude)
    events=g.event_sample.to_numpy()-raw.first_samp
    before=powers(raw,roi,events)
    assert np.allclose(before*1e12,g[['task_power_uV2','baseline_power_uV2']],rtol=1e-4)
    cache={():np.zeros(4)}
    for label,art,brain in rules:
        inds=tuple(sc.loc[(sc[['eye','heart','muscle']].max(axis=1)>=art)&(sc.brain<=brain),'component'])
        if label=='Original':assert set(inds)==set(original)
        if inds not in cache:
            fit.exclude=list(inds)
            change=10*np.log10(powers(fit.apply(raw.copy()),roi,events)/before)
            cache[inds]=np.r_[change[g.condition=='MI'].mean(0),change[g.condition=='rest'].mean(0)]*[1,-1,-1,1]
        vals=cache[inds]
        if label=='Original':
            saved=ref.loc[key,cols].to_numpy(dtype=float)
            assert np.max(np.abs(saved-vals))<1e-4,(key,vals,saved)
            vals=saved
        # Reuse the published terms for identical component projections.
        elif inds==original:vals=ref.loc[key,cols].to_numpy(dtype=float)
        net=vals.sum()
        rows.append([key,g.subject.iloc[0],g.session.iloc[0],label,len(inds),*vals,net,all(vals*np.array([1,-1,-1,1]) < -1e-8),net>1e-8])
    if len(rows)%150==0:print(len(rows)//3,flush=True)
pd.DataFrame(rows,columns=['key','subject','session','rule','components_removed',*cols,'shift','eligible','increased']).to_csv(out/'run_results.csv',index=False)
