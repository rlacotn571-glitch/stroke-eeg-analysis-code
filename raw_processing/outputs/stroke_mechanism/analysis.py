from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
base=out.parent
t=pd.read_csv(base/'stroke_analysis_full/trial_metrics_all_stages.csv').query("band=='beta'")
ref=pd.read_csv(base/'stroke_manuscript/record_contrasts.csv').query("band=='beta'").set_index('key')
methods=['before','after','iclabel_after']
idx=['subject','session','key','trial_index','event_sample','condition']
powers=['task_power_uV2','baseline_power_uV2']
w=t.pivot(index=idx,columns='stage',values=powers)
assert len(w)==5849 and (w>0).all().all() and np.isfinite(w).all().all()
target='sub-22_ses-pre_run-04'
allparts=[]; scales=[]; patients=[]; influence=[]
for new,old in [('iclabel_after','before'),('after','before'),('iclabel_after','after')]:
    label=new+'-'+old
    logs=pd.DataFrame({col:10*np.log10(w[col][new]/w[col][old]) for col in powers})
    avg=logs.groupby(['subject','session','key','condition']).mean().unstack('condition')
    parts=pd.DataFrame({'MI_task':avg[powers[0],'MI'],'MI_baseline':-avg[powers[1],'MI'],'rest_task':-avg[powers[0],'rest'],'rest_baseline':avg[powers[1],'rest']})
    parts['delta']=parts.sum(axis=1)
    parts=parts.reset_index().set_index('key')
    assert len(parts)==293 and np.allclose(parts.delta,ref.loc[parts.index,label],atol=1e-9,rtol=0)
    parts['pair']=label;allparts.append(parts)
    for (s,v),g in parts.groupby(['subject','session']):
        distances=[abs(a-b) for a,b in combinations(ref.loc[g.index,old],2)]
        if distances:
            scales.append(dict(pair=label,subject=s,session=v,n=len(g),median_cleaning_shift=g.delta.abs().median(),median_repeat_difference=np.median(distances)))
    trial_delta=logs[powers[0]]-logs[powers[1]]
    for k,g in trial_delta.reset_index(name='d').groupby('key'):
        estimates=[]
        for cond,h in g.groupby('condition'):
            sign=1 if cond=='MI' else -1
            estimates.extend(parts.loc[k,'delta']+sign*(h.d.mean()-h.d)/(len(h)-1))
        influence.append(dict(pair=label,key=k,min_after_omission=min(estimates),max_after_omission=max(estimates)))
parts=pd.concat(allparts);parts.to_csv(out/'four_terms.csv')
scale=pd.DataFrame(scales);scale.to_csv(out/'repeat_scale.csv',index=False)
inf=pd.DataFrame(influence);inf.to_csv(out/'leave_one_trial_out.csv',index=False)
main=parts.query("pair=='iclabel_after-before'")
a=main.query("subject=='sub-22' and session=='ses-pre'")
print(a.to_string());print(inf[(inf.key==target)&(inf.pair=='iclabel_after-before')])
for pair,g in scale.groupby('pair'):
    s=g.groupby('subject')[['median_cleaning_shift','median_repeat_difference']].mean()
    print(pair,len(g),len(s),s.median().to_dict(),sum(g.median_cleaning_shift>g.median_repeat_difference))
fig,ax=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
a.iloc[:,2:6].plot.bar(ax=ax[0]);ax[0].set_xticklabels([k[-2:] for k in a.index],rotation=0);ax[0].set(xlabel='Pre-training recording, participant 22',ylabel='Signed contribution (dB)');ax[0].axhline(0,color='gray',lw=.7);ax[0].legend(fontsize=7)
g=scale.query("pair=='iclabel_after-before'")
ax[1].scatter(g.median_repeat_difference,g.median_cleaning_shift,alpha=.65);lim=max(g.median_repeat_difference.max(),g.median_cleaning_shift.max());ax[1].plot([0,lim],[0,lim],'--',color='gray');ax[1].set(xlabel='Median between-recording difference (dB)',ylabel='Median preprocessing shift (dB)',title='Each point = one participant at one visit')
fig.savefig(out/'mechanism.png',dpi=180)

