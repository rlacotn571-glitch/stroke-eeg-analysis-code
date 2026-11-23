from pathlib import Path
import numpy as np,pandas as pd,mne
from scipy.signal import butter,sosfiltfilt
from threadpoolctl import threadpool_limits
mne.set_log_level('ERROR')
src=Path('outputs/stroke_analysis_full')
out=Path('outputs/BSPC/auxiliary_check')
out.mkdir(exist_ok=True,parents=True)
t=pd.read_csv(src/'trial_metrics_all_stages.csv').query("band=='beta'")
idx=['subject','key','trial_index','event_sample','condition']
w=t.pivot(index=idx,columns='stage',values='task_vs_baseline_db').reset_index()
roi=['FCz','Cz','C1','C2'];aux=['HEOR','HEOL','VEOU','VEOL','ECG']
def filt(x,lo,hi):
 return sosfiltfilt(butter(4,[lo,hi],fs=250,btype='bandpass',output='sos'),x,axis=-1)
def corr(eeg,refs):
 x=eeg-eeg.mean(axis=1,keepdims=True);y=refs-refs.mean(axis=1,keepdims=True)
 den=np.linalg.norm(x,axis=1)[:,None]*np.linalg.norm(y,axis=1)
 if np.any(den<=0): return np.nan
 return np.abs((x@y.T)/den).max(axis=1).mean()
rows=[]
with threadpool_limits(limits=1):
 for k,g in w.groupby('key'):
    d=src/'runs'/k
    a=mne.io.read_raw_fif(d/(k+'_before_raw.fif'))
    b=mne.io.read_raw_fif(d/'iclabel'/(k+'_iclabel_raw.fif'))
    v=a.get_data(picks=aux)
    assert np.array_equal(v,b.get_data(picks=aux)) and a.first_samp==b.first_samp
    eye=filt(np.array([v[0]-v[1],v[2]-v[3]]),1,15);heart=filt(v[4:],5,25)
    both=np.concatenate([a.get_data(picks=roi),b.get_data(picks=roi)])
    x,y=filt(both,1,15),filt(both,5,25)
    flags=pd.read_csv(d/'iclabel/auxiliary_channel_qc.csv').set_index('channel').flag_large_1_15_gt1000uV
    for e in g.itertuples():
        j=e.event_sample-a.first_samp; wins=[slice(j-875,j-375),slice(j+125,j+1125)]
        amp=max(np.sqrt(np.mean(eye[:,v]**2,axis=1)).max() for v in wins)*1e6
        vals=[np.mean([corr(z[off:off+4,v],ref[:,v]) for v in wins]) for z,ref in [(x,eye),(y,heart)] for off in [0,4]]
        rows.append([e.subject,k,e.trial_index,e.condition,amp,e.iclabel_after-e.before,*vals,flags[aux[:4]].any(),bool(flags.ECG)])
    if len(rows)//20%70==0:print(k,flush=True)
data=pd.DataFrame(rows,columns=['subject','key','trial','condition','eog_rms','beta_shift','eye_before','eye_after','heart_before','heart_after','eog_flag','ecg_flag'])
assert len(data)==5849 and data.key.nunique()==293
data['stratum']=np.where(data.eog_rms<=data.groupby(['key','condition']).eog_rms.transform('median'),'lower','higher')
data.to_csv(out/'trial_results.csv',index=False)
