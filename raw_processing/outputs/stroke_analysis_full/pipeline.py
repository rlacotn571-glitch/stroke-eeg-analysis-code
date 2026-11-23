"""Exploratory paired EEG artifact pilot; raw source files are never modified."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import argparse, json, hashlib, warnings, sys, platform
import numpy as np
import pandas as pd
import scipy.signal as ss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
DATA=OUT.parent/'EEG_datasets'/'nm000270_v1.0.6'
AUX={'ECG':'ecg','HEOR':'eog','HEOL':'eog','VEOU':'eog','VEOL':'eog'}
CFG=dict(subjects=[f'sub-{i:02d}' for i in range(1,28)],sessions=['ses-pre','ses-post','ses-follow'],sfreq=250,
         passband=[1,100],notch=[50,100],ica_fit_band=[1,40],pca_variance=0.999,
         eog_threshold=0.4,ecg_threshold=0.3,seed=20261117,
         epoch=[-4,5],baseline=[-3.5,-1.5],task=[0.5,4.5],
         roi=['FCz','Cz','CPz','C1','C2'],bands={'mu':[8,13],'beta':[13,30]})
mne.set_log_level('ERROR')

def inventory():
    rows=[]
    for p in sorted((DATA/'sourcedata').glob('sub-*/ses-*/eeg/*_eeg.vhdr')):
        text=p.read_text(encoding='utf-8-sig'); fields={}
        for line in text.splitlines():
            if '=' in line:
                k,v=line.split('=',1); fields[k]=v
        n=int(fields['NumberOfChannels']); names=[fields[f'Ch{i}'].split(',')[0] for i in range(1,n+1)]
        ev=pd.read_csv(str(p).replace('_eeg.vhdr','_events.tsv'),sep='\t')
        eeg=p.with_suffix('.eeg'); hz=1e6/float(fields['SamplingInterval'])
        rows.append(dict(subject=p.parts[-4],session=p.parts[-3],run=p.stem.split('_run-')[1].split('_')[0],
                         header=str(p.relative_to(DATA)),channels=n,eeg_channels=sum(x not in AUX for x in names),
                         auxiliary_correct=all(x in names for x in AUX),sfreq=hz,
                         duration_s=eeg.stat().st_size/(n*4*hz),MI=int((ev.value==3).sum()),
                         rest=int((ev.value==9).sum()),all_finite_onsets=bool(np.isfinite(ev.onset).all())))
    df=pd.DataFrame(rows);df.to_csv(OUT/'source_inventory.csv',index=False)
    return df

def bandfilter(x,lo,hi,sf):
    return ss.sosfiltfilt(ss.butter(4,[lo,hi],btype='bandpass',fs=sf,output='sos'),x,axis=-1)

def correlations(x,y):
    x=x-x.mean(axis=-1,keepdims=True);y=y-y.mean(axis=-1,keepdims=True)
    den=np.linalg.norm(x,axis=-1)[:,None]*np.linalg.norm(y,axis=-1)[None,:]
    return (x@y.T)/np.maximum(den,np.finfo(float).tiny)

def references(raw):
    a=raw.get_data(picks=list(AUX))
    return np.stack([a[3]-a[4],a[1]-a[2],a[0]])

def metrics(raw,stage,key):
    x=raw.get_data(picks='eeg');sf=raw.info['sfreq'];ref=references(raw)
    f,p=ss.welch(x,fs=sf,nperseg=int(2*sf),axis=-1)
    def power(lo,hi):
        mask=(f>=lo)&(f<=hi)
        return float(np.median(np.trapezoid(p[:,mask],f[mask],axis=-1))*1e12)
    eye=np.abs(correlations(bandfilter(x,1,15,sf),bandfilter(ref[:2],1,15,sf)))
    ecg=np.abs(correlations(bandfilter(x,5,25,sf),bandfilter(ref[2:],5,25,sf)))
    return dict(key=key,stage=stage,rms_median_uV=float(np.median(np.sqrt(np.mean(x*x,axis=1)))*1e6),
                eog_corr_median=float(np.median(eye.max(axis=1))),eog_corr_max=float(eye.max()),
                ecg_corr_median=float(np.median(ecg[:,0])),power_30_45_uV2=power(30,45),
                power_70_90_uV2=power(70,90),line_49_51_uV2=power(49,51)),f,p

def epoch_metrics(raw,events,subject,key,stage):
    epochs=mne.Epochs(raw,events,event_id={'MI':3,'rest':9},tmin=-4,tmax=5,baseline=None,
                      picks='eeg',preload=True,reject=None,flat=None,reject_by_annotation=False)
    x=epochs.get_data(copy=True);t=epochs.times
    roi=[epochs.ch_names.index(c) for c in CFG['roi'] if c in epochs.ch_names]
    b=(t>=-3.5)&(t<-1.5);a=(t>=.5)&(t<4.5)
    rows=[]
    for name,(lo,hi) in CFG['bands'].items():
        # Filter continuous data, then epoch to avoid trial-edge filter artifacts.
        filtered=raw.copy().filter(lo,hi,method='iir',iir_params=dict(order=4,ftype='butter'))
        ep=mne.Epochs(filtered,events,event_id={'MI':3,'rest':9},tmin=-4,tmax=5,baseline=None,
                      picks='eeg',preload=True,reject=None,reject_by_annotation=False)
        z=ep.get_data(copy=True)[:,roi,:]
        bp=np.mean(z[:,:,b]**2,axis=(1,2));ap=np.mean(z[:,:,a]**2,axis=(1,2))
        erd=10*np.log10(np.maximum(ap,1e-30)/np.maximum(bp,1e-30))
        for j,(event,value) in enumerate(zip(ep.events,erd)):
            rows.append(dict(subject=subject,key=key,stage=stage,trial_index=int(ep.selection[j]),
                             event_sample=int(event[0]),condition='MI' if event[2]==3 else 'rest',band=name,
                             task_vs_baseline_db=float(value),baseline_power_uV2=float(bp[j]*1e12),
                             task_power_uV2=float(ap[j]*1e12),peak_to_peak_max_uV=float(np.ptp(x[j],axis=-1).max()*1e6),
                             flag_ptp_200uV=bool(np.ptp(x[j],axis=-1).max()>200e-6)))
    return rows,epochs

def run_one(row):
    key=f'{row.subject}_{row.session}_run-{row.run}';folder=OUT/'runs'/key
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists():
        print('SKIP',key,flush=True);return
    print('START',key,flush=True)
    header=DATA/row.header
    raw=mne.io.read_raw_brainvision(header,preload=True,eog=['HEOR','HEOL','VEOU','VEOL'])
    raw.set_channel_types(AUX)
    raw.set_montage('standard_1005',on_missing='raise')
    assert len(mne.pick_types(raw.info,eeg=True))==59
    # Flag questionable channels; do not silently interpolate/exclude them in this pilot.
    native=raw.get_data(picks='eeg');sd=native.std(axis=1)
    logsd=np.log10(np.maximum(sd,1e-30));mad=np.median(np.abs(logsd-np.median(logsd)))
    flags=(sd<1e-8)|(np.abs(logsd-np.median(logsd))>6*max(1.4826*mad,1e-6))
    qc=pd.DataFrame(dict(channel=raw.copy().pick('eeg').ch_names,std_uV=sd*1e6,flag_review=flags))
    qc.to_csv(folder/'channel_qc.csv',index=False);del native
    raw.set_eeg_reference('average',projection=False)
    native_metric,fr,pr=metrics(raw,'raw_native_average_reference',key)
    proc=raw.copy()
    for freq in [50,100]:
        proc.notch_filter([freq],method='iir',iir_params=dict(order=2,ftype='butter'))
    proc.filter(1,100,method='iir',iir_params=dict(order=4,ftype='butter'))
    proc.resample(250)
    del raw
    ev=pd.read_csv(str(header).replace('_eeg.vhdr','_events.tsv'),sep='\t')
    ev=ev[ev.value.isin([3,9])].copy()
    events=np.column_stack([np.rint(ev.onset.to_numpy()*250).astype(int),np.zeros(len(ev),int),ev.value.astype(int)])
    fit=proc.copy().filter(1,40,method='iir',iir_params=dict(order=4,ftype='butter'))
    ica=ICA(n_components=.999,method='picard',random_state=CFG['seed'],max_iter=500,
            fit_params=dict(ortho=False,extended=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ica.fit(fit,picks='eeg',decim=2,reject=None)
    sources=ica.get_sources(fit).get_data();ref=references(fit)
    eog_scores=np.max(np.abs(correlations(bandfilter(sources,1,15,250),bandfilter(ref[:2],1,15,250))),axis=1)
    ecg_scores=np.abs(correlations(bandfilter(sources,5,25,250),bandfilter(ref[2:],5,25,250)))[:,0]
    del sources,fit
    excluded=np.where((eog_scores>=.4)|(ecg_scores>=.3))[0].tolist()
    ica.exclude=excluded
    components=pd.DataFrame(dict(component=np.arange(len(eog_scores)),eog_abs_r=eog_scores,
                                 ecg_abs_r=ecg_scores,excluded=np.isin(np.arange(len(eog_scores)),excluded)))
    components.to_csv(folder/'ica_components.csv',index=False)
    ica.save(folder/f'{key}-ica.fif',overwrite=True)
    clean=ica.apply(proc.copy())
    before,fb,pb=metrics(proc,'filtered_before_ica',key)
    after,fc,pc=metrics(clean,'filtered_after_ica',key)
    erd0,ep0=epoch_metrics(proc,events,row.subject,key,'before')
    erd1,ep1=epoch_metrics(clean,events,row.subject,key,'after')
    assert np.array_equal(ep0.events,ep1.events) and np.array_equal(ep0.selection,ep1.selection)
    pd.DataFrame(erd0+erd1).to_csv(folder/'trial_metrics.csv',index=False)
    ev.to_csv(folder/'selected_events.tsv',sep='\t',index=False)
    for obj,label in [(proc,'before'),(clean,'after')]:
        obj.save(folder/f'{key}_{label}_raw.fif',overwrite=True,fmt='single')
    rows=[native_metric,before,after]
    for v in rows:v.update(subject=row.subject,session=row.session,run=row.run,n_components=ica.n_components_,
                           n_removed=len(excluded),n_epochs=len(ep0),n_event_candidates=len(events),
                           n_channel_flags=int(flags.sum()),ica_iterations=int(ica.n_iter_))
    pd.DataFrame(rows).to_csv(folder/'run_metrics.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for f,p,label in [(fr,pr,'Raw (avg ref)'),(fb,pb,'Filtered'),(fc,pc,'Filtered + ICA')]:
        sel=(f>=1)&(f<=100);ax[0].plot(f[sel],10*np.log10(np.maximum(np.median(p[:,sel],axis=0)*1e12,1e-30)),label=label)
    ax[0].set(xlabel='Hz',ylabel='PSD (dB uV^2/Hz)',title='Median EEG spectrum');ax[0].legend(fontsize=8)
    ax[1].bar(np.arange(len(eog_scores))-.2,eog_scores,.4,label='EOG');ax[1].bar(np.arange(len(eog_scores))+.2,ecg_scores,.4,label='ECG')
    ax[1].axhline(.4,color='C0',ls=':');ax[1].axhline(.3,color='C1',ls=':');ax[1].set(xlabel='ICA component',ylabel='Absolute r',title=f'Excluded: {excluded}');ax[1].legend()
    # Same 10 seconds around the strongest observed vertical EOG excursion.
    peak=np.argmax(np.abs(ref[0]));start=max(0,min(peak-1250,proc.n_times-2500));stop=start+2500
    for obj,label in [(proc,'Before'),(clean,'After')]:ax[2].plot(np.arange(stop-start)/250,obj.get_data(picks=['Fp1'],start=start,stop=stop)[0]*1e6,label=label,alpha=.8)
    ax[2].set(xlabel='Seconds',ylabel='Fp1 (uV)',title='Same segment, strongest VEOG');ax[2].legend()
    fig.suptitle(key);fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    if excluded:
        fig=ica.plot_components(picks=excluded,show=False)
        figs=fig if isinstance(fig,list) else [fig]
        for i,f in enumerate(figs):f.savefig(folder/f'removed_topographies_{i}.png',dpi=130);plt.close(f)
    proof=dict(config=CFG,source=str(header),source_sha256=hashlib.sha256(header.with_suffix('.eeg').read_bytes()).hexdigest(),
               excluded=excluded,ica_iterations=int(ica.n_iter_),warnings=[str(w.message) for w in caught],
               channel_flags=qc.loc[qc.flag_review,'channel'].tolist(),epoch_drop_log=[list(x) for x in ep0.drop_log])
    (folder/'completed.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print('DONE',key,'excluded',excluded,'epochs',len(ep0),'iterations',ica.n_iter_,flush=True)

def summarize():
    folders=[p.parent for p in (OUT/'runs').glob('*/completed.json')]
    if not folders:return
    runs=pd.concat([pd.read_csv(p/'run_metrics.csv') for p in folders],ignore_index=True)
    trials=pd.concat([pd.read_csv(p/'trial_metrics.csv') for p in folders],ignore_index=True)
    runs.to_csv(OUT/'run_metrics.csv',index=False);trials.to_csv(OUT/'trial_metrics.csv',index=False)
    # Subject means, not trials-as-independent-subjects.
    subj=trials.groupby(['subject','stage','condition','band']).task_vs_baseline_db.mean().reset_index()
    subj.to_csv(OUT/'subject_erd.csv',index=False)
    paired=runs[runs.stage!='raw_native_average_reference'].pivot(index=['subject','run'],columns='stage',values=['eog_corr_median','ecg_corr_median','rms_median_uV','power_70_90_uV2'])
    changes=[]
    for (s,r),row in paired.iterrows():
        d=dict(subject=s,run=r)
        for metric in ['eog_corr_median','ecg_corr_median','rms_median_uV','power_70_90_uV2']:
            b=row[(metric,'filtered_before_ica')];a=row[(metric,'filtered_after_ica')]
            d[metric+'_before']=b;d[metric+'_after']=a;d[metric+'_change_pct']=100*(a/b-1) if b else np.nan
        changes.append(d)
    pd.DataFrame(changes).to_csv(OUT/'paired_changes.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,title in zip(axes[:2],['eog_corr_median','ecg_corr_median'],['EEG–EOG coupling','EEG–ECG coupling']):
        for s,g in runs[runs.stage!='raw_native_average_reference'].groupby('subject'):
            av=g.groupby('stage')[metric].mean();ax.plot([0,1],[av['filtered_before_ica'],av['filtered_after_ica']],'-o',label=s)
        ax.set(xticks=[0,1],xticklabels=['Before ICA','After ICA'],ylabel='Mean of run median |r|',title=title);ax.legend()
    for (s,band),g in subj[subj.condition=='MI'].groupby(['subject','band']):
        av=g.set_index('stage').task_vs_baseline_db;axes[2].plot([0,1],[av['before'],av['after']],'-o',label=f'{s} {band}')
    axes[2].axhline(0,color='gray',lw=1);axes[2].set(xticks=[0,1],xticklabels=['Before ICA','After ICA'],ylabel='MI task / baseline (dB)',title='Central ROI signal preservation');axes[2].legend(fontsize=7)
    fig.suptitle('Exploratory pilot — automated component selection, not clinical validation')
    fig.savefig(OUT/'pilot_summary.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'analysis_config.json').write_text(json.dumps(CFG,indent=2),encoding='utf-8')
    inv=inventory();selected=inv[(inv.subject.isin(CFG['subjects']))&(inv.session.isin(CFG['sessions']))]
    if args.limit:selected=selected.head(args.limit)
    with threadpool_limits(limits=2):
        for row in selected.itertuples(index=False):run_one(row)
        summarize()
    print('COMPLETE',len(selected),'runs',flush=True)
