"""Separate exploratory branch: ICLabel scores are not calibrated stroke probabilities."""
import pipeline as p
from pathlib import Path
import json,warnings
import numpy as np,pandas as pd
from mne_icalabel.iclabel import iclabel_label_components

CLASSES=['brain','muscle','eye','heart','line','channel','other']
CONFIG=dict(threshold=.90,brain_max=.10,fit_band=[1,100],decimation=1,
            algorithm='picard extended=True ortho=False (extended-infomax-compatible)',
            status='exploratory; topographies and component spectra require manual review')

def main(folders=None,aggregate=True):
    all_metrics=[];all_trials=[];all_aux=[]
    for folder in (sorted((p.OUT/'runs').glob('*')) if folders is None else folders):
        if not (folder/'completed.json').exists():continue
        key=folder.name;dest=folder/'iclabel';dest.mkdir(exist_ok=True)
        if not (dest/'completed.json').exists():
            print('START ICLABEL',key,flush=True)
            before=p.mne.io.read_raw_fif(folder/f'{key}_before_raw.fif',preload=True)
            ica=p.ICA(n_components=.999,method='picard',random_state=p.CFG['seed'],max_iter=500,
                      fit_params=dict(ortho=False,extended=True))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');ica.fit(before,picks='eeg',reject=None)
                scores=iclabel_label_components(before,ica,backend='onnx')
            artifact=np.max(scores[:,1:4],axis=1)
            excluded=np.where((artifact>=CONFIG['threshold'])&(scores[:,0]<=CONFIG['brain_max']))[0].tolist()
            table=pd.DataFrame(scores,columns=CLASSES);table.insert(0,'component',np.arange(len(scores)))
            table['excluded']=np.isin(np.arange(len(scores)),excluded)
            table['predicted_class']=[CLASSES[i] for i in np.argmax(scores,axis=1)]
            table.to_csv(dest/'component_scores.csv',index=False)
            ica.exclude=excluded;after=ica.apply(before.copy())
            ica.save(dest/f'{key}-ica.fif',overwrite=True)
            after.save(dest/f'{key}_iclabel_raw.fif',overwrite=True,fmt='single')
            ev=pd.read_csv(folder/'selected_events.tsv',sep='\t')
            events=np.column_stack([np.rint(ev.onset.to_numpy()*250).astype(int),np.zeros(len(ev),int),ev.value.astype(int)])
            trial,ep=p.epoch_metrics(after,events,key.split('_')[0],key,'iclabel_after')
            pd.DataFrame(trial).to_csv(dest/'trial_metrics.csv',index=False)
            metric,f,psd=p.metrics(after,'iclabel_after',key)
            metric.update(subject=key.split('_')[0],n_components=int(ica.n_components_),n_removed=len(excluded),
                          n_epochs=len(ep),ica_iterations=int(ica.n_iter_))
            pd.DataFrame([metric]).to_csv(dest/'run_metrics.csv',index=False)
            # Keep all components, including those not removed, for transparent manual review.
            source=ica.get_sources(before).get_data();freq,power=p.ss.welch(source,fs=250,nperseg=500,axis=-1)
            np.savez_compressed(dest/'component_spectra.npz',frequencies=freq,psd=power)
            if excluded:
                figs=ica.plot_components(picks=excluded,show=False)
                if not isinstance(figs,list):figs=[figs]
                for i,fig in enumerate(figs):fig.savefig(dest/f'removed_topographies_{i}.png',dpi=140);p.plt.close(fig)
            fig,axes=p.plt.subplots(1,2,figsize=(11,4),layout='constrained')
            _,fb,pb=p.metrics(before,'before',key)
            for ff,pp,label in [(fb,pb,'Before'),(f,psd,'ICLabel after')]:
                sel=(ff>=1)&(ff<=100);axes[0].plot(ff[sel],10*np.log10(np.maximum(np.median(pp[:,sel],axis=0)*1e12,1e-30)),label=label)
            axes[0].set(xlabel='Hz',ylabel='Median EEG PSD (dB uV^2/Hz)',title='Identical filtering and reference');axes[0].legend()
            for idx in excluded:axes[1].plot(freq,10*np.log10(np.maximum(power[idx],1e-30)),label=f'IC {idx}: {table.loc[idx,"predicted_class"]}')
            axes[1].set(xlim=(1,100),xlabel='Hz',ylabel='Component PSD (arbitrary dB)',title=f'Automatically removed: {excluded}')
            if excluded:axes[1].legend(fontsize=7)
            fig.suptitle(key+' — exploratory ICLabel');fig.savefig(dest/'comparison.png',dpi=150);p.plt.close(fig)
            proof=dict(config=CONFIG,excluded=excluded,iterations=int(ica.n_iter_),warnings=[str(w.message) for w in caught])
            (dest/'completed.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
            print('DONE ICLABEL',key,'removed',excluded,flush=True)
        all_metrics.append(pd.read_csv(dest/'run_metrics.csv'));all_trials.append(pd.read_csv(dest/'trial_metrics.csv'))
        raw=p.mne.io.read_raw_fif(folder/f'{key}_before_raw.fif',preload=True)
        aux=raw.get_data(picks=list(p.AUX));filtered=p.bandfilter(aux,1,15,250)
        for name,a,b in zip(p.AUX,aux,filtered):
            all_aux.append(dict(key=key,channel=name,std_all_uV=float(a.std()*1e6),std_1_15_uV=float(b.std()*1e6),
                                flag_large_1_15_gt1000uV=bool(b.std()>1e-3)))
    if aggregate:
        pd.concat(all_metrics).to_csv(p.OUT/'iclabel_run_metrics.csv',index=False)
        pd.concat(all_trials).to_csv(p.OUT/'iclabel_trial_metrics.csv',index=False)
        pd.DataFrame(all_aux).to_csv(p.OUT/'auxiliary_channel_qc.csv',index=False)
        (p.OUT/'iclabel_config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    else:
        pd.DataFrame(all_aux).to_csv(folders[0]/'iclabel'/'auxiliary_channel_qc.csv',index=False)
    print('COMPLETE ICLABEL',len(all_metrics),flush=True)

if __name__=='__main__':
    with p.threadpool_limits(limits=2):main()
