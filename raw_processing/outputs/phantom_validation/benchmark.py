from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import numpy as np,pandas as pd,mne
from scipy.io import loadmat
from scipy.signal import resample_poly,butter,sosfiltfilt
from mne_icalabel.iclabel import iclabel_label_components
from threadpoolctl import threadpool_limits
out=Path(__file__).resolve().parent
root=out.parent/'EEG_datasets/on004784_v1.0.0'
gt=loadmat(next(root.rglob('GT*.mat')))['GTdata']
def filt(x,lo,hi):return sosfiltfilt(butter(4,[lo,hi],fs=250,btype='bandpass',output='sos'),x,axis=-1)
def corr(a,b):
    a=a-a.mean(-1,keepdims=True);b=b-b.mean(-1,keepdims=True)
    return (a@b.T)/np.maximum(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1),1e-30)
g=resample_poly(gt[:,:20].T,125,256,axis=1)
train=slice(500,36500);test=slice(38500,74500)
rows=[];scores=[];blocks=[];pred={}
with threadpool_limits(limits=2):
 for cond in ['Brain','Eyes','Facial','Neck','Walking','All']:
    path=next(root.rglob(f'*task-{cond}_eeg.set'));h=loadmat(path,simplify_cells=True)
    scalp=[i for i,a in enumerate(h['chanlocs']) if a['type']=='EEG'];assert len(scalp)==128
    ev=[round(float(a['latency']))-1 for a in h['event'] if str(a['type'])=='65471'];assert len(ev)==2
    src=np.memmap(path.with_suffix('.fdt'),mode='r',dtype='<f4',shape=(h['pnts'],h['nbchan']))
    ab=np.arange(ev[0],ev[1]+1);grid=np.linspace(ab[0],ab[-1],len(gt))
    x=np.stack([resample_poly(np.interp(grid,ab,src[ab,i]),125,256) for i in scalp])*1e-6
    names=[h['chanlocs'][i]['labels'] for i in scalp]
    pos={a['labels']:np.array([-a['Y'],a['X'],a['Z']])*.001 for a in [h['chanlocs'][i] for i in scalp]}
    raw=mne.io.RawArray(x,mne.create_info(names,250,'eeg'),verbose=False)
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos,coord_frame='head'))
    raw.set_eeg_reference('average',verbose=False);raw.notch_filter([60],method='iir',verbose=False)
    raw.filter(1,100,method='iir',iir_params=dict(order=4,ftype='butter'),verbose=False)
    fit=mne.preprocessing.ICA(n_components=.999,method='picard',random_state=20261117,max_iter=500,fit_params=dict(ortho=False,extended=True))
    fit.fit(raw,verbose=False)
    prob=iclabel_label_components(raw,fit,backend='onnx')
    remove=np.where((prob[:,1:4].max(axis=1)>=.90)&(prob[:,0]<=.10))[0].tolist()
    er=np.abs(corr(filt(fit.get_sources(raw).get_data(),1,15),filt(g[10:12],1,15))).max(axis=1)
    oracle=np.where(er>=.4)[0].tolist() if cond in ['Eyes','All'] else []
    for i,row in enumerate(prob):scores.append([cond,i,*row,i in remove,er[i],i in oracle])
    for method,ex in [('filter',[]),('iclabel',remove),('injected_eye_reference',oracle)]:
        y=fit.apply(raw.copy(),exclude=ex,verbose=False).get_data() if ex else raw.get_data()
        for band,lo,hi in [('beta',13,30),('mu',8,13)]:
            z=filt(y,lo,hi);brain=filt(g[:10],lo,hi)
            if cond=='Brain' and method=='filter':
                mix=np.linalg.lstsq(brain[:,train].T,z[:,train].T,rcond=None)[0].T
                pred[band]=mix@brain[:,test]
            target=pred[band];signal=z[:,test]
            def metrics(a,b):
                return [np.linalg.norm(a-b)/np.linalg.norm(b),np.sum(a*b)/np.sum(b*b),np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))]
            rows.append([cond,method,band,len(ex),fit.n_components_,fit.n_iter_,*metrics(signal,target)])
            for k in range(6):
                ix=slice(6000*k,6000*(k+1));blocks.append([cond,method,band,k,*metrics(signal[:,ix],target[:,ix])])
    print(cond,len(remove),fit.n_components_,flush=True)
pd.DataFrame(rows,columns=['condition','method','band','removed','components','iterations','nrmse','gain','cosine']).to_csv(out/'results.csv',index=False)
pd.DataFrame(scores,columns=['condition','IC','brain','muscle','eye','heart','line','channel','other','excluded','eye_input_corr','eye_excluded']).to_csv(out/'components.csv',index=False)
pd.DataFrame(blocks,columns=['condition','method','band','block','nrmse','gain','cosine']).to_csv(out/'blocks.csv',index=False)
