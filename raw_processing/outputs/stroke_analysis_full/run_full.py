"""Resumable per-record isolated processing, bounded by available RAM."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='2'
from pathlib import Path
import sys,json,time,subprocess,traceback,gc,argparse
from types import SimpleNamespace
import psutil
ROOT=Path(__file__).resolve().parent

def validate(folder):
    import pipeline as p
    import numpy as np,pandas as pd
    proof=json.loads((folder/'completed.json').read_text())
    manifest=json.loads((p.DATA/'full_remote_manifest.json').read_text(encoding='utf-8-sig'))
    rel=str(Path(proof['source']).with_suffix('.eeg').relative_to(p.DATA)).replace('\\','/')
    expected=next(x['checksum'] for x in manifest if x['path']==rel)
    assert proof['source_sha256']==expected
    key=folder.name;shapes=[];before=None;rows=[]
    for label,file in [('before',folder/f'{key}_before_raw.fif'),('reference_after',folder/f'{key}_after_raw.fif'),('iclabel_after',folder/'iclabel'/f'{key}_iclabel_raw.fif')]:
        r=p.mne.io.read_raw_fif(file,preload=True);x=r.get_data()
        assert x.shape[0]==64 and r.info['sfreq']==250 and np.isfinite(x).all()
        types=r.get_channel_types();assert types.count('eeg')==59 and types.count('eog')==4 and types.count('ecg')==1
        if before is None:before=x.copy()
        elif label=='reference_after' and not proof['excluded']:assert np.allclose(x,before,atol=1e-10,rtol=2e-6)
        assert np.array_equal(x[-5:],before[-5:])
        shapes.append(x.shape);rows.append(dict(stage=label,samples=x.shape[1],finite=True))
        del r,x
    assert len(set(shapes))==1
    a=pd.read_csv(folder/'trial_metrics.csv');b=pd.read_csv(folder/'iclabel'/'trial_metrics.csv')
    ids=['key','trial_index','condition','band','event_sample']
    before_trials=a[a.stage=='before'][ids].reset_index(drop=True)
    assert before_trials.equals(b[ids].reset_index(drop=True))
    (folder/'validated.json').write_text(json.dumps(dict(key=key,files=rows,source_sha256_verified=True,
                              matched_trials=len(b)//2,validated_at=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))

def worker(key):
    try:psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (AttributeError,psutil.Error):pass
    import pipeline as p
    import iclabel_comparison as ic
    rows=json.loads((ROOT/'selected_records.json').read_text())
    row=next(r for r in rows if r['key']==key)
    folder=ROOT/'runs'/key
    with p.threadpool_limits(limits=2):
        p.run_one(SimpleNamespace(**row));gc.collect()
        ic.main([folder],aggregate=False);gc.collect()
        validate(folder)

def manage():
    import pipeline as p
    import iclabel_comparison as ic
    inv=p.inventory();sel=inv[inv.subject.isin(p.CFG['subjects'])&inv.session.isin(p.CFG['sessions'])].copy()
    # Complete pre first; then paired post and follow. No stimulus sessions are mixed in.
    sel['session_order']=sel.session.map({'ses-pre':0,'ses-post':1,'ses-follow':2})
    sel=sel.sort_values(['session_order','subject','run'])
    sel['key']=sel.subject+'_'+sel.session+'_run-'+sel.run
    assert len(sel)==293
    rows=sel.to_dict(orient='records')
    (ROOT/'selected_records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'analysis_config.json').write_text(json.dumps(p.CFG,indent=2))
    (ROOT/'iclabel_config.json').write_text(json.dumps(ic.CONFIG,indent=2))
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    pending=[r['key'] for r in rows if not (ROOT/'runs'/r['key']/'validated.json').exists()]
    live={};failed=[];done=293-len(pending);last=0
    while pending or live:
        available=psutil.virtual_memory().available/1e9
        limits=json.loads((ROOT/'runtime_limits.json').read_text()) if (ROOT/'runtime_limits.json').exists() else {}
        cap=int(limits.get('max_workers',2))
        floor=float(limits.get('second_worker_min_free_gb',1.5))
        if len(live)>=2:
            max_workers=min(cap,3) if available>=float(limits.get('third_worker_min_free_gb',2.0)) else min(cap,2)
        else:
            max_workers=min(cap,2) if live and available>=floor else 1
        while pending and len(live)<max_workers:
            key=pending.pop(0);log=open(logs/(key+'.log'),'w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',key],stdout=log,stderr=subprocess.STDOUT)
            live[key]=(proc,log,time.time())
            print('START',key,'workers',len(live),'freeGB',round(available,1),flush=True)
        for key,(proc,log,start) in list(live.items()):
            if proc.poll() is not None:
                log.close();del live[key]
                if proc.returncode==0:done+=1;print('DONE',done,'/293',key,'seconds',round(time.time()-start),flush=True)
                else:failed.append(key);print('FAILED',key,'exit',proc.returncode,flush=True)
        if time.time()-last>10:
            status=dict(total=293,validated=done,failed=failed,active=list(live),pending=len(pending),available_ram_GB=round(available,2),updated=time.strftime('%Y-%m-%d %H:%M:%S'))
            temp=ROOT/'progress.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(ROOT/'progress.json');last=time.time()
        time.sleep(2)
    status=dict(total=293,validated=done,failed=failed,active=[],pending=0,updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    (ROOT/'progress.json').write_text(json.dumps(status,indent=2))
    print('FINISHED',json.dumps(status),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:manage()
