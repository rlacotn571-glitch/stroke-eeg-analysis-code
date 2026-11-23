"""Check beta window powers in continuous EEG and restore one original-rule IC at a time."""
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

out=Path(__file__).resolve().parent
root=out.parents[1]/"stroke_analysis_full"
splits=pd.read_csv(out.parent/"run_splits/participant_splits.csv").query("endpoint=='ses-post'")
cases=splits.query("method=='Original' and not same_direction").set_index("subject")
emergent=cases.index.intersection(splits.query("method=='Filter only' and same_direction").subject)
hybrid=pd.read_csv(out/"run_cleaning_effect.csv")
key_checks=set(hybrid[hybrid.subject.isin(cases.index)&hybrid.status_with_run_filtered.eq("agree")].key)
trials=pd.read_csv(root/"trial_metrics_all_stages.csv").query("band=='beta' and session in ['ses-pre','ses-post'] and stage in ['before','iclabel_after']")
trials=trials[trials.subject.isin(cases.index)].copy()
trials["parity"]=np.where(trials.key.str[-2:].astype(int)%2,"odd","even")
counts=trials[trials.stage=="iclabel_after"].groupby(["subject","session","parity","condition"]).size()
names=["MI_task","MI_baseline","rest_task","rest_baseline"]
def powers(raw,events):
    central=raw.copy().pick(["FCz","Cz","C1","C2"])
    x=central.filter(13.,30.,method="iir",iir_params=dict(order=4,ftype="butter")).get_data()
    return np.array([[np.mean(x[:,e+a:e+b]**2)*1.e12 for a,b in [(125,1125),(-875,-375)]] for e in events])

def calculate():
    checks=[]
    restores=[]
    for key,group in trials.groupby("key"):
        folder=root/"runs"/key
        before=mne.io.read_raw_fif(folder/(key+"_before_raw.fif"),preload=True)
        for stage in ["before","iclabel_after"]:
            g=group[group.stage==stage].sort_values("event_sample")
            raw=before if stage=="before" else mne.io.read_raw_fif(folder/"iclabel"/(key+"_iclabel_raw.fif"),preload=True)
            ev=g.event_sample.to_numpy()-raw.first_samp
            measure=powers(raw,ev)
            stored=g[["task_power_uV2","baseline_power_uV2"]].to_numpy()
            err=np.max(abs(10*np.log10(measure/stored)))
            assert err<1e-4,(key,stage,err)
            checks.append([key,stage,len(g),err])
        subject,session,parity=g.iloc[0][["subject","session","parity"]]
        if subject not in emergent and key not in key_checks:
            continue
        labels=pd.read_csv(folder/"iclabel/component_scores.csv").set_index("component")
        ica=mne.preprocessing.read_ica(folder/"iclabel"/(key+"-ica.fif"))
        excluded=list(ica.exclude)
        assert set(excluded)==set(labels.index[labels.excluded])
        # Validate the saved fit, then compare each restoration with that exact reconstruction.
        original=powers(ica.apply(before.copy()),ev)
        assert np.max(abs(10*np.log10(original/stored)))<1.e-4
        ica.exclude=[]
        all_restored=powers(ica.apply(before.copy()),ev)
        source_values=group[group.stage=="before"].sort_values("event_sample")[["task_power_uV2","baseline_power_uV2"]].to_numpy()
        assert np.max(abs(10*np.log10(all_restored/source_values)))<1.e-4
        for ic in excluded:
            ica.exclude=[j for j in excluded if j!=ic]
            ratio=10*np.log10(powers(ica.apply(before.copy()),ev)/original)
            terms=[]
            for condition,signs in [("MI",[1,-1]),("rest",[-1,1])]:
                terms.extend(ratio[g.condition.to_numpy()==condition].sum(0)*signs/counts[subject,session,parity,condition])
            terms=np.array(terms)*(1 if session=="ses-post" else -1)
            odd,even=cases.loc[subject,["odd_change_dB","even_change_dB"]].to_numpy(float)
            if parity=="odd":
                odd+=terms.sum()
            else:
                even+=terms.sum()
            restores.append([subject,key,parity,int(ic),labels.loc[ic,"predicted_class"],*terms,terms.sum(),odd,even,odd*even>0])
    pd.DataFrame(checks,columns=["key","stage","trials","max_window_error_dB"]).to_csv(out/"signal_checks.csv",index=False)
    df=pd.DataFrame(restores,columns=["subject","key","parity","component_0_based","ICLabel_class",*names,"change_in_half_dB","odd_after_restore","even_after_restore","same_direction"])
    df.to_csv(out/"component_restoration.csv",index=False)
    print("Checked",len(checks)//2,"paired runs. Maximum error (dB)",max(r[-1] for r in checks))
    print("Restorations",len(df),"; same sign",df.same_direction.sum())
    print(df.loc[df.same_direction,["key","component_0_based","ICLabel_class","change_in_half_dB","odd_after_restore","even_after_restore"]].to_string(index=False))

if __name__=="__main__":
    mne.set_log_level("ERROR")
    with threadpool_limits(limits=1):
        calculate()
