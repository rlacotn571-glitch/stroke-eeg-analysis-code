"""Trace the original-rule post-minus-pre odd/even gap on the original trial weights."""
from pathlib import Path
import numpy as np
import pandas as pd

out=Path(__file__).resolve().parent
source=out.parents[1]/"stroke_analysis_full"
windows=["MI_task","MI_baseline","rest_task","rest_baseline"]
signs=np.array([1.,-1.,-1.,1.])
tol=1.e-8
split=pd.read_csv(out.parent/"run_splits/participant_splits.csv").query("endpoint=='ses-post'")
methods=split.pivot(index="subject",columns="method",values="same_direction")
trials=pd.read_csv(source/"trial_metrics_all_stages.csv").query("band=='beta' and session in ['ses-pre','ses-post'] and stage in ['before','iclabel_after']")
trials=trials[trials.subject.isin(methods.index)].copy()
index=["subject","key","session","condition","trial_index","event_sample"]
original=trials[trials.stage=="iclabel_after"].set_index(index).sort_index()
before=trials[trials.stage=="before"].set_index(index).sort_index()
assert original.index.equals(before.index) and original.index.is_unique
logs=["task_power_uV2","baseline_power_uV2"]
data=original.reset_index()[index].copy()
data["half"]=np.where(data.key.str[-2:].astype(int)%2,"odd","even")
# Logs are calculated per trial before pooling, exactly as in the manuscript.
for prefix,g in [("clean",original),("filter",before)]:
    log_power=10*np.log10(g[logs].to_numpy())
    np.testing.assert_allclose(log_power[:,0]-log_power[:,1],g.task_vs_baseline_db,atol=1.e-8,rtol=0)
    for j,k in enumerate(["task","baseline"]):
        data[prefix+"_"+k]=log_power[:,j]

def parts(g,stage):
    by_condition=g.groupby("condition")[[stage+"_task",stage+"_baseline"]].mean()
    return by_condition.loc[["MI","rest"]].to_numpy().ravel()*signs

def signed(x):
    return 1 if x>tol else -1 if x < -tol else 0

def label_pair(x,y):
    a,b=signed(x),signed(y)
    return "zero" if a*b==0 else "agree" if a==b else "opposite"

participant_rows=[]
visit_rows=[]
run_rows=[]
trial_rows=[]
clean_rows=[]
for subject,g in data.groupby("subject"):
    ref=split[split.subject==subject].set_index("method")
    cells=dict(tuple(g.groupby(["session","half"])))
    clean={k:parts(h,"clean") for k,h in cells.items()}
    filt={k:parts(h,"filter") for k,h in cells.items()}
    halves={}
    for method,d in [("Original",clean),("Filter only",filt)]:
        halves[method]=np.array([(d["ses-post",h]-d["ses-pre",h]).sum() for h in ["odd","even"]])
        np.testing.assert_allclose(halves[method],ref.loc[method,["odd_change_dB","even_change_dB"]].to_numpy(float),atol=1.e-8,rtol=0)
    odd,even=halves["Original"]
    gap=odd-even
    status=label_pair(odd,even)
    assert status!= "zero"
    by_visit={}
    for session,vsign in [("ses-pre",-1),("ses-post",1)]:
        by_visit[session]=vsign*(clean[session,"odd"]-clean[session,"even"])
        visit_rows.append([subject,status,session,*by_visit[session],by_visit[session].sum()])
    np.testing.assert_allclose(sum(v.sum() for v in by_visit.values()),gap,rtol=0,atol=1.e-8)
    winning_visit=max(by_visit,key=lambda v:abs(by_visit[v].sum()))
    assert abs(abs(by_visit["ses-pre"].sum())-abs(by_visit["ses-post"].sum()))>tol
    leader=int(np.argmax(by_visit[winning_visit]*np.sign(gap)))
    cells_by_size=sorted([(abs(value),visit,windows[i],value) for visit,values in by_visit.items()
                          for i,value in enumerate(values)],reverse=True)
    assert cells_by_size[0][0]-cells_by_size[1][0]>tol
    _,peak_visit,peak_window,peak_value=cells_by_size[0]
    run_cases=[]
    trials_changed=0
    for (session,half),h in cells.items():
        vsign=1 if session=="ses-post" else -1
        hsign=1 if half=="odd" else -1
        ci=0 if half=="odd" else 1
        counts=h.condition.value_counts()
        stage_delta=[]
        for key,run in h.groupby("key"):
            kept=h[h.key!=key]
            difference=vsign*(clean[session,half]-parts(kept,"clean"))
            without=halves["Original"].copy()
            without[ci]-=difference.sum()
            after=label_pair(*without)
            # Signed influence equals the observed gap minus the gap after omitting this run.
            effect=difference*hsign
            np.testing.assert_allclose(effect.sum(),gap-(without[0]-without[1]),atol=1.e-9,rtol=0)
            run_rows.append([subject,status,session,key,half,*effect,effect.sum(),*without,after])
            run_cases.append((key,effect.sum(),after,effect))
            weights=np.repeat(run.condition.value_counts().loc[["MI","rest"]].to_numpy()/counts.loc[["MI","rest"]].to_numpy(),2)
            corr=vsign*(parts(run,"clean")-parts(run,"filter"))*weights
            replace=halves["Original"].copy()
            replace[ci]-=corr.sum()
            clean_rows.append([subject,key,session,half,*corr,corr.sum(),*replace,label_pair(*replace)])
            stage_delta.append(corr)
        np.testing.assert_allclose(np.sum(stage_delta,0),vsign*(clean[session,half]-filt[session,half]),rtol=0,atol=1.e-8)
        for condition,t in h.groupby("condition"):
            # Removing one trial changes its condition mean by (x - mean)/(n - 1).
            score=t.clean_task-t.clean_baseline
            diffs=vsign*(1 if condition=="MI" else -1)*(score.to_numpy()-score.mean())/(len(t)-1)
            for (key,trial),diff in zip(t[["key","trial_index"]].to_numpy(),diffs):
                after_halves=halves["Original"].copy()
                after_halves[ci]-=diff
                after=label_pair(*after_halves)
                if after!=status:
                    trials_changed+=1
                trial_rows.append([subject,status,key,int(trial),condition,hsign*diff,*after_halves,after])
    max_abs=max(abs(r[1]) for r in run_cases)
    leaders=[r for r in run_cases if abs(abs(r[1])-max_abs)<=tol]
    # Direction-restoring runs are listed separately from those with greatest absolute leverage.
    switched=[r[0] for r in run_cases if r[2]!=status]
    participant_rows.append([subject,status,*halves["Original"],*halves["Filter only"],gap,
        by_visit["ses-pre"].sum(),by_visit["ses-post"].sum(),winning_visit,windows[leader],by_visit[winning_visit][leader],
        ";".join(r[0] for r in leaders),max_abs,";".join(switched),len(switched),len(run_cases),
        trials_changed,len(g),bool(methods.loc[subject,"Relaxed"]),bool(methods.loc[subject,"Strict"]),
        peak_visit,peak_window,peak_value])

def export(rows,names,filename):
    df=pd.DataFrame(rows,columns=names)
    df.to_csv(out/filename,index=False)
    return df

people=export(participant_rows,["subject","status","odd","even","filter_odd","filter_even","gap",
    "pre_contribution","post_contribution","larger_visit","leading_supporting_window","window_contribution",
    "highest_leverage_runs","max_abs_run_influence","deletions_changing_status","n_run_status_changes","n_runs",
    "n_trial_status_changes","n_trials","relaxed_agree","strict_agree","largest_abs_cell_visit",
    "largest_abs_cell_window","largest_abs_cell_signed_dB"],"participants.csv")
export(visit_rows,["subject","status","session",*windows,"total_gap_contribution"],"visit_windows.csv")
export(run_rows,["subject","status","session","key","half",*windows,"signed_gap_influence","odd_without","even_without","status_without"],"run_influence.csv")
export(trial_rows,["subject","status","key","trial_index","condition","signed_gap_influence","odd_without","even_without","status_without"],"trial_influence.csv")
export(clean_rows,["subject","key","session","half",*windows,"cleaning_shift_in_half","odd_with_run_filtered","even_with_run_filtered","status_with_run_filtered"],"run_cleaning_effect.csv")
assert len(people)==21 and (people.status=="opposite").sum()==8
print(people[people.status=="opposite"].to_string(index=False))
