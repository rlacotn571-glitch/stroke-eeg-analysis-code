from pathlib import Path
import numpy as np
import pandas as pd
from population import inventory, ambiguous, unique_keys

root=Path("outputs")
b=root/"BSPC"
out=Path("tables")
out.mkdir(exist_ok=True)
def write(d,name):
    d.to_csv(out/(name+".csv"),index=False)
def ci(v):
    x=np.asarray(v)
    draws=x[np.random.default_rng(20261117).integers(0,len(x),(10000,len(x)))]
    return [len(x),x.mean(),*np.quantile(draws.mean(1),[.025,.975])]
trials=pd.read_csv(root/"stroke_analysis_full/trial_metrics_all_stages.csv").query("band=='beta' and stage=='before'")
counts=trials.groupby("key").size()
inv=inventory.copy()
inv["trials"]=inv.key.map(counts)
inv["unique_representative"]=inv.key.isin(unique_keys)
inv["identity_resolved"]=~inv.subject.isin(ambiguous)
complete=pd.read_csv(b/"longitudinal_rules/participant_changes.csv").subject.unique()
inv["complete"]=inv.subject.isin(complete)
write(inv,"recording_inclusion")
split=pd.read_csv(b/"run_splits/participant_splits.csv")
flow=[]
def row(label,d,people=None):
    flow.append([label,people if people is not None else d.subject.nunique(),d.key.nunique(),int(d.trials.sum())])
row("Released filenames and codes",inv)
row("Distinct EEG payloads",inv[inv.unique_representative],"")
row("Identity-resolved records, any visit",inv[inv.identity_resolved])
row("Identity-resolved, three visits",inv[inv.complete])
for visit in ["ses-post","ses-follow"]:
    ids=split[split.endpoint==visit].subject.unique()
    row(visit[4:]+" compared with pre, split eligible",inv[inv.subject.isin(ids)&inv.session.isin(["ses-pre",visit])])
write(pd.DataFrame(flow,columns=["analysis_set","people_or_codes","recordings","trials"]),"flow")

rule=pd.read_csv(b/"sensitivity/run_results.csv")
components=["components_removed","MI_task","MI_baseline","rest_task","rest_baseline","shift"]
for _,pair in inv[inv.duplicated("sha256",keep=False)].groupby("sha256"):
    a,c=[rule[rule.key==key].sort_values("rule") for key in pair.key]
    np.testing.assert_allclose(a[components],c[components],atol=1e-10,rtol=0)
rule=rule[rule.key.isin(unique_keys)]
orig=rule[rule.rule=="Original"].copy()
# Reconstruct the original-rule recording shift directly from the two trial stages.
stages=pd.read_csv(root/"stroke_analysis_full/trial_metrics_all_stages.csv").query("band=='beta' and stage in ['before','iclabel_after']")
direct=stages.groupby(["key","stage","condition"]).task_vs_baseline_db.mean().unstack("condition")
contrasts=(direct.MI-direct.rest).unstack("stage")
np.testing.assert_allclose(orig["shift"],(contrasts.iclabel_after-contrasts.before).loc[orig.key],atol=1e-8,rtol=0)
person=stages[stages.subject.isin(complete)].groupby(["subject","session","stage","condition"]).task_vs_baseline_db.mean().unstack("condition")
pooled=(person.MI-person.rest).unstack(["session","stage"])
pc=pd.read_csv(b/"longitudinal_rules/participant_changes.csv").set_index(["endpoint","subject"])
for visit in ["ses-post","ses-follow"]:
    for stage,col in [("before","Filter only"),("iclabel_after","Original")]:
        v=pooled[visit,stage]-pooled["ses-pre",stage]
        np.testing.assert_allclose(v,pc.loc[visit,col].reindex(v.index),atol=1e-8,rtol=0)
metrics={"median positive shift among four decreases":orig[orig.eligible & orig.increased]["shift"].median(),
         **{f"absolute shift quantile {q}":orig["shift"].abs().quantile(q) for q in [.5,.9,1]}}
spans=pd.read_csv(b/"record_review/record_spans.csv")
for flag,g in spans.groupby("switch"):
    for q in [.25,.5,.75]:metrics[f"span {flag} quantile {q}"]=g.span.quantile(q)
rep=pd.read_csv(root/"stroke_mechanism/repeat_scale.csv").query("pair=='iclabel_after-before'")
rep=rep[~rep.subject.isin(ambiguous)]
metrics.update({"repeat visits":len(rep),"repeat median shift":rep.groupby("subject").median_cleaning_shift.mean().median(),
                "repeat median difference":rep.groupby("subject").median_repeat_difference.mean().median(),
                "visits with smaller shift":(rep.median_cleaning_shift<rep.median_repeat_difference).sum()})
selected=pd.read_csv(root/"PLOS_ONE/selection.csv")
for col in ["iclabel","correlation","overlap"]:metrics["selection "+col]=selected.loc[selected.key.isin(unique_keys),col].sum()
vis=pd.read_csv(b/"longitudinal_rules/visit_contrasts.csv")
vis=vis[vis.subject.isin(complete)]
for visit,g in vis.groupby("session"):metrics["mean cleaning shift "+visit]=(g.Original-g["Filter only"]).mean()
write(pd.DataFrame(metrics.items(),columns=["measure","value"]),"recording_and_visit_metrics")
comp=pd.read_csv(root/"PLOS_ONE/subject_estimates.csv")
extra=[]
for band in ["beta","mu"]:
    for col in ["iclabel_after","samefit_corr","selection_difference","fit_difference","before"]:
        d=comp.query("band==@band and subject in @complete").pivot(index="subject",columns="session",values=col)
        for visit in ["ses-post","ses-follow"]:
            extra.append([band,col,visit,*ci(d[visit]-d["ses-pre"])])
write(pd.DataFrame(extra,columns=["band","quantity","visit","people","mean","low","high"]),"comparators")

rules=["Relaxed","Original","Strict"]
matched=[]
for (person,visit),group in split.groupby(["subject","endpoint"]):
    d=group.set_index("method").loc[rules]
    odd,even=(d[c] for c in ["odd_change_dB","even_change_dB"])
    a,bspan=odd.max()-odd.min(),even.max()-even.min()
    gap=abs(odd.Original-even.Original)
    matched.append([person,visit,*[np.sign(x).nunique()==1 for x in [odd,even]],np.sign(odd.Original)==np.sign(even.Original),
                    gap,a,bspan,(a+bspan)/2,gap>max(a,bspan)])
m=pd.DataFrame(matched,columns=["subject","visit","odd_stable","even_stable","between_agree","absolute_gap","odd_span","even_span","mean_span","gap_exceeds_both"])
write(m,"matched_halves")
comparison_rows=[]
for visit in ["ses-post","ses-follow"]:
    participants=m[m.visit==visit]
    assert participants.subject.is_unique
    comparison_rows.append([
        visit,len(participants),int(participants.odd_stable.sum()),int(participants.even_stable.sum()),
        int((participants.odd_stable & participants.even_stable).sum()),participants.mean_span.median(),
        int(participants.between_agree.sum()),participants.absolute_gap.median(),
        int(participants.gap_exceeds_both.sum())])
write(pd.DataFrame(comparison_rows,columns=[
    "visit","participants","odd_stable","even_stable","both_stable","median_within_set_range_dB",
    "odd_even_agree","median_odd_even_gap_dB","gap_exceeds_both_ranges"]),"within_visit_comparisons")
check=pd.read_csv("identity/matched_recording_comparison.csv").query("scenario=='omit_23_and_24'").set_index("visit")
for v,g in m.groupby("visit"):
    np.testing.assert_allclose([g.absolute_gap.median(),g.mean_span.median(),g.between_agree.sum(),g.odd_stable.sum(),g.even_stable.sum()],
       check.loc[v,["between_halves_abs_median_dB","within_half_mean_span_median_dB","between_halves_sign_agree","odd_threshold_stable","even_threshold_stable"]].astype(float))
assert list(pd.DataFrame(flow).iloc[:4,2])==[293,289,269,253]
print(pd.DataFrame(flow).to_string(index=False))
