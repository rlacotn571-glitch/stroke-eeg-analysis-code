"""Verify included BrainVision recordings and reassess the paper without ambiguous codes."""
from pathlib import Path
from itertools import combinations
from hashlib import sha256
from zipfile import ZipFile
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
root = out.parents[1]
dataset = root / "EEG_datasets/nm000270_v1.0.6"
stroke = root / "stroke_analysis_full"
bspc = out.parent
manifest = pd.read_csv(dataset / "full_download_verification.csv").set_index("path")
catalog = pd.read_csv(stroke / "source_inventory.csv")
catalog = catalog[catalog.session.isin(["ses-pre","ses-post","ses-follow"])]
runs = []
def digest(stream):
    h = sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()
for row in catalog.itertuples():
    path = dataset / Path(row.header.replace("\\","/")).with_suffix(".eeg")
    with path.open("rb") as stream:
        fingerprint = digest(stream)
    record = manifest.loc[path.relative_to(dataset).as_posix()] if path.relative_to(dataset).as_posix() in manifest.index else manifest.loc[str(path.relative_to(dataset))]
    assert record.algorithm == "sha256" and record.checksum == fingerprint
    key = f"{row.subject}_{row.session}_run-{int(row.run):02}"
    runs.append([key,row.subject,row.session,int(row.run),fingerprint,path.stat().st_size])
runs = pd.DataFrame(runs, columns=["key","subject","session","run","sha256","bytes"])
assert len(runs) == 293
runs.to_csv(out/"recording_fingerprints.csv",index=False)
pairs = []
for fp,g in runs.groupby("sha256"):
    for a,b in combinations(g.itertuples(),2):
        payloads = []
        markers = []
        for row in [a,b]:
            f = dataset / "sourcedata" / row.subject / row.session / "eeg" / (row.key.replace("_run-", "_task-MotorImagery_run-")+"_eeg.eeg")
            markers.append(f.with_suffix(".vmrk").read_bytes().split(b"[Marker Infos]")[1])
            with ZipFile(dataset/"sourcedata/zenodo/18987384"/(row.subject+".zip")) as z:
                with z.open(f.relative_to(dataset/"sourcedata").as_posix()) as stream:
                    assert digest(stream)==fp
            bdf = next((dataset/row.subject).rglob(f"*run-{row.run-1}_eeg.bdf"))
            with bdf.open("rb") as stream:
                header = stream.read(256)
                stream.seek(int(header[184:192]))
                payloads.append(digest(stream))
        pairs.append([a.key,b.key,fp,a.bytes,markers[0]==markers[1],payloads[0]==payloads[1],True])
pairs = pd.DataFrame(pairs,columns=["code_1_record","code_2_record","signal_sha256","bytes","identical_markers","identical_bdf_payload","both_zenodo_members_verified"])
assert len(pairs)==4 and pairs.iloc[:,-3:].all().all()
pairs.to_csv(out/"verified_duplicates.csv",index=False)

scores=pd.read_csv(bspc/"sensitivity/run_results.csv")
trials=pd.read_csv(stroke/"trial_metrics_all_stages.csv").query("band=='beta' and stage=='before'")
changes=pd.read_csv(bspc/"longitudinal_rules/participant_changes.csv")
splits=pd.read_csv(bspc/"run_splits/participant_splits.csv")
audit=pd.read_csv(bspc/"split_attribution/participants.csv")
rules=["Relaxed","Original","Strict"]
rows=[]
grain=[]
for scenario,excluded in [("as_released",[]),("omit_23",["sub-23"]),("omit_24",["sub-24"]),("omit_23_and_24",["sub-23","sub-24"])]:
    def keep(x):
        return x[~x.subject.isin(excluded)]
    t=keep(trials)
    ss=keep(scores)
    so=ss[ss.rule=="Original"]
    eligible=so[so.eligible]
    allrules=ss.groupby("key").eligible.all()
    spans=ss.pivot(index="key",columns="rule",values="shift").loc[allrules[allrules].index]
    longitudinal=keep(changes)
    sp=keep(splits)
    row=dict(scenario=scenario,codes=t.subject.nunique(),recordings=t.key.nunique(),unique_signal_payloads=keep(runs).sha256.nunique(),
             trials=len(t),four_decreases=len(eligible),increases=(eligible["shift"]>1e-8).sum(),
             common_four=len(spans),reversal=((spans.min(axis=1)<-1e-8)&(spans.max(axis=1)>1e-8)).sum())
    for visit,prefix in [("ses-post","post"),("ses-follow","follow")]:
        d=longitudinal[longitudinal.endpoint==visit]
        h=sp[sp.endpoint==visit]
        original=h[h.method=="Original"]
        row[prefix+"_codes"]=len(d)
        row[prefix+"_threshold_stable"]=(~d.direction_switched).sum()
        row[prefix+"_pos_all"]=(d[rules].min(axis=1)>1e-8).sum()
        row[prefix+"_neg_all"]=(d[rules].max(axis=1)<-1e-8).sum()
        row[prefix+"_halves_n"]=len(original)
        row[prefix+"_halves_agree"]=original.same_direction.sum()
        row[prefix+"_balanced_agreement"]=original.agreement_fraction.mean()
        row[prefix+"_threshold_span_median"]=d.span.median()
        p=h[h.method.isin(rules)].pivot(index="subject",columns="method",values=["odd_change_dB","even_change_dB"])
        sides=[p[q][rules] for q in ["odd_change_dB","even_change_dB"]]
        consistent=[(s.min(axis=1)>1e-8)|(s.max(axis=1)<-1e-8) for s in sides]
        within=pd.concat([s.max(axis=1)-s.min(axis=1) for s in sides],axis=1)
        gap=(sides[0].Original-sides[1].Original).abs()
        grain.append([scenario,visit,len(p),int(consistent[0].sum()),int(consistent[1].sum()),int((consistent[0]&consistent[1]).sum()),
                      int(original.same_direction.sum()),gap.median(),within.mean(axis=1).median(),int((gap>within.max(axis=1)).sum())])
    cases=keep(audit)
    row["discordant_omit_one_run_to_agree"]=((cases.status=="opposite")&(cases.n_run_status_changes>0)).sum()
    row["discordant_omit_one_trial_to_agree"]=((cases.status=="opposite")&(cases.n_trial_status_changes>0)).sum()
    row["concordant_omit_one_run_to_disagree"]=((cases.status=="agree")&(cases.n_run_status_changes>0)).sum()
    rows.append(row)
table=pd.DataFrame(rows)
table.to_csv(out/"sensitivity_by_code.csv",index=False)
pd.DataFrame(grain,columns=["scenario","visit","codes","odd_threshold_stable","even_threshold_stable","both_threshold_stable",
                           "between_halves_sign_agree","between_halves_abs_median_dB","within_half_mean_span_median_dB",
                           "between_gap_above_both_spans"]).to_csv(out/"matched_recording_comparison.csv",index=False)
assert runs.sha256.nunique()==289
key_hash=runs[["key","sha256"]]
by_signal=scores.merge(key_hash,on="key",validate="many_to_one")
repeats=by_signal[by_signal.duplicated(["sha256","rule"],keep=False)].groupby(["sha256","rule"])
assert repeats["shift"].nunique().max()==1 and repeats.eligible.nunique().max()==1
unique=by_signal.drop_duplicates(["sha256","rule"])
orig=unique[unique.rule=="Original"]
all_eligible=unique.groupby("sha256").eligible.all()
u=unique.pivot(index="sha256",columns="rule",values="shift").loc[all_eligible[all_eligible].index]
trial_ids=trials.merge(key_hash,on="key")
summary=dict(unique_recordings=len(orig),unique_trials=len(trial_ids.drop_duplicates(["sha256","trial_index"])),
             four_decreases=int(orig.eligible.sum()),increases=int((orig.eligible&orig.increased).sum()),
             common_four=len(u),reversal=int(((u.max(axis=1)>1e-8)&(u.min(axis=1)<-1e-8)).sum()))
pd.DataFrame([summary]).to_csv(out/"unique_recording_results.csv",index=False)
assert summary==dict(unique_recordings=289,unique_trials=5769,four_decreases=264,increases=138,common_four=251,reversal=37)
print(table[["scenario","codes","recordings","trials","increases","four_decreases","reversal","common_four","post_threshold_stable","post_halves_agree","follow_threshold_stable","follow_codes","follow_halves_agree","follow_halves_n"]].to_string(index=False))
