"""Internal agreement of longitudinal beta changes across disjoint sets of recordings."""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
root = out.parents[1]
methods = ["Filter only", "Relaxed", "Original", "Strict"]
stages = ["ses-post", "ses-follow"]
tol = 1e-8


def direction(x):
    return np.where(x > tol, 1, np.where(x < -tol, -1, 0))


def partitions(n, anchor):
    indexes = range(n)
    for size in sorted({n // 2, n - n // 2}):
        for subset in combinations(indexes, size):
            if not anchor or 0 in subset:
                yield tuple(i for i in indexes if i in subset), tuple(i for i in indexes if i not in subset)


def estimate(values):
    total = values.sum(axis=0)
    return total[0] / total[2] - total[1] / total[3]


def halves(values, split):
    return np.array([estimate(values[list(ix)]) for ix in split])


trials = pd.read_csv(root / "stroke_analysis_full/trial_metrics_all_stages.csv").query("band == 'beta'")
ids = ["subject", "session", "key"]
before = trials[trials.stage == "before"].groupby(ids + ["condition"]).task_vs_baseline_db.agg(["mean","size"]).unstack()
before.columns = ["_".join(c) for c in before.columns]
terms = pd.read_csv(out.parent / "sensitivity/run_results.csv")
zeros = before.reset_index()[ids].assign(rule="Filter only", MI_task=0., MI_baseline=0., rest_task=0., rest_baseline=0.)
data = pd.concat([terms,zeros],ignore_index=True).merge(before.reset_index(),on=ids,validate="many_to_one")
data["MI"] = data.size_MI * (data.mean_MI + data.MI_task + data.MI_baseline)
data["rest"] = data.size_rest * (data.mean_rest - data.rest_task - data.rest_baseline)
assert not data.duplicated(ids+["rule"]).any() and len(data)==293*4
assert data[["MI","rest","size_MI","size_rest"]].notna().all().all() and data[["size_MI","size_rest"]].min().min()>0
full = pd.read_csv(out.parent / "longitudinal_rules/participant_changes.csv").set_index(["subject","endpoint"])
summary = []
details = []
excluded = []
for (subject, endpoint), reference in full.iterrows():
    arrays, keys, sizes = [], [], []
    for visit in ["ses-pre",endpoint]:
        g = data[(data.subject==subject)&(data.session==visit)]
        table = g.set_index(["key","rule"])[["MI","rest","size_MI","size_rest"]].unstack("rule")
        shape = (len(table),4,len(methods))
        arr = table.reindex(columns=pd.MultiIndex.from_product([["MI","rest","size_MI","size_rest"],methods])).to_numpy().reshape(shape)
        arrays.append(arr)
        keys.append(table.index.to_list())
        sizes.append(len(table))
    np.testing.assert_allclose(estimate(arrays[1])-estimate(arrays[0]),reference[methods].astype(float),atol=1.e-8,rtol=0)
    if min(sizes)<2:
        excluded.append([subject,endpoint,*sizes,"Fewer than two recordings at one visit"])
        continue
    # At each visit retain the recording's archived run number, rather than splitting individual trials.
    odds = []
    for visit_keys in keys:
        run_numbers = [int(k.rsplit("-",1)[1]) for k in visit_keys]
        odds.append((tuple(i for i,r in enumerate(run_numbers) if r%2),tuple(i for i,r in enumerate(run_numbers) if not r%2)))
    fixed = halves(arrays[1],odds[1])-halves(arrays[0],odds[0])
    split_lists = [list(partitions(n,anchor=(v==0))) for v,n in enumerate(sizes)]
    visit_halves = [[halves(arr,p) for p in parts] for arr,parts in zip(arrays,split_lists)]
    paired = []
    found_fixed = False
    for (i,pa),(j,pb) in product(enumerate(split_lists[0]),enumerate(split_lists[1])):
        changes = visit_halves[1][j] - visit_halves[0][i]
        if pa==odds[0] and pb==odds[1]:
            found_fixed=True
            np.testing.assert_array_equal(changes,fixed)
        paired.append(changes)
        parts = [",".join(keys[v][i].split("_")[-1][4:] for i in ix) for v,p in enumerate([pa,pb]) for ix in p]
        for m,method in enumerate(methods):
            details.append([subject,endpoint,method,*parts,*changes[:,m]])
    assert found_fixed and len(paired)==len(split_lists[0])*len(split_lists[1])
    changes=np.stack(paired)
    signs = direction(changes)
    assert np.all(signs!=0) and np.all(direction(fixed)!=0)
    equal=(signs[:,0,:]==signs[:,1,:])
    fixed_equal=direction(fixed[0])==direction(fixed[1])
    for m,method in enumerate(methods):
        summary.append([subject,endpoint,method,*sizes,reference[method],*fixed[:,m],fixed_equal[m],
                        abs(fixed[0,m]-fixed[1,m]),len(paired),int(equal[:,m].sum()),equal[:,m].mean()])
rows=pd.DataFrame(summary,columns=["subject","endpoint","method","pre_runs","endpoint_runs","full_change_dB",
    "odd_change_dB","even_change_dB","same_direction","abs_odd_even_gap_dB","partitions","agreeing_partitions","agreement_fraction"])
rows.to_csv(out/"participant_splits.csv",index=False)
pd.DataFrame(details,columns=["subject","endpoint","method","pre_A","pre_B","endpoint_A","endpoint_B","change_A_dB","change_B_dB"]).to_csv(out/"partition_estimates.csv",index=False)
pd.DataFrame(excluded,columns=["subject","endpoint","pre_runs","endpoint_runs","reason"]).to_csv(out/"exclusions.csv",index=False)
stats=[]
for (endpoint,method), group in rows.groupby(["endpoint","method"],sort=False):
    fractions=group.agreement_fraction.to_numpy()
    # Use the same subject resamples for all methods within an endpoint.
    draws=np.random.default_rng(4627).integers(0,len(group),size=(10000,len(group)))
    lo,hi=np.quantile(fractions[draws].mean(1),[.025,.975])
    stats.append([endpoint,method,len(group),int(group.same_direction.sum()),float(np.median(group.abs_odd_even_gap_dB)),
                  fractions.mean(),lo,hi,np.sum(fractions==1),np.sum(fractions==0)])
stats=pd.DataFrame(stats,columns=["endpoint","method","people","fixed_agree","median_abs_gap_dB",
              "balanced_mean","balanced_low","balanced_high","all_splits_agree","no_splits_agree"])
stats.to_csv(out/"summary.csv",index=False)
print(stats.to_string(index=False))
print("Excluded",excluded)
