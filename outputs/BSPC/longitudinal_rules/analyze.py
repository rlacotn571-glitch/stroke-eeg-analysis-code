"""Reconstruct trial-weighted participant contrasts on identical trials and ICA fits."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from population import ambiguous
root = Path("outputs")
out = root / "BSPC/longitudinal_rules"
out.mkdir(parents=True, exist_ok=True)
rules = ["Relaxed", "Original", "Strict"]
methods = ["Filter only", *rules]
visits = ["ses-post", "ses-follow"]
tol = 1e-8
trials = pd.read_csv(root / "stroke_analysis_full/trial_metrics_all_stages.csv").query(
    "band == 'beta' and stage == 'before'"
)
keys = ["subject", "session", "key"]
assert len(trials) == 5849 and not trials.duplicated(["key", "trial_index"]).any()
base = trials.groupby([*keys, "condition"]).task_vs_baseline_db.agg(["size", "mean"]).unstack()
base.columns = ["_".join(c) for c in base.columns]
base = base.reset_index()
scores = pd.read_csv(root / "BSPC/sensitivity/run_results.csv")
assert len(scores) == 293 * 3 and not scores.duplicated(["key", "rule"]).any()
assert set(scores.rule) == set(rules) and len(base) == 293
data = scores.merge(base, on=keys, validate="many_to_one")
assert len(data) == len(scores) and data.notna().all().all()
raw = base.assign(rule="Filter only", MI_task=0, MI_baseline=0, rest_task=0, rest_baseline=0)
data = pd.concat([raw, data], ignore_index=True)
data["MI_total"] = data.size_MI * (data.mean_MI + data.MI_task + data.MI_baseline)
data["rest_total"] = data.size_rest * (data.mean_rest - data.rest_task - data.rest_baseline)
groups = data.groupby(["subject", "session", "rule"])
agg = groups[["MI_total", "rest_total", "size_MI", "size_rest"]].sum()
agg["runs"] = groups.size()
agg["contrast"] = agg.MI_total / agg.size_MI - agg.rest_total / agg.size_rest
table = agg.contrast.unstack("rule")[methods].sort_index()
ref = pd.read_csv(root / "stroke_manuscript/subject_contrasts.csv").query(
    "band == 'beta'").set_index(["subject", "session"]).sort_index()
assert table.index.equals(ref.index)
np.testing.assert_allclose(table[["Filter only","Original"]], ref[["before","iclabel_after"]], atol=1e-8, rtol=0)
table.loc[~table.index.get_level_values('subject').isin(ambiguous)].join(agg.xs("Original", level="rule")[["runs", "size_MI", "size_rest"]]).to_csv(out / "visit_contrasts.csv")

table = table.loc[~table.index.get_level_values('subject').isin(ambiguous)]
wide = table.unstack("session").dropna()
assert len(wide) == 21
rows = []
summary = []
paired = []
rng = np.random.default_rng(20261117)
sample = rng.integers(0, len(wide), size=(10000, len(wide)))
def interval(x):
    v = np.asarray(x)
    return [v.mean(), *np.quantile(v[sample].mean(axis=1), [.025, .975])]

for visit in visits:
    changes = wide.xs(visit, level="session", axis=1) - wide.xs("ses-pre", level="session", axis=1)
    by_rule = changes[rules]
    changes["direction_switched"] = (by_rule.min(axis=1) < -tol) & (by_rule.max(axis=1) > tol)
    assert not (by_rule.abs() <= tol).any().any()
    changes["span"] = by_rule.max(axis=1) - by_rule.min(axis=1)
    rows.append(changes.assign(endpoint=visit).reset_index())
    for method in methods:
        summary.append([visit, method, len(changes), *interval(changes[method])])
    for first, second in [("Strict", "Relaxed"), ("Original", "Filter only")]:
        paired.append([visit, first, second, *interval(changes[first] - changes[second])])
long = pd.concat(rows, ignore_index=True)
long.to_csv(out / "participant_changes.csv", index=False)
stats = pd.DataFrame(summary, columns=["endpoint", "method", "n", "mean", "low", "high"])
stats.to_csv(out / "group_changes.csv", index=False)
pd.DataFrame(paired, columns=["endpoint","first","second","mean","low","high"]).to_csv(out / "paired_differences.csv",index=False)

