"""Figure 5: signed visit changes under paired threshold and recording comparisons."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

base = Path("outputs/BSPC")
out = base / "full_paper_figures"
out.mkdir(parents=True, exist_ok=True)
splits = pd.read_csv(base / "run_splits/participant_splits.csv")
summary = pd.read_csv("tables/within_visit_comparisons.csv").set_index("visit")
subjects = pd.read_csv(base / "split_attribution/participants.csv").set_index("subject")
substitutions = pd.read_csv(base / "split_attribution/run_cleaning_effect.csv").set_index("key")
methods = ["Relaxed", "Original", "Strict"]
stable, crossed, ink = "#526f7b", "#ad3719", "#17313b"
odd_color, even_color = "#086d78", "#993719"
# The full rebuild draws other panels first. Give this figure the same styling in isolation.
plt.rcdefaults()
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "text.color": ink,
    "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink, "axes.edgecolor": ink,
})

fig = plt.figure(figsize=(7.55, 7.2))
grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.92], left=.102, right=.975,
                        bottom=.115, top=.862, hspace=1.03, wspace=.26)
fig.text(.102, .968, "A   Same runs, vary threshold", fontsize=12)
fig.text(.102, .940, "Two lines per person · one per run set", fontsize=9)
fig.text(.592, .968, "B   Same threshold, vary runs", fontsize=12)
fig.text(.592, .940, "One line per participant · original ICLabel", fontsize=9)
same_line = Line2D([], [], color=stable, lw=1.4, label="Same sign")
cross_line = Line2D([], [], color=crossed, lw=1.7, ls="--", label="Crosses zero")
fig.legend(handles=[same_line, cross_line], bbox_to_anchor=(.99,1.013),
           loc="lower right", frameon=False, ncols=2, fontsize=8.3)

def signed_trajectories(ax, values, ticklabels):
    """Each row is one line. A sign change requires both a positive and a negative estimate."""
    assert np.isfinite(values).all() and np.all(np.abs(values)>1.e-8)
    crossing = (values.min(axis=1)<0) & (values.max(axis=1)>0)
    xx = np.arange(values.shape[1])
    ax.axhline(0, color="#253944", lw=.95, zorder=1)
    for changed in [False, True]:
        yy = values[crossing==changed]
        if len(yy):
            ax.plot(xx, yy.T, ls="--" if changed else "-", marker="o", ms=2.2,
                    lw=1.25 if changed else .75, alpha=.96 if changed else .69,
                    color=crossed if changed else stable, zorder=3 if changed else 2)
    ax.set(xticks=xx, xticklabels=ticklabels, xlim=(-.12, xx[-1]+.12),
           ylim=(-2.5, 3.6), yticks=[-2,0,2])
    ax.tick_params(length=3, pad=3)
    return crossing

for row, (visit, endpoint) in enumerate([("ses-post","Post − pre"), ("ses-follow","Follow-up − pre")]):
    numbers = summary.loc[visit]
    group = splits[(splits.endpoint==visit) & splits.method.isin(methods)]
    odd = group.pivot(index="subject", columns="method", values="odd_change_dB")[methods]
    even = group.pivot(index="subject", columns="method", values="even_change_dB").reindex(index=odd.index,columns=methods)
    n = len(odd)
    assert n == numbers.participants
    a, b = (fig.add_subplot(grid[row, col]) for col in range(2))
    cross_a = signed_trajectories(a, np.vstack([odd.to_numpy(),even.to_numpy()]), methods)
    cross_b = signed_trajectories(b, np.column_stack([odd.Original, even.Original]), ["Odd runs","Even runs"])
    assert sum(~(cross_a[:n] | cross_a[n:])) == numbers.both_stable
    assert sum(~cross_b) == numbers.odd_even_agree
    a.set_ylabel("Visit change (dB)")
    a.set_title(f"{endpoint}  ·  {int(numbers.both_stable)}/{n} stable in both sets", loc="left", fontsize=9, pad=6)
    b.set_title(f"{endpoint}  ·  {int(numbers.odd_even_agree)}/{n} agree across sets", loc="left", fontsize=9, pad=6)
    a.set_xlabel(f"Median threshold range   {numbers.median_within_set_range_dB:.3f} dB",loc="left",fontsize=8.5,labelpad=5)
    b.set_xlabel(f"Median odd–even gap   {numbers.median_odd_even_gap_dB:.3f} dB",loc="left",fontsize=8.5,labelpad=5)

c = fig.add_subplot(grid[2,:])
p = subjects.loc["sub-12"]
h = substitutions.loc["sub-12_ses-post_run-01"]
values = np.array([[p.filter_odd,p.filter_even],[p.odd,p.even],
                   [h.odd_with_run_filtered,h.even_with_run_filtered]])
c.set_title("C   P12 · same trials, reverse cleaning in one post-training run", loc="left", fontsize=10.6, pad=8)
c.axhline(0, color="#253944", lw=.95)
for col,(color,label,marker) in enumerate([(odd_color,"Odd runs","o"),(even_color,"Even runs","^")]):
    c.plot(range(3),values[:,col],color=color,marker=marker,ms=5,lw=1.65,label=label)
    for j,value in enumerate(values[:,col]):
        c.annotate(f"{value:+.3f}".replace("-","−"), (j,value), xytext=(0,7 if col==0 else -14),
                   textcoords="offset points",ha="center",color=color, fontsize=8.5)
c.set(xlim=(-.18,2.18),ylim=(-1.70,.49),yticks=[-1.5,-1,-.5,0],xticks=[0,1,2],
      xticklabels=["Filter only\nin every run","Original ICLabel\nin every run","Filter only in post run 1\nICLabel elsewhere"],
      ylabel="Post − pre (dB)")
c.legend(loc="center left",frameon=False,ncols=2,fontsize=8.5)
for label in fig.findobj(Text):
    label.set_fontweight("bold")
for ext in ["pdf","png"]:
    fig.savefig(out/("Figure_5."+ext),dpi=440,bbox_inches="tight",pad_inches=.08)
plt.close(fig)
