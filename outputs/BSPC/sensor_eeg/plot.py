"""Publication figures from the matched sensor, waveform and spectral estimates."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.text import Text
import mne
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
out = folder.parent / "full_paper_figures"
out.mkdir(exist_ok=True)
case = "sub-22_ses-pre_run-04"
stages = ["before", "iclabel_after"]
visits = ["ses-pre", "ses-post", "ses-follow"]
titles = ["Filter only", "ICLabel cleaned", "Cleaned − filter only"]
visit_titles = ["Pre-training", "Post-training", "Follow-up"]
ink, blue, orange, muted = "#172B35", "#08677A", "#A94020", "#526872"
cmap = "RdBu_r"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "font.weight":"bold",
                     "axes.labelweight":"bold", "axes.titleweight":"bold", "axes.spines.top":False,
                     "axes.spines.right":False, "axes.labelcolor":ink, "axes.edgecolor":muted,
                     "text.color":ink, "xtick.color":ink, "ytick.color":ink, "pdf.fonttype":42,
                     "figure.facecolor":"white", "axes.unicode_minus":True})
rows = pd.read_csv(folder / "sensor_condition_means.csv")
channels = rows.columns[6:].to_list()
info = mne.create_info(channels, 250., "eeg").set_montage("standard_1005")
montage = info.get_montage().get_positions()
origin = (montage["lpa"] + montage["rpa"]) / 2.
xyz = np.stack([montage["ch_pos"][ch] - origin for ch in channels])
azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
colatitude = np.arctan2(np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2])
radius = .94 * colatitude / colatitude.max()
xy = np.column_stack([radius * np.cos(azimuth), radius * np.sin(azimuth)])
mask = np.isin(channels, ["FCz", "Cz", "C1", "C2"])
assert len(channels) == 59 and mask.sum() == 4 and len(rows) == 293 * 4


def full_scale(values):
    peak = np.max(np.abs(values))
    size = 10. ** np.floor(np.log10(peak))
    return float(np.ceil(2. * peak / size) * size / 2.)


def topology(ax, values, limit):
    return mne.viz.plot_topomap(values, xy, axes=ax, show=False, cmap=cmap,
              contours=0, image_interp="linear", extrapolate="head", border="mean",
              vlim=(-limit, limit), sphere=(0.,0.,0.,1.), res=128, sensors=True, mask=mask,
              mask_params=dict(marker="o", markersize=3.5, markerfacecolor="white",
                               markeredgecolor=ink, markeredgewidth=.85, linewidth=0))


def colorbar(fig, rect, lim, orientation="vertical"):
    c = fig.colorbar(ScalarMappable(norm=Normalize(-lim, lim), cmap=cmap),
                     cax=fig.add_axes(rect), orientation=orientation, ticks=[-lim, 0, lim])
    c.outline.set_visible(False)
    c.ax.tick_params(length=2, pad=3)
    if orientation == "vertical":
        c.ax.set_title("dB", fontsize=9, pad=5)
    return c


def export(fig, name):
    for item in fig.findobj(Text):
        item.set_fontweight("bold")
    fig.savefig(out / (name + ".pdf"), dpi=400)
    fig.savefig(out / (name + ".png"), dpi=420)
    plt.close(fig)


def before_after_change(frame):
    two = frame.loc[stages].to_numpy()
    return np.stack([two[0], two[1], two[1] - two[0]])


case_rows = rows[rows.key == case].set_index(["stage", "condition"])[channels]
maps = before_after_change(case_rows.xs("MI", level="condition") - case_rows.xs("rest", level="condition"))
tfr = np.load(folder / "case_spectra.npz")
power = tfr["contrast"]
power = np.concatenate([power, (power[1] - power[0])[None]])
wave = np.load(folder / "case_timecourse.npz")
fig = plt.figure(figsize=(7.55, 6.45))
fig.text(.025,.971,"A   Central beta contrast across the trial",size=12)
ax = fig.add_axes([.108,.755,.84,.153])
eog = fig.add_axes([.108,.672,.84,.067], sharex=ax)
for axis in [ax, eog]:
    axis.axvspan(-3.5,-1.5,color="#EAF0F2",lw=0)
    axis.axvspan(.5,4.5,color="#E9F2EC",lw=0)
    axis.axvline(0,color=ink,lw=.9,ls="--")
    axis.set_xlim(-3.5,4.5)
    axis.tick_params(axis="both",length=2,pad=2)
for signal, color, label, width in zip(wave["contrast"], [blue,orange], titles, [1.25,1.45]):
    ax.plot(wave["time"],signal,color=color,label=label,lw=width)
ax.legend(frameon=False,ncol=2,loc="lower right",bbox_to_anchor=(1.,1.003),borderaxespad=0,
          handlelength=1.6,columnspacing=1.4)
ax.set_ylabel("Contrast (dB)")
ax.axhline(0,color=muted,lw=.55,ls=":")
ax.tick_params(labelbottom=False,bottom=False)
ax.spines["bottom"].set_visible(False)
eog.fill_between(wave["time"],*wave["bounds"],color="#DBCDC8",lw=0)
eog.axhline(0,color=muted,lw=.65,ls=":")
eog.plot(wave["time"],wave["contrast"][1]-wave["contrast"][0],color=ink,lw=.95)
eog.set_ylabel("Shift (dB)",labelpad=3)
eog.set(xticks=[-3,-2,-1,0,1,2,3,4],xlabel="Time from task onset (s)")
eog.xaxis.labelpad=1
fig.text(.025,.562,"B   Imagery − rest beta contrast at 59 electrodes",size=12)
centers = [.20,.475,.75]
limit = full_scale(maps)
for center, name, values in zip(centers, titles, maps):
    ax = fig.add_axes([center-.105,.310,.21,.213])
    topology(ax, values, limit)
    ax.set_title(name,fontsize=10,pad=1)
colorbar(fig,[.917,.328,.013,.17],limit)
fig.text(.025,.268,"C   Imagery − rest power at the four central electrodes",size=12)
limit = full_scale(power)
for i,(center,values) in enumerate(zip(centers,power)):
    ax=fig.add_axes([center-.113,.092,.226,.150])
    ax.pcolormesh(tfr["time"],tfr["freq"],values,shading="nearest",cmap=cmap,
                  vmin=-limit,vmax=limit,rasterized=True)
    ax.axvline(0,color="black",ls="--",lw=.85)
    for hz in [13,30]:
        ax.axhline(hz,color="black",lw=.60,ls=":")
    ax.set(xlim=(-3.5,4.5),ylim=(8,40),xticks=[-3,0,4],yticks=[8,13,30,40])
    if i==0:
        ax.set_ylabel("Frequency (Hz)",labelpad=2)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(length=2,pad=2)
colorbar(fig,[.917,.095,.013,.143],limit)
fig.text(.475,.023,"Time from task onset (s)",ha="center",size=10)
export(fig,"Figure_2")

# Pool trials in each condition per person and visit, then weight the same people equally.
counts = rows.groupby("subject").session.nunique()
from population import ambiguous
ids = counts[counts == 3].index.difference(ambiguous)
sub = rows[rows.subject.isin(ids)].copy()
group_keys = ["subject", "session", "stage", "condition"]
weights = sub.groupby(group_keys).n.sum()
sums = sub[channels].mul(sub.n, axis=0).groupby([sub[k] for k in group_keys]).sum()
means = sums.div(weights,axis=0)
person = means.xs("MI",level="condition") - means.xs("rest",level="condition")
group = person.groupby(["session","stage"]).mean()
grid = np.stack([before_after_change(group.loc[visit]) for visit in visits])
limits = [full_scale(grid[:, :2]), full_scale(grid[:, 2])]
assert len(ids) == 21 and sub.key.nunique() == 253 and sub[sub.stage=="before"].n.sum() == 5052
vis = pd.read_csv(folder.parent / "longitudinal_rules/visit_contrasts.csv")
vis = vis[vis.subject.isin(ids)].copy()
vis["shift"] = vis["Original"] - vis["Filter only"]
shifts = vis.pivot(index="subject",columns="session",values="shift")[visits]
fig=plt.figure(figsize=(7.25,7.65))
fig.text(.04,.96,"A   Sensor-level imagery − rest beta contrast",size=13)
fig.text(.04,.927,"21 participants with all three visits  ·  253 recordings  ·  5,052 trials",size=10)
centers=[.295,.565,.83]
for center,name in zip(centers,titles):
    fig.text(center,.882,name,ha="center",size=10)
for i,(visit,heading) in enumerate(zip(visits,visit_titles)):
    y=.71-i*.173
    fig.text(.032,y+.066,heading,va="center",size=10)
    count=sub[(sub.session==visit)&(sub.stage=="before")]
    fig.text(.032,y+.033,f"{count.key.nunique()} runs",va="center",size=9,color=muted)
    for j,center in enumerate(centers):
        ax=fig.add_axes([center-.1,y,.2,.155])
        topology(ax,grid[i,j],limits[int(j==2)])
colorbar(fig,[.28,.327,.30,.012],limits[0],"horizontal")
colorbar(fig,[.756,.327,.15,.012],limits[1],"horizontal")
fig.text(.43,.282,"Contrast (dB)",ha="center",size=9)
fig.text(.83,.282,"Change (dB)",ha="center",size=9)
fig.text(.04,.23,"B   Cleaning shift in the four-electrode beta contrast",size=12)
ax=fig.add_axes([.14,.062,.75,.133])
ax.axhline(0,ls="--",color=muted,lw=.8)
for values in shifts.to_numpy():
    ax.plot(range(3),values,color="#99AAB0",alpha=.58,lw=.7,marker="o",ms=2)
ax.plot(range(3),shifts.mean(),color=orange,marker="D",ms=5,lw=1.6,label="Participant mean")
ax.set(xlim=(-.25,2.65),xticks=range(3),xticklabels=visit_titles,ylabel="Change (dB)")
ax.legend(loc="center right",bbox_to_anchor=(1.14,.98),frameon=False,handlelength=1.2)
export(fig,"Figure_S4")
