"""Publication panels for the full paper, using only tabulated study measurements."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd

root = Path("outputs")
base = root / "BSPC"
out = base / "full_paper_figures"
out.mkdir(parents=True, exist_ok=True)
blue, orange, slate = "#08677A", "#A94020", "#4E606B"
light, ink, grid = "#A9B8BC", "#172B35", "#DBE3E5"
plt.rcParams.update({
    "font.family":"DejaVu Sans", "font.size":9.5, "axes.labelsize":9.5,
    "font.weight":"bold", "axes.labelweight":"bold",
    "xtick.labelsize":9, "ytick.labelsize":9, "axes.spines.top":False,
    "axes.spines.right":False, "axes.edgecolor":slate, "text.color":ink,
    "axes.labelcolor":ink, "xtick.color":ink, "ytick.color":ink,
    "axes.titlesize":10, "axes.titleweight":"bold", "axes.titlelocation":"left",
    "pdf.fonttype":42, "savefig.facecolor":"white", "lines.solid_capstyle":"round",
})
rules = ["Relaxed", "Original", "Strict"]
terms = ["MI_task","MI_baseline","rest_task","rest_baseline"]

def save(fig,number):
    for label in fig.findobj(Text):
        label.set_fontweight("bold")
        label.set_fontsize(max(label.get_fontsize(), 9))
    for ext in ["png","pdf"]:
        fig.savefig(out/f"Figure_{number}.{ext}",dpi=600,bbox_inches="tight",pad_inches=.09)
    plt.close(fig)

def title(ax,letter,heading):
    ax.set_title(f"{letter}   {heading}",loc="left",pad=8)

def signed(v,n=3):
    return f"{v:+.{n}f}".replace("-","−")

source = pd.read_csv(base/"Figure_1_source.csv").set_index("key")
from population import unique_keys, ambiguous
source=source.loc[source.index.isin(unique_keys)]
case = source.loc["sub-22_ses-pre_run-04"]
assert np.isclose(case[terms].astype(float).sum(),case.delta)
fig = plt.figure(figsize=(7.5,4.95),layout="constrained")
gs = fig.add_gridspec(2,2,height_ratios=[.7,1.75],hspace=.14,wspace=.09)
a,b,c = fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])
title(a,"A","Windows relative to task onset")
a.set(xlim=(-4.05,5.05),ylim=(-.57,.92))
a.axis("off")
a.hlines(0,-4,5,color=slate,lw=1)
for left,right,color,label in [(-3.5,-1.5,"#DDEBED","Baseline"),(.5,4.5,"#BCDAD9","Task window")]:
    a.add_patch(Rectangle((left,.03),right-left,.37,facecolor=color,edgecolor=slate,lw=.7))
    a.text((left+right)/2,.215,label,ha="center",va="center",weight="bold")
    for end in [left,right]:
        a.plot([end,end],[-.06,.02],color=slate,lw=.8)
        a.text(end,-.14,f"{end:g} s".replace("-","−"),ha="center",va="top",fontsize=8)
for event,text,x,ha in [(-1,"Instruction (approx. −1 s)",-.97,"right"),(0,"Task onset (0 s)",.03,"left")]:
    a.plot([event,event],[0,.66],color=ink,lw=.8,ls="--")
    a.text(x,.72,text,fontsize=8.4,ha=ha,va="bottom")

title(b,"B","Shift across 289 recordings")
for flag,color,style,label in [(False,slate,"--","Other recordings"),(True,blue,"-","Four decreases")]:
    v=source.loc[source.all_four_decreased==flag,"delta"]
    b.hist(v,bins=np.linspace(-1.55,1.55,32),histtype="step",color=color,ls=style,lw=1.6,
           label=f"{label} ({len(v)})")
b.axvline(0,color=slate,lw=.7)
b.set(xlabel="Change in beta contrast (dB)",ylabel="Recordings per bin")
b.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.25),handlelength=1.3)
title(c,"C","Participant 22 · pre-training run 4")
level=0.
for i,value in enumerate([*case[terms].astype(float),case.delta]):
    if i==4:level=0.
    color=slate if i==4 else blue if value>0 else orange
    c.bar(i,value,bottom=level,color=color,width=.83)
    c.text(i,level+value/2,signed(value,2),va="center",ha="center",color="white",fontsize=8,weight="bold")
    level+=value
    if i<3:c.plot([i+.415,i+.585],[level,level],color=slate,lw=.8)
c.axhline(0,color=slate,lw=.7)
c.set(xticks=range(5),xticklabels=["MI\ntask","MI\nbase","Rest\ntask","Rest\nbase","Net"],
      ylabel="Cumulative change (dB)",ylim=(-.62,2.65))
save(fig,1)

records=pd.read_csv(base/"record_review/record_spans.csv").set_index("key")
fig=plt.figure(figsize=(7.5,5.05),layout="constrained")
gs=fig.add_gridspec(2,2,height_ratios=[.76,1.45],hspace=.15,wspace=.1)
a,b,c=fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])
title(a,"A","What happened to the original direction?")
for y,sign,term in [(1,True,"positive"),(0,False,"negative")]:
    g=records[(records.Original>0)==sign]
    stable=int((~g.switch).sum())
    count=int(g.switch.sum())
    a.barh(y,stable,height=.5,color="#425E68")
    a.barh(y,count,left=stable,height=.5,color=orange)
    a.text(stable/2,y,f"{stable} stayed {term}",va="center",ha="center",color="white",fontsize=8.5,weight="bold")
    a.text(stable+count/2,y,str(count),va="center",ha="center",color="white",weight="bold")
    other="negative" if sign else "positive"
    a.text(len(g)+2,y,f"became {other}",va="center",fontsize=8.4,color=orange)
a.set(xlim=(0,165),ylim=(-.45,1.48),yticks=[1,0],yticklabels=["Originally positive (132)","Originally negative (119)"],xticks=[])
a.tick_params(axis="y",length=0)
a.spines[["left","bottom"]].set_visible(False)
title(b,"B","Original shift and threshold range")
for flag,color,label,marker in [(False,slate,"Stable (214)","o"),(True,orange,"Switched (37)","^")]:
    g=records[records.switch==flag]
    b.scatter(g.Original,g.span,marker=marker,color=color,alpha=.72,s=15,lw=0,label=label)
b.axvline(0,ls="--",color=slate,lw=.7)
b.set(xlabel="Original shift (dB)",ylabel="Range across thresholds (dB)",ylim=(-.035,None))
b.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.20),handletextpad=.22,ncols=2,columnspacing=.6)
title(c,"C","The 37 switching recordings")
switches=records[records.switch]
examples=switches.nlargest(3,"span").index
for selected in [False,True]:
    for _,r in switches[switches.index.isin(examples)==selected].iterrows():
        c.plot(range(3),r[rules].astype(float),color=orange if selected else light,
               lw=1.65 if selected else .74,marker="o" if selected else None,ms=3)
c.axhline(0,ls="--",color=slate,lw=.7)
c.set(xticks=range(3),xticklabels=rules,ylabel="Cleaned minus filter-only (dB)")
save(fig,3)

reps=pd.read_csv(root/"stroke_mechanism/repeat_scale.csv").query("pair=='iclabel_after-before'")
reps=reps[~reps.subject.isin(ambiguous)]
vis=pd.read_csv(base/"longitudinal_rules/visit_contrasts.csv").set_index(["subject","session"])
changes=pd.read_csv(base/"longitudinal_rules/participant_changes.csv")
fig,axes=plt.subplots(2,2,figsize=(7.5,6.25),layout="constrained",gridspec_kw={"hspace":.09,"wspace":.08})
(a,b),(c,d)=axes
title(a,"A","Within-visit medians · 66 visits")
for visit,marker,color,label in [("ses-pre","o",blue,"Pre"),("ses-post","^",orange,"Post"),("ses-follow","s",slate,"Follow-up")]:
    g=reps[reps.session==visit]
    a.scatter(g.median_repeat_difference,g.median_cleaning_shift,marker=marker,color=color,s=17,alpha=.8,lw=0,label=label)
guide=np.ceil(10*reps.median_cleaning_shift.max())/10+.12
a.plot([0,guide],[0,guide],ls="--",color=slate,lw=.85)
a.set(xlabel="Between-run difference (dB)",ylabel="Absolute cleaning shift (dB)",ylim=(-.03,guide))
a.legend(frameon=False,fontsize=7.6,loc="upper right",ncol=1)
title(b,"B","Participant 22 · four runs per visit")
for method,color,linestyle,marker in [("Filter only",slate,"--","o"),("Original",blue,"-","D")]:
    vals=np.array([vis.loc[("sub-22",v),method] for v in ["ses-pre","ses-post"]])
    v=vals[1]-vals[0]
    label=("Filter only" if method=="Filter only" else "ICLabel")+"  "+signed(v)+" dB"
    b.plot([0,1],vals,color=color,ls=linestyle,lw=1.6,marker=marker,ms=5,label=label)
    for i,val in enumerate(vals):
        offset=-12 if ((method=="Filter only" and i==0) or (method=="Original" and i==1)) else 7
        b.annotate(signed(val),(i,val),xytext=(0,offset),textcoords="offset points",ha="center",color=color,fontsize=7.8)
b.axhline(0,color=slate,lw=.6)
b.set(xticks=[0,1],xticklabels=["Pre-training","Post-training"],xlim=(-.25,1.24),ylim=(-1.75,1.22),
      ylabel="Visit beta contrast (dB)")
b.legend(loc="upper left",bbox_to_anchor=(0,1.01),frameon=False,fontsize=7.7,title="Post minus pre",title_fontsize=8)
lo=changes[rules].min().min()-.22
hi=changes[rules].max().max()+.24
for ax,ep,letter,heading in [(c,"ses-post","C","Post minus pre · 21 participants"),(d,"ses-follow","D","Follow-up minus pre · 21 participants")]:
    title(ax,letter,heading)
    sub=changes[changes.endpoint==ep]
    for _,r in sub.sort_values("direction_switched").iterrows():
        ax.plot(range(3),r[rules].astype(float),lw=1.55 if r.direction_switched else .79,
                color=orange if r.direction_switched else light,marker="s" if r.direction_switched else None,ms=3.5)
    ax.axhline(0,color=slate,lw=.7,ls="--",zorder=0)
    ax.plot(range(3),sub[rules].mean(),color=blue,marker="D",lw=1.9,ms=4.5)
    ax.set(xlim=(-.13,2.13),ylim=(lo,hi),xticks=range(3),xticklabels=rules,
           ylabel="Visit change (dB)")
fig.legend(handles=[Line2D([],[],color=light,label="Participant"),Line2D([],[],color=blue,marker="D",label="Mean")],loc="outside lower center",ncols=2,frameon=False)
save(fig,4)

ph=pd.read_csv(root/"phantom_validation/results.csv").query("band=='beta'")
names=["Brain","Eyes","Facial","Neck","Walking","All"]
labels=["Brain only","Eyes","Facial","Neck","Walking","Combined"]
fig,(a,b,c)=plt.subplots(1,3,figsize=(7.6,3.4),layout="constrained",
                         gridspec_kw={"width_ratios":[1,.53,1.05],"wspace":.04},sharey=True)
for ax,metric,head,letter in [(a,"nrmse","Reconstruction error","A"),(c,"gain","Projection gain","C")]:
    title(ax,letter,head)
    v=ph.pivot(index="condition",columns="method",values=metric).loc[names]
    for j,(_,row) in enumerate(v.iterrows()):
        x,y=row["filter"],row["iclabel"]
        ax.plot([x,y],[j,j],color=slate,lw=1.1)
        ax.scatter(x,j,marker="o",s=44,facecolor="white",edgecolor=slate,linewidth=1.1,zorder=3)
        ax.scatter(y,j,marker="D",s=15,color=blue,zorder=4)
    ax.tick_params(axis="y",length=0)
a.set(yticks=range(6),yticklabels=labels,ylim=(5.5,-.5),xlabel="Normalized RMS error")
c.axvline(1,color=slate,lw=.8,ls="--")
c.set(xlabel="Gain (unit target = 1)",xticks=[.86,.90,.94,.98,1.02])
c.spines["left"].set_visible(False)
title(b,"B","Error change")
v=ph.pivot(index="condition",columns="method",values="nrmse").loc[names]
for j,r in enumerate(v.itertuples()):
    change=100*(r.iclabel-r.filter)/r.filter
    label=f"{change:+.1f}%".replace("-","−") if abs(change)>.05 else "0.0%"
    b.text(.45,j,label,ha="center",va="center",color=orange if change>0 else ink,fontsize=9.5,
           weight="bold" if abs(change)>=10 else "normal")
b.set(xlim=(0,1),xticks=[])
b.spines[["left","bottom"]].set_visible(False)
b.tick_params(left=False)
fig.legend(handles=[Line2D([],[],color=slate,marker="o",mfc="white",ls="",label="Filter only"),
                    Line2D([],[],color=blue,marker="D",ms=5,ls="",label="Original ICLabel")],
           loc="outside lower center",ncols=2,frameon=False,fontsize=8)
save(fig,6)


totals=pd.read_csv(base/"sensitivity/summary.csv").set_index("rule").loc[rules]
fractions=pd.read_csv(base/"sensitivity/participants.csv").pivot(index="subject",columns="rule",values="fraction")
fig,(a,b)=plt.subplots(1,2,figsize=(7.5,4.7),layout="constrained",gridspec_kw={"wspace":.08})
x=np.arange(3)
title(a,"A","Recordings at each threshold")
for series,color,label in [(totals.runs,"#DBE1E4","All recordings"),(totals.eligible,"#8AB3BC","Four decreases"),(totals.increased,blue,"Contrast rose among eligible")]:
    a.bar(x,series,.69,color=color,label=label)
for i,row in enumerate(totals.itertuples()):
    a.text(i,row.increased/2,f"{row.increased}/{row.eligible}",color="white",ha="center",va="center")
a.set(ylabel="Recordings",ylim=(0,320))
a.legend(loc="upper center",bbox_to_anchor=(.5,-.2),frameon=False,handlelength=1.2)
title(b,"B","Within-participant proportions")
for _,row in fractions[rules].iterrows():
    b.plot(x,row*100,color="#87999E",lw=.8,marker="o",ms=2)
b.errorbar(x,totals.equal_mean*100,yerr=np.array([totals.equal_mean-totals.low,totals.high-totals.equal_mean])*100,
           color=orange,fmt="D",capsize=4,label="Equal-weight mean\nand 95% interval")
b.set(ylabel="Eligible runs with contrast increase (%)",ylim=(-4,104))
b.legend(loc="upper center",bbox_to_anchor=(.5,-.2),frameon=False)
for ax in [a,b]:
    ax.set_xticks(x,["Relaxed\n.80 / .20","Original\n.90 / .10","Strict\n.95 / .05"])
save(fig,"S1")
