"""Plot fixed and exhaustive disjoint-run comparisons for original ICLabel."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

d = Path(__file__).resolve().parent
dest = d.parent / "full_paper_figures"
dest.mkdir(exist_ok=True)
data = pd.read_csv(d/"participant_splits.csv").query("method == 'Original'")
stats = pd.read_csv(d/"summary.csv").query("method == 'Original'").set_index("endpoint")
endpoints = ["ses-post","ses-follow"]
names = ["Post minus pre","Follow-up minus pre"]
ink, teal, orange, gray = "#172F3A","#096C78","#AC421B","#4B636E"
plt.rcParams.update({"font.family":"DejaVu Sans","font.weight":"bold","font.size":10,
                     "axes.titleweight":"bold","axes.labelweight":"bold","text.color":ink,
                     "axes.labelcolor":ink,"axes.edgecolor":gray,"axes.spines.top":False,
                     "axes.spines.right":False,"xtick.color":ink,"ytick.color":ink,"pdf.fonttype":42})
fig=plt.figure(figsize=(7.35,6.45),layout="constrained")
grid=fig.add_gridspec(3,2,height_ratios=[1.,.093,.78],wspace=.06,hspace=.045)
for j,(endpoint,title) in enumerate(zip(endpoints,names)):
    ax=fig.add_subplot(grid[0,j])
    group=data[data.endpoint==endpoint]
    for agree in [True,False]:
        g=group[group.same_direction==agree]
        ax.scatter(g.odd_change_dB,g.even_change_dB,s=26,marker="o" if agree else "^",
                   color=teal if agree else orange,edgecolors="white",linewidths=.55,zorder=3)
    ax.axhline(0,color=gray,lw=.85)
    ax.axvline(0,color=gray,lw=.85)
    ax.plot([-2.5,5.7],[-2.5,5.7],ls=":",color="#86999F",lw=.85)
    ax.set(xlim=(-2.5,5.7),ylim=(-2.5,5.7),aspect="equal",xticks=[-2,0,2,4],yticks=[-2,0,2,4],
           xlabel="Odd runs (dB)",ylabel="Even runs (dB)" if j==0 else "")
    ax.set_title(("A" if j==0 else "B")+"   "+title,loc="left",fontsize=11,pad=7)
    ax.text(.03,.97,f"{int(group.same_direction.sum())}/{len(group)} same direction",
            transform=ax.transAxes,va="top",fontsize=9)
    r=group[group.subject=="sub-22"].iloc[0]
    ax.annotate("P22",(r.odd_change_dB,r.even_change_dB),xytext=(5,6 if j==0 else -11),
                textcoords="offset points",fontsize=9,color=ink)
leg=fig.add_subplot(grid[1,:]);leg.axis("off")
handles=[Line2D([],[],color=color,marker=mark,ls="none",label=name,ms=6)
         for color,mark,name in [(teal,"o","Same direction"),(orange,"^","Opposite directions")]]
leg.legend(handles=handles,loc="center",ncol=2,frameon=False,handlelength=1)
ax=fig.add_subplot(grid[2,:])
for y,(endpoint,name) in enumerate(zip(endpoints,names)):
    group=data[data.endpoint==endpoint]
    for fraction,g in group.groupby("agreement_fraction"):
        offsets=(np.arange(len(g))-(len(g)-1)/2)*.056
        ax.scatter(np.full(len(g),100*fraction),y+offsets,s=24,facecolors="white",
                   edgecolors=teal,linewidths=1.0,zorder=3)
    r=stats.loc[endpoint]
    ax.errorbar(r.balanced_mean*100,y+.30,
                xerr=np.array([[r.balanced_mean-r.balanced_low],[r.balanced_high-r.balanced_mean]])*100,
                marker="D",ms=6,capsize=3,color=orange,lw=1.6)
    ax.text(r.balanced_high*100+2.5,y+.30,f"{r.balanced_mean*100:.1f}%",va="center",fontsize=9)
ax.set(xlim=(-3,104),ylim=(1.6,-.3),xticks=[0,25,50,75,100],yticks=[0,1],yticklabels=names,
       xlabel="Balanced partitions giving the same direction (%)")
ax.xaxis.grid(True,color="#DFE7E9",lw=.7)
ax.set_title("C   Agreement across all balanced partitions",loc="left",fontsize=11)
for ext in ["pdf","png"]:
    fig.savefig(dest/f"Figure_S5.{ext}",dpi=400)
plt.close(fig)
