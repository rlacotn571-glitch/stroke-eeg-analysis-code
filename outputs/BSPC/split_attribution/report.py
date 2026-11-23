"""Three-page companion to the post-training disjoint-run comparison."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

out=Path(__file__).resolve().parent
people=pd.read_csv(out/"participants.csv").set_index("subject")
cases=people[people.status=="opposite"]
vis=pd.read_csv(out/"visit_windows.csv").set_index(["subject","session"])
runs=pd.read_csv(out/"run_influence.csv").set_index("key")
hybrid=pd.read_csv(out/"run_cleaning_effect.csv").set_index("key")
restored=pd.read_csv(out/"component_restoration.csv")
sessions=["ses-pre","ses-post"]
terms=["MI_task","MI_baseline","rest_task","rest_baseline"]
ink="#162F37"; muted="#445963"; grid="#D2DEE0"; teal="#086E76"
plt.rcParams.update({"font.family":"DejaVu Sans","font.weight":"bold","font.size":10,
                     "text.color":ink,"pdf.fonttype":42})
def page(num,heading,sub):
    fig=plt.figure(figsize=(10,7.1),facecolor="white")
    fig.text(.055,.951,f"SUPPLEMENT S6   /   {num} OF 3",fontsize=10,color=teal)
    fig.text(.055,.885,heading,fontsize=19)
    fig.text(.055,.838,sub,fontsize=10)
    return fig
def footer(fig):
    fig.text(.055,.031,"nm000270 v1.0.6   ·   Central beta, 13–30 Hz   ·   Post-training minus pre-training",
             fontsize=9,color=muted)
def number(x,places=2):
    return f"{x:+.{places}f}".replace("-","−")
def matrix(fig,box,ncols):
    ax=fig.add_axes(box); ax.set(xlim=(-.5,ncols-.5),ylim=(7.5,-1.95))
    ax.set_axis_off()
    return ax
def tile(ax,j,i,v,scale,outlined=False):
    is_missing=pd.isna(v)
    cmap=matplotlib.colormaps["RdBu_r"]
    col="#F1F4F4" if is_missing else cmap(Normalize(-scale,scale)(v))
    ax.add_patch(Rectangle((j-.482,i-.46),.964,.92,facecolor=col,edgecolor=ink if outlined else "white",
                           lw=1.7 if outlined else .45))
    label="·" if is_missing else number(v)
    ax.text(j,i,label,ha="center",va="center",fontsize=9.4,
            color="white" if not is_missing and abs(v)/scale>=.64 else ink)
def top(ax,x,txt):
    ax.text(x,-.85,txt,ha="center",va="center",fontsize=9.4)
def band(ax,x,txt):
    ax.text(x,-1.65,txt,ha="center",va="center",fontsize=10.5,color=teal)
def note(fig,y,line):
    fig.text(.055,y,line,fontsize=10)

with PdfPages(out/"Supplement_S6.pdf") as pdf:
    fig=page(1,"Where the disagreement entered the beta estimate",
             "Each row is a participant with opposite signs in the odd and even runs under original ICLabel.")
    ax=matrix(fig,[.042,.29,.924,.51],11)
    top(ax,0,"Person"); top(ax,1,"Odd"); top(ax,2,"Even")
    band(ax,1.5,"Visit change (dB)")
    for k,name in enumerate(["Pre-training contribution","Post-training contribution"]):
        band(ax,4.5+4*k,name)
        for j,name in enumerate(["MI\ntask","MI\nbase","Rest\ntask","Rest\nbase"]):top(ax,3+4*k+j,name)
    for i,(person,row) in enumerate(cases.iterrows()):
        for j,txt in enumerate(["P"+person[-2:],number(row.odd),number(row.even)]):
            ax.text(j,i,txt,va="center",ha="center",fontsize=10.5)
        values=np.concatenate([vis.loc[(person,s),terms].to_numpy(float) for s in sessions])
        for j,v in enumerate(values):tile(ax,3+j,i,v,2.5,j==np.argmax(abs(values)))
    note(fig,.235,"Read across. The eight signed cells sum to odd minus even change (dB), before rounding.")
    note(fig,.189,"Each outline marks the largest absolute term. Blue lowers the gap and red raises it.")
    note(fig,.143,"MI = motor imagery. Relative to onset, task = +0.5 to +4.5 s and base = −3.5 to −1.5 s.")
    note(fig,.097,"For P20, the post-training rest baseline opposed the gap and the rest-task term offset it.")
    footer(fig); pdf.savefig(fig); plt.close(fig)

    fig=page(2,"Which recordings changed the result when omitted?",
             "One recording was omitted at a time. Its remaining half was reweighted using all retained trials.")
    ax=matrix(fig,[.04,.29,.93,.51],11)
    top(ax,0,"Person")
    for k,name in enumerate(["Pre-training run","Post-training run"]):
        band(ax,3+5*k,name)
        for j in range(5): top(ax,1+5*k+j,f"{j+1:02d}")
    for i,(person,row) in enumerate(cases.iterrows()):
        ax.text(0,i,"P"+person[-2:],va="center",ha="center",fontsize=10.5)
        for k,s in enumerate(sessions):
            for run in range(1,6):
                key=f"{person}_{s}_run-{run:02d}"
                exists=key in runs.index
                val=runs.loc[key,"signed_gap_influence"] if exists else np.nan
                tile(ax,1+5*k+run-1,i,val,1.8,exists and runs.loc[key,"status_without"]=="agree")
    note(fig,.235,"Cell value = observed odd–even gap minus the gap after omission (dB). Dots mark absent runs.")
    note(fig,.189,"Outlined cells give matching directions after that one omission. Each omission is a separate check.")
    note(fig,.143,"With two equally weighted runs in a half, their omissions can have equal and opposite effects.")
    note(fig,.097,"Among the 13 initially concordant participants, eight changed to opposite signs after at least one omission.")
    footer(fig);pdf.savefig(fig);plt.close(fig)

    fig=page(3,"What changed when only the cleaning was reversed?",
             "All trials and run assignments were held fixed. Each cell gives a post-minus-pre beta change in dB.")
    columns=[.055,.47,.60,.76,.89]
    fig.text(.55,.759,"Participant 09",ha="center",color=teal,fontsize=12)
    fig.text(.83,.759,"Participant 12",ha="center",color=teal,fontsize=12)
    for x,t in zip(columns,["Reconstruction","Odd","Even","Odd","Even"]):
        fig.text(x,.707,t,ha="left" if x==columns[0] else "center",fontsize=10)
    mat=[]
    for subject in ["sub-09","sub-12"]:
        row=people.loc[subject]
        change=hybrid.loc[subject+"_ses-post_run-01"]
        mat.append([(row.filter_odd,row.filter_even),(row.odd,row.even),
                    (change.odd_with_run_filtered,change.even_with_run_filtered)])
    for j,label in enumerate(["Filter only in every run","Original ICLabel in every run","Filter only in post run 01¹"]):
        y=.638-j*.074
        fig.text(columns[0],y,label,fontsize=10.5)
        for k,v in enumerate([*mat[0][j],*mat[1][j]]):
            fig.text(columns[k+1],y,number(v,3),ha="center",fontsize=11)
    note(fig,.408,"¹ All other recordings kept the original ICLabel reconstruction. Across the eight participants,")
    note(fig,.375,"these were the only two single-run substitutions that restored matching directions.")
    fig.text(.055,.299,"Individual component restorations",fontsize=13,color=teal)
    for j,subject in enumerate(["sub-09","sub-12"]):
        subset=restored[restored.subject==subject]
        top_row=subset.loc[subset.change_in_half_dB.abs().idxmax()]
        scope="post run 01" if j==0 else "all eight recordings"
        note(fig,.248-j*.047,f"P{subject[-2:]}  ·  {len(subset)} restorations ({scope})  ·  {int(subset.same_direction.sum())} recovered agreement  ·  Largest shift {number(top_row.change_in_half_dB,3)} dB")
    note(fig,.131,"Each component was returned to its own recording with all other exclusions fixed.")
    note(fig,.089,"Component effects depend on the reconstruction and do not add linearly in log power.")
    footer(fig);pdf.savefig(fig);plt.close(fig)
