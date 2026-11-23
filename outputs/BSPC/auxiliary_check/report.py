from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans'})
o=Path(__file__).resolve().parent
r=pd.read_csv(o/'summary.csv').query("subset=='complete_screened'")
fig=plt.figure(figsize=(8.3,10.5));y=.94
def para(s,size=10,weight='normal',width=96):
 global y
 for line in textwrap.wrap(s,width):
    fig.text(.08,y,line,fontsize=size,weight=weight,va='top');y-=.023
 y-=.015
para('Supplement S3. EEG coupling to auxiliary channels',15,'bold',70)
para('The trial grouping used only the two bipolar EOG signals. For each trial we took the largest RMS across vertical and horizontal EOG (1–15 Hz) and the two endpoint windows. Within each run and condition, lower comprises trials at or below the median. Higher comprises the remaining trials. This is a relative ranking; a low rank does not establish artifact-free EEG.')
para('Comparison on the same eligible runs',12,'bold')
ax=fig.add_axes([.08,y-.235,.84,.225]);ax.axis('off')
table=[]
for row in r.itertuples():
 name={'beta':'Beta shift (dB)','eye':'EEG–EOG |r|','heart':'EEG–ECG |r|'}[row.metric]
 table.append([name,row.stratum,str(row.runs),str(row.patients),f'{row.mean:+.3f}',f'{row.low:+.3f} to {row.high:+.3f}'])
tab=ax.table(cellText=table,colLabels=['Measure','EOG stratum','Runs','People','Mean Δ','95% interval'],loc='center',cellLoc='center',colWidths=[.21,.17,.09,.11,.13,.29]);tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.6)
for (i,j),cell in tab.get_celld().items():
    cell.set_edgecolor('#9daeb1')
    cell.set_facecolor('#dcebed' if i==0 else 'white')
    cell.get_text().set_weight('bold')
    cell.get_text().set_color('#14252e')
y-=.25
para('Measures and analysis sets',12,'bold')
para('All differences are ICLabel minus filtering alone. Beta is the imagery-minus-rest difference in task-to-baseline log power. For each window and each of four central electrodes, we computed the maximum absolute correlation with the two EOG leads or the absolute correlation with ECG (5–25 Hz). We averaged electrodes and the two windows equally, then averaged trials within run, runs within participant and participants equally. Intervals are descriptive percentile intervals from 10,000 participant resamples.')
para('The original acquisition QC flags a lead when its 1–15 Hz standard deviation exceeds 1,000 µV. The complete screened analysis also requires finite ocular correlations throughout a recording and both tasks in both strata. ECG results additionally require a usable ECG channel. Undefined correlations were omitted rather than set to zero. The latter completeness analysis was added after observing undefined correlations. Of 5,370 identity-resolved trials, 233 had undefined ocular and 2,625 had undefined cardiac correlations. Unscreened and flag-only sensitivity estimates accompany this supplement.')
para('Interpretation',12,'bold')
para('Correlation measures statistical coupling and can change when cleaning alters unrelated EEG variance. A positive change in EEG–ECG correlation does not quantify an increase in absolute cardiac artifact. Neither auxiliary signal identifies muscle activity. No equivalence margin was specified for beta.')
fig.savefig(o/'Supplement_S3.pdf',bbox_inches='tight',pad_inches=.45);plt.close(fig)
