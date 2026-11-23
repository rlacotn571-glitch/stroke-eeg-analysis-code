from pathlib import Path
import numpy as np,pandas as pd,mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
from scipy.signal import welch
from threadpoolctl import threadpool_limits
o=Path('outputs/BSPC/record_review');mne.set_log_level('ERROR')
plt.rcParams.update({'font.family':'DejaVu Sans','font.weight':'bold','axes.labelweight':'bold',
                     'axes.titleweight':'bold','figure.titleweight':'bold','pdf.fonttype':42,
                     'text.color':'#172B35','axes.labelcolor':'#172B35',
                     'xtick.color':'#172B35','ytick.color':'#172B35'})
r=pd.read_csv(o/'restoration.csv')
cases=pd.read_csv(o/'matches.csv')
records=[k for row in cases.itertuples() for k in [row.case,row.comparison]]
with PdfPages(o/'Component_review.pdf') as pdf,threadpool_limits(limits=1):
 for k in records:
    c=r[r.key==k];ic=int(c.loc[c.restore_change_db.abs().idxmax(),'component_zero_based'])
    p=Path('outputs/stroke_analysis_full/runs')/k
    raw=mne.io.read_raw_fif(p/(k+'_before_raw.fif'),preload=True)
    fit=mne.preprocessing.read_ica(p/'iclabel'/(k+'-ica.fif'))
    src=fit.get_sources(raw).get_data(picks=[ic])[0]
    hz=raw.info['sfreq']
    f,pow=welch(src,fs=hz,nperseg=int(2*hz))
    ev=pd.read_csv(p/'selected_events.tsv',sep='\t')
    onset=ev.loc[ev.value==3,'onset'].iloc[0]
    win=(raw.times>=onset-2)&(raw.times<onset+5)
    fig=plt.figure(figsize=(8.5,5),layout='constrained')
    ax1,ax2=fig.subplots(1,2)
    # Use the upper row for the spatial projection and spectrum; new layout for traces.
    ax1.remove();ax2.remove()
    gs=fig.add_gridspec(2,2);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);t=fig.add_subplot(gs[1,:])
    eeg_info=mne.pick_info(raw.info,[raw.ch_names.index(n) for n in fit.ch_names])
    mne.viz.plot_topomap(fit.get_components()[:,ic],eeg_info,axes=a,show=False,contours=4)
    a.set_title('Scalp projection (arbitrary sign)')
    b.plot(f,10*np.log10(np.maximum(pow,1e-30)),color='#176879');b.set(xlim=(1,100),xlabel='Frequency (Hz)',ylabel='Source PSD (dB)\nArbitrary units',title='Whole recording')
    t.plot(raw.times[win]-onset,(src[win]-src.mean())/src.std(),lw=.7,color='#414d55')
    t.axvline(0,ls='--',color='#AC4826');t.set(xlabel='Seconds from first imagery-task onset (marker 3)',ylabel='Source amplitude (SD)',title='First imagery trial (fixed display window)')
    fig.suptitle(f'{k}   |   IC {ic} (zero-based)',fontsize=14)
    for label in fig.findobj(Text):
        label.set_fontweight('bold')
    pdf.savefig(fig,dpi=420)
    plt.close(fig)
