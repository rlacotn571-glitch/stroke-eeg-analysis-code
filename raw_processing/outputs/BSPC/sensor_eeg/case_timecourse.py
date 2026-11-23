"""Central beta time course and paired, pointwise trial-resampling intervals."""
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")
out = Path(__file__).resolve().parent
key = "sub-22_ses-pre_run-04"
folder = out.parents[1] / "stroke_analysis_full" / "runs" / key
trials = pd.read_csv(folder / "trial_metrics.csv")
trials = trials.query("stage == 'before' and band == 'beta'").sort_values("event_sample")
imagery = trials.condition.eq("MI").to_numpy()
centers = np.arange(-875, 1126, 10)
states = []
for path in [folder / (key + "_before_raw.fif"), folder / "iclabel" / (key + "_iclabel_raw.fif")]:
    raw = mne.io.read_raw_fif(path, preload=True).pick(["FCz", "Cz", "C1", "C2"])
    events = trials.event_sample.to_numpy() - raw.first_samp
    assert events.min()+centers.min()-50 >= 0 and events.max()+centers.max()+50 <= raw.n_times
    signal = raw.filter(13., 30., method="iir", iir_params=dict(order=4, ftype="butter")).get_data()
    power = (signal ** 2).mean(0)
    window = events[:,None,None] + centers[None,:,None] + np.arange(-50,50)
    baseline = np.stack([power[e-875:e-375].mean() for e in events])
    states.append(10 * np.log10(power[window].mean(2) / baseline[:,None]))
response = np.stack(states)
def contrast(x):
    return x[:, imagery].mean(1) - x[:, ~imagery].mean(1)
shift = response[1] - response[0]
rng = np.random.default_rng(4601)
draws = [rng.integers(0, mask.sum(), size=(10000,mask.sum())) for mask in [imagery, ~imagery]]
bootstrap = shift[imagery][draws[0]].mean(1) - shift[~imagery][draws[1]].mean(1)
np.savez_compressed(out / "case_timecourse.npz", time=centers / 250., contrast=contrast(response),
                    bounds=np.quantile(bootstrap,[.025,.975],axis=0))
