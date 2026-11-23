"""Rebuild sensor estimates from the two saved continuous reconstructions per run."""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[1] / "stroke_analysis_full"
CASE = "sub-22_ses-pre_run-04"
ROI = ["FCz", "Cz", "C1", "C2"]
STAGES = ["before", "iclabel_after"]
VISITS = ["ses-pre", "ses-post", "ses-follow"]
mne.set_log_level("ERROR")


def case_spectrum(raw, events, imagery):
    # The longest five-cycle wavelet extends less than 0.51 s either side at 8 Hz.
    # Transform padded trials, then keep only full-support time points.
    offsets = np.arange(-1125, 1376)
    assert events.min() + offsets[0] >= 0 and events.max() + offsets[-1] < raw.n_times
    signal = raw.get_data(picks=ROI)[:, events[:, None] + offsets].transpose(1, 0, 2)
    freqs = np.arange(8., 41.)
    power = mne.time_frequency.tfr_array_morlet(
        signal, 250., freqs, n_cycles=5., zero_mean=True, output="power",
        use_fft=True, decim=5, n_jobs=1, verbose=False).mean(axis=1)
    t = offsets[::5] / 250.
    base = (t >= -3.5) & (t < -1.5)
    display = (t >= -3.5) & (t <= 4.5)
    log_ratio = 10 * np.log10(power / power[:, :, base].mean(axis=2, keepdims=True))
    return (log_ratio[imagery].mean(0) - log_ratio[~imagery].mean(0))[:, display], t[display], freqs


def calculate():
    trials = pd.read_csv(DATA / "trial_metrics_all_stages.csv").query("band == 'beta'")
    saved = pd.read_csv(OUT.parent / "longitudinal_rules" / "visit_contrasts.csv")
    records = []
    visit_checks = []
    spectra = []
    channels = None
    max_error = 0.
    for count, (key, rows) in enumerate(trials.groupby("key", sort=True), 1):
        folder = DATA / "runs" / key
        first = rows[rows.stage == "before"].sort_values("event_sample")
        imagery = first.condition.eq("MI").to_numpy()
        paths = [folder / (key + "_before_raw.fif"), folder / "iclabel" / (key + "_iclabel_raw.fif")]
        for stage, path in zip(STAGES, paths):
            target = rows[rows.stage == stage].sort_values("event_sample")
            assert first[["condition", "event_sample"]].equals(target[["condition", "event_sample"]].set_axis(first.index))
            raw = mne.io.read_raw_fif(path, preload=True).pick("eeg")
            assert raw.info["sfreq"] == 250. and not raw.info["bads"]
            if channels is None:
                channels = raw.ch_names.copy()
            assert raw.ch_names == channels and len(channels) == 59
            events = target.event_sample.to_numpy() - raw.first_samp
            if key == CASE:
                spectrum, times, freqs = case_spectrum(raw, events, imagery)
                spectra.append(spectrum)
            data = raw.filter(13., 30., method="iir", iir_params=dict(order=4, ftype="butter")).get_data()
            ix = [channels.index(ch) for ch in ROI]
            task, base = [np.stack([(data[:, e + a:e + b] ** 2).mean(1) for e in events]) for a, b in [(125, 1125), (-875, -375)]]
            sensor = 10 * np.log10(task / base)
            pooled = 10 * np.log10(task[:, ix].mean(1) / base[:, ix].mean(1))
            max_error = max(max_error, np.max(abs(pooled - target.task_vs_baseline_db.to_numpy())))
            assert max_error < 1.e-4, (key, max_error)
            for cond, mask in [("MI", imagery), ("rest", ~imagery)]:
                records.append([key, target.subject.iloc[0], target.session.iloc[0], stage,
                                cond, mask.sum(), *sensor[mask].mean(0)])
                visit_checks.append([target.subject.iloc[0], target.session.iloc[0], stage, cond,
                                     pooled[mask].sum(), mask.sum()])
        if count % 40 == 0:
            print(count, "of", trials.key.nunique(), flush=True)
    frame = pd.DataFrame(records, columns=["key", "subject", "session", "stage", "condition", "n", *channels])
    frame.to_csv(OUT / "sensor_condition_means.csv", index=False, float_format="%.9g")
    checks = pd.DataFrame(visit_checks, columns=["subject", "session", "stage", "condition", "total", "n"])
    checks = checks.groupby(["subject", "session", "stage", "condition"])[["total", "n"]].sum()
    contrast = (checks.total / checks.n).unstack("condition")
    contrast = (contrast.MI - contrast.rest).unstack("stage").sort_index()
    published = saved.set_index(["subject", "session"]).loc[contrast.index, ["Filter only", "Original"]]
    assert np.max(abs(contrast[STAGES].to_numpy() - published.to_numpy())) < 1.e-4
    assert len(spectra) == 2 and frame.key.nunique() == 293 and frame.subject.nunique() == 27
    np.savez_compressed(OUT / "case_spectra.npz", contrast=np.stack(spectra), time=times, freq=freqs)
    print("Matched ROI across all trials, maximum absolute discrepancy (dB):", max_error)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        calculate()
