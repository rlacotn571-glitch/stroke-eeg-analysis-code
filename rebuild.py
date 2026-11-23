"""Rebuild the identity-audited release from the archived derived inputs."""
import os
import runpy
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
import matplotlib
matplotlib.use("Agg")
paths=["longitudinal_rules/analyze.py","sensitivity/summarize.py","record_review/record_review.py",
       "run_splits/analyze.py","split_attribution/analyze.py","auxiliary_check/summarize.py","auxiliary_check/report.py",
       "split_attribution/report.py","../../main_results.py","full_paper_figures/plot.py","sensor_eeg/plot.py",
       "run_splits/plot.py","../../matched_figure.py"]
for path in paths:
    print("Rebuilding",path,flush=True)
    runpy.run_path(str((root/"outputs/BSPC"/path).resolve()),run_name="__main__")
