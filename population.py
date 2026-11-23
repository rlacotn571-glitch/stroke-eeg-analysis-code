from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
inventory=pd.read_csv(root/"identity/recording_fingerprints.csv").sort_values("key")
assert inventory.key.is_unique and len(inventory)==293
pairs=inventory[inventory.duplicated("sha256", keep=False)]
ambiguous=set(pairs.subject)
assert ambiguous=={"sub-23","sub-24"} and len(pairs)==8
assert set(pairs.session)=={"ses-pre"} and pairs.groupby("sha256").size().eq(2).all()
# The alphabetical representative indexes the signal; it does not identify its owner.
unique_keys=set(inventory.drop_duplicates("sha256").key)
assert len(unique_keys)==289
