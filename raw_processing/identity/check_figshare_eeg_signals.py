"""Read the eight selected original EEGLAB files, then hash numerical signal arrays."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from struct import pack
from zlib import crc32
import subprocess
import shutil
import time
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat, whosmat

url="https://ndownloader.figshare.com/files/49521294"
review=Path("outputs/BSPC/interim_review")
cache=Path("work/figshare_pair_check")
cache.mkdir(exist_ok=True)
rows=pd.read_csv(review/"figshare_members.csv")
def block(data):
    data=bytes([len(data)])+data
    return pack("<I",crc32(data))+data

def fetch(r):
    name=r.name
    dest=cache/Path(name).name
    if dest.is_file() and dest.stat().st_size==r.uncompressed_bytes and f"{crc32(dest.read_bytes()):08x}"==r.source_crc32:
        return dest
    start=r.header_offset
    end=start+r.header_length+r.compressed_bytes-1
    rar=dest.with_suffix(".rar")
    for attempt in range(3):
        try:
            with requests.get(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"},stream=True,timeout=(20,70)) as response:
                response.raise_for_status()
                assert response.status_code==206 and response.headers["Content-Range"]==f"bytes {start}-{end}/18618424407"
                hdr=response.raw.read(r.header_length)
                assert hdr==bytes.fromhex(r.header_hex) and r.compression&0x40==0
                with rar.open("wb") as f:
                    f.write(b"Rar!\x1a\x07\x01\0"+block(bytes([1,0,0]))+hdr)
                    pos=f.tell()
                    shutil.copyfileobj(response.raw,f,1024*1024)
                    assert f.tell()-pos==r.compressed_bytes
                    f.write(block(bytes([5,4,0])))
            break
        except (requests.RequestException, OSError):
            if attempt==2: raise
            time.sleep(1+attempt)
    with dest.open("wb") as f:
        subprocess.run(["tar.exe","-xOf",str(rar),name],stdout=f,check=True)
    assert dest.stat().st_size==r.uncompressed_bytes and f"{crc32(dest.read_bytes()):08x}"==r.source_crc32
    print("Verified original archive member",dest.name,flush=True)
    return dest

def inspect(path):
    fields=whosmat(path)
    key="EEG" if any(x[0]=="EEG" for x in fields) else "data"
    obj=loadmat(path,simplify_cells=True,variable_names=[key])
    signal=obj[key]["data"] if key=="EEG" else obj[key]
    assert isinstance(signal,np.ndarray) and signal.ndim==2 and signal.size>1000000
    return [path.name,*signal.shape,str(signal.dtype),sha256(np.ascontiguousarray(signal).tobytes()).hexdigest()]

if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        files=list(pool.map(fetch,rows.itertuples(index=False)))
    df=pd.DataFrame([inspect(p) for p in files],columns=["figshare_member","channels","samples","dtype","data_sha256_c_order"])
    df.to_csv(review/"original_signal_hashes.csv",index=False)
    print(df.to_string(index=False))
