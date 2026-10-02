"""Download MovieLens 100K; dataset is not redistributed in this repository."""
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

def download():
    root=Path("data/ml-100k")
    if (root/"u.data").exists() and (root/"u.item").exists():return
    url="https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    with urlopen(url,timeout=60) as response:content=response.read(20_000_000)
    expected="0e33842e24a9c977be4e0107933c0723"
    # GroupLens's stable archive; fail rather than train on a changed download.
    if hashlib.md5(content).hexdigest()!=expected:raise ValueError("MovieLens archive checksum changed; verify upstream before updating expected checksum")
    with ZipFile(io.BytesIO(content)) as archive:
        for name in ["u.data","u.item","README"]:
            root.mkdir(parents=True,exist_ok=True);(root/name).write_bytes(archive.read("ml-100k/"+name))
    print("Downloaded MovieLens 100K. Read data/ml-100k/README for dataset terms.")
if __name__=="__main__":download()
