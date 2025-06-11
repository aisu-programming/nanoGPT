import os
import pickle
import requests
import numpy as np
import zipfile

# URLs for enwik8 dataset. If one fails, try the next.
URLS = [
    'https://data.deepai.org/enwik8.zip',
    'http://mattmahoney.net/dc/enwik8.zip',
    'http://data.statmt.org/acl2007-hypercurves/enwik8.gz',
]

data_dir = os.path.dirname(__file__)
zip_path = os.path.join(data_dir, 'enwik8.zip')
raw_path = os.path.join(data_dir, 'enwik8')

def download(url):
    print(f'Downloading {url}...')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

if __name__ == '__main__':
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(raw_path):
        if not os.path.exists(zip_path):
            for url in URLS:
                try:
                    download(url)
                    break
                except Exception as e:
                    print(f'Failed to download from {url}: {e}')
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
            else:
                raise RuntimeError('Failed to download enwik8 from all mirrors.')
        print('Extracting...')
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(data_dir)
        except zipfile.BadZipFile:
            import gzip
            with gzip.open(zip_path, 'rb') as f_in, open(raw_path, 'wb') as f_out:
                f_out.write(f_in.read())
    with open(raw_path, 'rb') as f:
        data = f.read()
    n = len(data)
    train_bytes = data[:int(n*0.9)]
    val_bytes = data[int(n*0.9):]
    train_ids = np.frombuffer(train_bytes, dtype=np.uint8).astype(np.uint16)
    val_ids = np.frombuffer(val_bytes, dtype=np.uint8).astype(np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    meta = {'vocab_size': 256}
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f'train.bin has {train_ids.size:,} tokens')
    print(f'val.bin has {val_ids.size:,} tokens')
