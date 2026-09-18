"""Fetch the full public data and verify upstream SHA-256 hashes."""
import json
import urllib.request
from .connectome import ROOT, REGISTRY, file_digest


def main():
    config = json.loads(REGISTRY.read_text())['datasets']['malecns_v1']
    lock = json.loads((ROOT/'data-provenance/malecns_v1/source.lock.json').read_text())
    out = ROOT/'connectome_data/malecns_v1'
    out.mkdir(parents=True, exist_ok=True)
    for name, url in config['files'].items():
        target = out/name
        if not target.exists():
            partial = target.with_suffix('.partial')
            print(f'Downloading {name}', flush=True)
            urllib.request.urlretrieve(url, partial)
            if file_digest(partial) != lock[name]['sha256']:
                raise ValueError(f'Checksum mismatch: {partial}')
            partial.replace(target)
        if file_digest(target) != lock[name]['sha256']:
            raise ValueError(f'Checksum mismatch: {target}')
        print(f'Verified {name}', flush=True)
    (out/'source.lock.json').write_text(json.dumps(lock, indent=2)+'\n')

if __name__ == '__main__':
    main()
