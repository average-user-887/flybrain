"""Fetch the full public MaleCNS v1.0 source tables and verify their upstream SHA-256 hashes.

Downloads annotations.feather, neurotransmitters.feather and edges.feather (about
1.1 GB in total, CC BY 4.0) from the FlyEM public bucket into
connectome_data/malecns_v1/.  Files already present are re-verified, not re-fetched.
These files are DOWNLOADED, so their bytes are fixed by the server and are checked
by byte hash; the files built from them locally are checked by content
(brainlab/graph_identity.py).
"""
import argparse
import json
import urllib.request
from .connectome import ROOT, REGISTRY, file_digest


ATTRIBUTION = """\
MaleCNS v1.0 connectome -- FlyEM / HHMI Janelia Research Campus, University of Cambridge,
MRC Laboratory of Molecular Biology, and Google Research.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
Source: https://male-cns.janelia.org/download/
Paper:  https://doi.org/10.1016/j.cell.2026.08.015
If you publish or share results built on these data, credit the MaleCNS authors and cite
the paper (see NOTICE)."""


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog='neurofly download-data',
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Next steps: python -m brainlab.connectome  (normalise), then '
               'python -m brainlab.prepare  (build graph.npz).\n\n' + ATTRIBUTION)


def main(argv=None):
    # Parse before any side effect: --help / -h / a bad flag must never start a download.
    build_parser().parse_args(argv)
    print(ATTRIBUTION + '\n', flush=True)
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
    print('Next: python -m brainlab.connectome  then  python -m brainlab.prepare  '
          '(builds the graph `neurofly status` verifies)', flush=True)

if __name__ == '__main__':
    main()
