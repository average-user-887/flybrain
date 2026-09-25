"""Read-only census of the mushroom-body (MB) learning circuit in MaleCNS v1.0.

Produces the numbers quoted in ``docs/design/mb_learning.md``: Kenyon cell (KC),
MB output neuron (MBON), dopaminergic neuron (DAN), APL and DPM populations,
PN -> KC input, KC -> MBON connectivity per compartment, DAN -> KC / MBON
anatomy, APL feedback, MBON output toward descending neurons and the inputs to
the PPL1 punishment DANs.

It reads the three released files that ``brainlab/datasets.json`` pins
(annotations, neurotransmitters, edges), applies the SAME node policy as the
simulation graph (``brainlab.connectome.normalize_nodes``: an assigned
superclass, explicit Glia excluded) and the same edge policy (every released
edge between retained neurons, no threshold), so every ``node_index`` below is
the index the engine uses.  Nothing is written except the ``--out`` JSON.

    PYTHONPATH=. python scripts/mb_circuit_census.py \
        --source connectome_data/malecns_v1 --out docs/design/mb_circuit_census.json

The directory must hold ``annotations.feather``, ``neurotransmitters.feather``
and ``edges.feather`` whose sha256 match ``data-provenance/malecns_v1/source.lock.json``.

The compartment split (``--synapses``, ``--points``) needs two more released
tables from the same bucket (flyem-male-cns/v1.0/connectome-data/flat-connectome/),
not pinned in the lock file; the committed JSON was made from
syn-partners-male-cns-v1.0-minconf-0.5.feather (sha256 959d8ef4...87bc07) and
syn-points-male-cns-v1.0-minconf-0.5.feather (sha256 c16b1b63...58f284).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYNAPTIC_SCALE = 0.275   # prepared-graph weight per synaptic contact (brainlab/prepare.py)

# Annotation queries.  Every population in the design doc is one of these
# regular expressions on the released ``type`` column, or the ``class`` column.
QUERIES = {
    'KC': ('type', r'^KC'),
    'MBON': ('type', r'^MBON'),
    'PAM': ('type', r'^PAM\d'),
    'PPL1': ('type', r'^PPL1\d'),
    'PPL2': ('type', r'^PPL2\d'),
    'APL': ('type', r'^APL$'),
    'DPM': ('type', r'^DPM$'),
    'MB-C1': ('type', r'^MB-C1$'),
    'OA-VPM3/4': ('type', r'^OA-VPM[34]$'),
    'ALPN': ('class', r'^ALPN$'),
    'DAN_class': ('class', r'^DAN$'),
    'DN': ('superclass', r'^descending_neuron'),
}

# The released neurotransmitter prediction for these classes gets no fast weight
# under the v3 transmitter policy (brainlab/transmitter_policy.py).
MODULATORY = ('dopamine', 'octopamine', 'serotonin')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def compartment(instance: str) -> str | None:
    match = re.search(r'\(([^)]*)\)', str(instance))
    return match.group(1) if match else None


def named_compartments(name: str | None) -> set[str]:
    """Dendritic compartments named in an MBON instance, in the ROI spelling.

    ``y1pedc>a/B`` -> {g1, PED}; ``y2a'1`` -> {g2, a'1}; ``a'1a`` -> {a'1}.
    Only the part before '>' (the dendrite) counts.
    """
    if not name:
        return set()
    text = name.split('>')[0].replace('y', 'g').replace('B', 'b')
    text = re.sub(r'a(\d)p(\d)p', r'a\1a\2', text)       # MBON19 'a2p3p': alpha2p and alpha3p
    comps = {f'{lobe}{num}' for lobe, num in re.findall(r"(g|a'|a|b'|b)(\d)", text)}
    if 'ped' in text:
        comps.add('PED')
    return comps


def lobe_of_kc(kc_type: str) -> str:
    if kc_type.startswith("KCa'b'"):
        return "a'b'"
    if kc_type.startswith('KCab'):
        return 'ab'
    if kc_type.startswith('KCg'):
        return 'g'
    return 'unassigned'


def load(source: Path):
    import pyarrow.feather as feather
    from brainlab.connectome import normalize_nodes
    frame = feather.read_table(source / 'annotations.feather').to_pandas()
    nt = feather.read_table(source / 'neurotransmitters.feather').to_pandas()
    _, nodes = normalize_nodes('malecns_v1', frame, nt)
    extra = frame.set_index('bodyId')[['instance', 'somaSide', 'rootSide', 'class', 'superclass']]
    ids = nodes.source_id.to_numpy(dtype=np.int64)
    nodes['instance'] = extra.instance.reindex(ids).to_numpy()
    nodes['side'] = extra.somaSide.reindex(ids).fillna(extra.rootSide.reindex(ids)).to_numpy()
    nodes['klass'] = extra['class'].reindex(ids).to_numpy()
    return nodes


def select(nodes, name):
    column, pattern = QUERIES[name]
    column = {'class': 'klass', 'type': 'cell_type'}.get(column, column)
    return nodes[column].fillna('').astype(str).str.contains(pattern, regex=True).to_numpy()


def read_edges(source: Path, ids: np.ndarray, keep_pre: np.ndarray, keep_post: np.ndarray):
    """Retained edges (as node indices) with pre in keep_pre OR post in keep_post."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    reader = ipc.open_file(pa.memory_map(str(source / 'edges.feather'), 'r'))
    out_i, out_j, out_c = [], [], []
    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)
        pre = batch.column(batch.schema.get_field_index('body_pre')).to_numpy()
        post = batch.column(batch.schema.get_field_index('body_post')).to_numpy()
        count = batch.column(batch.schema.get_field_index('weight')).to_numpy()
        i = np.searchsorted(ids, pre)
        j = np.searchsorted(ids, post)
        ok = (i < len(ids)) & (j < len(ids))
        ok &= ids[np.minimum(i, len(ids) - 1)] == pre
        ok &= ids[np.minimum(j, len(ids) - 1)] == post
        i, j, count = i[ok], j[ok], count[ok]
        sel = keep_pre[i] | keep_post[j]
        out_i.append(i[sel]); out_j.append(j[sel]); out_c.append(count[sel])
    return (np.concatenate(out_i).astype(np.int64), np.concatenate(out_j).astype(np.int64),
            np.concatenate(out_c).astype(np.int64))


def block(pre_mask, post_mask, i, j, c):
    sel = pre_mask[i] & post_mask[j]
    return dict(edges=int(sel.sum()), contacts=int(c[sel].sum()),
                pre_neurons=int(len(np.unique(i[sel]))), post_neurons=int(len(np.unique(j[sel]))))


def synapse_rois(path: Path, ids, ctype, kc, mbon, dan):
    """Split KC->MBON and DAN->KC synapses by the released ROI of the postsynaptic site.

    The flat edge table has one row per neuron pair; the engine has one weight per
    pair.  An MBON whose dendrite spans several compartments (e.g. MBON20, gamma1
    gamma2) receives each KC's synapses in more than one compartment, so the
    compartment of a synapse has to come from here.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    reader = ipc.open_file(pa.memory_map(str(path), 'r'))
    names = reader.schema.names
    n = len(ids)
    kc_mbon = Counter(); edge_roi = Counter(); dan_kc = Counter()
    rows = synapses = 0
    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)
        pre = batch.column(names.index('body_pre')).to_numpy()
        post = batch.column(names.index('body_post')).to_numpy()
        roi_col = batch.column(names.index('primary_post'))
        i = np.searchsorted(ids, pre); j = np.searchsorted(ids, post)
        ok = (i < n) & (j < n)
        ok &= ids[np.minimum(i, n - 1)] == pre
        ok &= ids[np.minimum(j, n - 1)] == post
        rows += len(pre); synapses += int(ok.sum())
        want = ok.copy()
        want[ok] = (kc[i[ok]] & mbon[j[ok]]) | (dan[i[ok]] & kc[j[ok]])
        if not want.any():
            continue
        idx = np.flatnonzero(want)
        rois = roi_col.take(pa.array(idx)).to_pylist()
        for k, roi in zip(idx, rois):
            a, b = int(i[k]), int(j[k]); roi = roi or 'none'
            if kc[a]:
                kc_mbon[(ctype[b], roi)] += 1
                edge_roi[(a, b, roi)] += 1
            else:
                dan_kc[(ctype[a], roi)] += 1
    per_mbon = {}
    for (t, roi), v in sorted(kc_mbon.items()):
        per_mbon.setdefault(t, {})[roi] = v
    per_dan = {}
    for (t, roi), v in sorted(dan_kc.items()):
        per_dan.setdefault(t, {})[roi] = v
    # edges (neuron pairs) with >= 1 synapse in each ROI, per MBON type
    edges_per = {}
    for (a, b, roi), v in edge_roi.items():
        cell = edges_per.setdefault(ctype[b], {}).setdefault(roi, [0, 0])
        cell[0] += 1; cell[1] += v
    return dict(source=path.name, synapse_rows=rows, synapses_between_retained=synapses,
                kc_to_mbon_synapses_by_type_and_roi=per_mbon,
                kc_to_mbon_edges_and_synapses_by_type_and_roi={t: {r: dict(edges=e, synapses=s) for r, (e, s) in d.items()}
                                                              for t, d in sorted(edges_per.items())},
                dan_to_kc_synapses_by_type_and_roi=per_dan)


def compartment_split(partners: Path, points: Path, ids, ctype, kc, mbon, dan, instances):
    """Assign KC->MBON synapses (by the MBON's postsynaptic point) and DAN->KC synapses
    (by the DAN's presynaptic point) to the released ``subprimary`` ROI, which resolves
    the MB lobes into compartments (g1..g5, a1..a3, a'1..a'3, b1, b2, b'1, b'2).

    Returns, per MBON type and compartment, the number of KC->MBON neuron pairs
    (engine edges) with synapses there and the synapse count, plus the same for each
    DAN type's output onto KCs.  Also returns, for every KC->MBON engine edge, how its
    synapses split across compartments, summarised as the fraction of edges whose
    synapses all lie in one compartment.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc
    n = len(ids)
    mbon_ids = pa.array(ids[mbon]); dan_ids = pa.array(ids[dan]); kc_ids = pa.array(ids[kc])
    # 1. the synapses we need, from the partner table
    reader = ipc.open_file(pa.memory_map(str(partners), 'r'))
    want = []
    for number in range(reader.num_record_batches):
        b = reader.get_batch(number)
        km = pc.and_(pc.is_in(b['body_pre'], kc_ids), pc.is_in(b['body_post'], mbon_ids))
        dk = pc.and_(pc.is_in(b['body_pre'], dan_ids), pc.is_in(b['body_post'], kc_ids))
        for mask, side, kind in ((km, 'post', 'kc_mbon'), (dk, 'pre', 'dan_kc')):
            t = b.filter(mask)
            if len(t):
                want.append(pd.DataFrame(dict(kind=kind, pre=t['body_pre'].to_numpy(), post=t['body_post'].to_numpy(),
                                              x=t[f'x_{side}'].to_numpy(), y=t[f'y_{side}'].to_numpy(),
                                              z=t[f'z_{side}'].to_numpy())))
    syn = pd.concat(want, ignore_index=True)
    syn['body'] = np.where(syn.kind == 'kc_mbon', syn.post, syn.pre)
    # 2. the ROI of those points
    bodies = pa.array(np.unique(syn.body.to_numpy()))
    reader = ipc.open_file(pa.memory_map(str(points), 'r'))
    found = []
    for number in range(reader.num_record_batches):
        b = reader.get_batch(number)
        t = b.filter(pc.is_in(b['body'], bodies))
        if len(t):
            found.append(pd.DataFrame(dict(body=t['body'].to_numpy(), x=t['x'].to_numpy(), y=t['y'].to_numpy(),
                                           z=t['z'].to_numpy(), kind=t['kind'].to_pylist(),
                                           roi=t['subprimary'].to_pylist(),
                                           primary=t['primary'].to_pylist())))
    pts = pd.concat(found, ignore_index=True)
    pts = pts.drop_duplicates(['body', 'x', 'y', 'z', 'kind'])
    syn['point_kind'] = np.where(syn.kind == 'kc_mbon', 'PostSyn', 'PreSyn')
    merged = syn.merge(pts, how='left', left_on=['body', 'x', 'y', 'z', 'point_kind'],
                       right_on=['body', 'x', 'y', 'z', 'kind'], suffixes=('', '_pt'))
    merged['roi'] = merged.roi.fillna('unmatched')
    # The pedunculus has no subprimary ROI: fall back to the primary ROI (PED, CA, ...)
    # wherever the subprimary is unspecified, so gamma1pedc = g1 + PED.
    unspecified = merged.roi.isin(['<unspecified>']) | merged.roi.str.endswith('-unspecified(L)') \
        | merged.roi.str.endswith('-unspecified(R)')
    merged.loc[unspecified, 'roi'] = merged.loc[unspecified, 'primary'].fillna('<unspecified>')
    # strip the side, keep the compartment name
    merged['comp'] = merged.roi.str.replace(r'\((L|R)\)$', '', regex=True)
    index = pd.Series(np.arange(n), index=ids)
    merged['pre_i'] = index.reindex(merged.pre).to_numpy()
    merged['post_i'] = index.reindex(merged.post).to_numpy()
    out = dict(points_source=points.name, synapses=int(len(merged)),
               unmatched=int((merged.roi == 'unmatched').sum()))
    km = merged[merged.kind == 'kc_mbon'].copy()
    km['mtype'] = ctype[km.post_i.to_numpy()]
    per = {}
    for (mt, comp), g in km.groupby(['mtype', 'comp']):
        per.setdefault(mt, {})[comp] = dict(edges=int(g[['pre', 'post']].drop_duplicates().shape[0]),
                                            synapses=int(len(g)))
    out['kc_to_mbon_by_type_and_compartment'] = per
    split = km.groupby(['pre', 'post']).comp.nunique()
    out['kc_to_mbon_edges'] = dict(total=int(len(split)), single_compartment=int((split == 1).sum()))
    # The declared plastic sets of docs/design/mb_learning.md section 2.5: synapses
    # in a PPL1 compartment ROI onto an MBON whose released instance name places its
    # dendrite in that compartment (so incidental synapses of other MBONs that
    # cross a ROI boundary stay fixed).
    named = {}
    for idx in np.flatnonzero(mbon):
        named.setdefault(ctype[idx], set()).update(named_compartments(compartment(instances[idx])))
    out['mbon_named_compartments'] = {t: sorted(v) for t, v in sorted(named.items())}
    km['named'] = [comp in named.get(mt, ()) for comp, mt in zip(km.comp, km.mtype)]
    sets = {'primary': ['g1', 'PED', 'g2', "a'1"],
            'secondary': ['g1', 'PED', 'g2', "a'1", "a'2", 'a2', 'a3', "a'3"]}
    out['plastic_sets'] = {}
    for name, comps in sets.items():
        g = km[km.comp.isin(comps) & km.named]
        out['plastic_sets'][name] = dict(
            compartments=comps, synapses=int(len(g)),
            edge_compartment_pairs=int(g[['pre', 'post', 'comp']].drop_duplicates().shape[0]),
            engine_edges=int(g[['pre', 'post']].drop_duplicates().shape[0]),
            mbon_types=sorted(set(g.mtype)),
            synapses_by_mbon_type={t: int(v) for t, v in g.mtype.value_counts().sort_index().items()},
            fraction_of_kc_mbon_synapses=round(len(g) / len(km), 4))
    dk = merged[merged.kind == 'dan_kc'].copy()
    dk['dtype'] = ctype[dk.pre_i.to_numpy()]
    per = {}
    for (dt, comp), g in dk.groupby(['dtype', 'comp']):
        per.setdefault(dt, {})[comp] = int(len(g))
    out['dan_to_kc_synapses_by_type_and_compartment'] = per
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=ROOT / 'connectome_data/malecns_v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--synapses', type=Path, default=None,
                        help='syn-partners-male-cns-v1.0-minconf-0.5.feather (6.8 GB); adds the per-compartment '
                             'split of KC->MBON and DAN->KC synapses from the released primary_post ROI')
    parser.add_argument('--points', type=Path, default=None,
                        help='syn-points-male-cns-v1.0-minconf-0.5.feather (13 GB); with --synapses, assigns each '
                             'KC->MBON and DAN->KC synapse to its MB compartment (released subprimary ROI)')
    parser.add_argument('--skip-hash', action='store_true', help='do not verify the source sha256 (slow on 1 GB)')
    args = parser.parse_args()

    lock = json.loads((ROOT / 'data-provenance/malecns_v1/source.lock.json').read_text())
    hashes = {}
    for name in ('annotations.feather', 'neurotransmitters.feather', 'edges.feather'):
        if not args.skip_hash:
            got = sha256(args.source / name)
            if got != lock[name]['sha256']:
                raise SystemExit(f'{name}: sha256 {got} does not match source.lock.json')
        hashes[name] = lock[name]['sha256']

    nodes = load(args.source)
    n = len(nodes)
    ids = nodes.source_id.to_numpy(dtype=np.int64)
    ctype = nodes.cell_type.fillna('').astype(str).to_numpy()
    nt = nodes.neurotransmitter.fillna('missing').astype(str).str.lower().to_numpy()
    masks = {name: select(nodes, name) for name in QUERIES}

    kc, mbon, pam, ppl1 = masks['KC'], masks['MBON'], masks['PAM'], masks['PPL1']
    apl, dpm, alpn, dn = masks['APL'], masks['DPM'], masks['ALPN'], masks['DN']
    dan = masks['DAN_class']
    mb_core = kc | mbon | dan | apl | dpm | masks['MB-C1'] | masks['OA-VPM3/4']
    everything = np.ones(n, dtype=bool)
    i, j, c = read_edges(args.source, ids, everything, everything)
    if len(i) != 25_582_938:
        raise SystemExit(f'{len(i)} retained edges; the pinned simulation graph has 25,582,938')

    def census(mask):
        sub = nodes[mask]
        return dict(
            neurons=int(mask.sum()),
            by_type={t: int(v) for t, v in sub.cell_type.value_counts().sort_index().items()},
            by_side={str(s): int(v) for s, v in sub.side.fillna('unknown').value_counts().sort_index().items()},
            transmitter={str(t): int(v) for t, v in sub.neurotransmitter.fillna('missing').value_counts().items()},
        )

    report = dict(
        schema='flybrain.mb-circuit-census.v1',
        dataset='MaleCNS v1.0', source_sha256=hashes,
        node_policy='brainlab.connectome.normalize_nodes (assigned superclass, Glia excluded)',
        edge_policy='every released edge between retained neurons; no threshold',
        synaptic_scale=SYNAPTIC_SCALE, retained_neurons=n,
        queries={k: dict(column=v[0], regex=v[1]) for k, v in QUERIES.items()},
        populations={name: census(masks[name]) for name in
                     ('KC', 'MBON', 'PAM', 'PPL1', 'PPL2', 'APL', 'DPM', 'MB-C1', 'OA-VPM3/4', 'ALPN', 'DAN_class')},
    )
    report['populations']['DN'] = dict(neurons=int(dn.sum()))

    # KC classes by lobe system
    lobes = Counter(lobe_of_kc(t) for t in ctype[kc])
    report['kc_lobe_system'] = dict(sorted(lobes.items()))

    # -- compartments, from the released instance names --------------------------
    comp = {}
    for idx in np.flatnonzero(mbon | pam | ppl1):
        comp.setdefault(ctype[idx], compartment(nodes.instance.iat[idx]))
    report['compartment_from_instance'] = dict(sorted(comp.items()))

    # -- PN -> KC ---------------------------------------------------------------
    sel = alpn[i] & kc[j]
    pn_kc = dict(edges=int(sel.sum()), contacts=int(c[sel].sum()))
    kc_idx = np.flatnonzero(kc)
    for thr in (1, 3, 5):
        s = sel & (c >= thr)
        partners = np.bincount(j[s], minlength=n)[kc_idx]
        pn_kc[f'distinct_PN_inputs_per_KC_min{thr}syn'] = dict(
            mean=float(partners.mean()), median=float(np.median(partners)),
            p10=float(np.percentile(partners, 10)), p90=float(np.percentile(partners, 90)),
            kcs_with_zero=int((partners == 0).sum()))
    by_lobe = {}
    for lobe in ('g', "a'b'", 'ab', 'unassigned'):
        members = kc_idx[[lobe_of_kc(t) == lobe for t in ctype[kc_idx]]]
        if not len(members):
            continue
        s = sel & (c >= 3)
        partners = np.bincount(j[s], minlength=n)[members]
        by_lobe[lobe] = dict(kcs=int(len(members)), mean_PN_inputs_min3=float(partners.mean()))
    pn_kc['by_lobe_min3'] = by_lobe
    # all KC input, by presynaptic class
    into_kc = kc[j]
    classes = nodes.klass.fillna(nodes.superclass).fillna('unknown').astype(str).to_numpy()
    totals = Counter()
    for cls, cnt in zip(classes[i[into_kc]], c[into_kc]):
        totals[cls] += int(cnt)
    total = sum(totals.values())
    pn_kc['kc_input_contacts_by_presynaptic_class_top'] = {
        k: dict(contacts=v, fraction=round(v / total, 4)) for k, v in totals.most_common(12)}
    report['pn_to_kc'] = pn_kc

    # -- KC -> MBON, per MBON type (= compartment) ------------------------------
    per_mbon = {}
    for t in sorted(set(ctype[mbon])):
        post_mask = (ctype == t) & mbon
        s = kc[i] & post_mask[j]
        lobe_contacts = Counter()
        for pre, cnt in zip(i[s], c[s]):
            lobe_contacts[lobe_of_kc(ctype[pre])] += int(cnt)
        per_mbon[t] = dict(compartment=comp.get(t), neurons=int(post_mask.sum()),
                           transmitter=Counter(nt[post_mask]).most_common(1)[0][0],
                           kc_edges=int(s.sum()), kc_contacts=int(c[s].sum()),
                           presynaptic_kcs=int(len(np.unique(i[s]))),
                           kc_contacts_by_lobe=dict(lobe_contacts))
    report['kc_to_mbon'] = dict(total=block(kc, mbon, i, j, c), per_mbon_type=per_mbon)

    # -- DAN anatomy: DAN -> KC, DAN -> MBON (fast weight is 0 under v3) --------
    per_dan = {}
    for t in sorted(set(ctype[pam | ppl1])):
        pre_mask = (ctype == t) & (pam | ppl1)
        to_kc = pre_mask[i] & kc[j]
        lobe_contacts = Counter()
        for post, cnt in zip(j[to_kc], c[to_kc]):
            lobe_contacts[lobe_of_kc(ctype[post])] += int(cnt)
        to_mbon = pre_mask[i] & mbon[j]
        mb_targets = Counter()
        for post, cnt in zip(j[to_mbon], c[to_mbon]):
            mb_targets[ctype[post]] += int(cnt)
        from_mbon = mbon[i] & pre_mask[j]
        mb_inputs = Counter()
        for pre, cnt in zip(i[from_mbon], c[from_mbon]):
            mb_inputs[ctype[pre]] += int(cnt)
        from_kc = kc[i] & pre_mask[j]
        per_dan[t] = dict(compartment=comp.get(t), neurons=int(pre_mask.sum()),
                          transmitter=Counter(nt[pre_mask]).most_common(1)[0][0],
                          to_kc=dict(edges=int(to_kc.sum()), contacts=int(c[to_kc].sum()),
                                     kcs=int(len(np.unique(j[to_kc]))), by_lobe=dict(lobe_contacts)),
                          to_mbon_top=dict(mb_targets.most_common(4)),
                          from_mbon_top=dict(mb_inputs.most_common(4)),
                          from_kc=dict(edges=int(from_kc.sum()), contacts=int(c[from_kc].sum())))
    report['dan'] = per_dan

    # -- APL, DPM ------------------------------------------------------------------
    report['apl'] = dict(kc_to_apl=block(kc, apl, i, j, c), apl_to_kc=block(apl, kc, i, j, c),
                         apl_to_mbon=block(apl, mbon, i, j, c), dpm_to_kc=block(dpm, kc, i, j, c),
                         kc_to_dpm=block(kc, dpm, i, j, c), kc_to_kc=block(kc, kc, i, j, c),
                         kc_to_dan=block(kc, dan, i, j, c), mbon_to_dan=block(mbon, dan, i, j, c),
                         mbon_to_mbon=block(mbon, mbon, i, j, c))

    # -- MBON -> descending neurons -------------------------------------------------
    sel = mbon[i] & dn[j]
    dn_targets = Counter()
    for post, cnt in zip(j[sel], c[sel]):
        dn_targets[ctype[post] or 'untyped'] += int(cnt)
    per_type = Counter()
    for pre, cnt in zip(i[sel], c[sel]):
        per_type[ctype[pre]] += int(cnt)
    report['mbon_to_dn'] = dict(total=block(mbon, dn, i, j, c),
                                top_dn_types=dict(dn_targets.most_common(10)),
                                by_mbon_type=dict(per_type.most_common(12)))

    # -- MBON -> DNa02, the steering neurons the P3 T-maze decoder reads ------------
    # (validation/paradigms/tmaze_odor.py on the validation-harness branch).
    fast = ~np.isin(nt, MODULATORY)[i]           # v3: aminergic out-edges carry no weight
    c_in = np.bincount(j, weights=c, minlength=n)
    dna02 = ctype == 'DNa02'
    direct = Counter()
    for pre, cnt in zip(i[mbon[i] & dna02[j]], c[mbon[i] & dna02[j]]):
        direct[ctype[pre]] += int(cnt)
    # Two-hop input share: sum over X of [c(M,X)/C_in(X)] * [c(X,D)/C_in(D)], summed
    # over both DNa02.  A ranking of anatomical reach, not a model of dynamics.
    x_to_d = fast & dna02[j]
    share_xd = np.zeros(n)
    np.add.at(share_xd, i[x_to_d], c[x_to_d] / c_in[j[x_to_d]])
    m_to_x = mbon[i] & fast
    two_hop = Counter()
    for pre, x, cnt in zip(i[m_to_x], j[m_to_x], c[m_to_x]):
        if share_xd[x]:
            two_hop[ctype[pre]] += float(cnt / c_in[x] * share_xd[x])
    hops = {}
    for thr in (1, 5, 10):
        keep = fast & (c >= thr)
        order = np.argsort(j[keep], kind='stable')          # reverse CSR: post -> pres
        rev_pre = i[keep][order]
        rev_ptr = np.r_[0, np.cumsum(np.bincount(j[keep], minlength=n))]
        dist = np.full(n, -1, dtype=np.int64)
        frontier = np.flatnonzero(dna02); dist[frontier] = 0; level = 0
        while len(frontier) and level < 6:
            level += 1
            nxt = np.unique(np.concatenate([rev_pre[rev_ptr[v]:rev_ptr[v + 1]] for v in frontier]))
            nxt = nxt[dist[nxt] < 0]
            dist[nxt] = level
            frontier = nxt
        hops[f'min{thr}syn'] = {t: (int(dist[(ctype == t) & mbon & (dist >= 0)].min())
                                    if ((ctype == t) & mbon & (dist >= 0)).any() else None)
                                for t in sorted(set(ctype[mbon]))}
    report['mbon_to_dna02'] = dict(
        dna02_input_contacts={str(nodes.side.iat[k]): int(c_in[k]) for k in np.flatnonzero(dna02)},
        direct_contacts_by_mbon_type=dict(direct.most_common()),
        two_hop_input_share_by_mbon_type={k: round(v, 5) for k, v in two_hop.most_common()},
        min_hops_to_dna02_v3_fast_edges=hops,
        note='two-hop share = sum_X c(M,X)/C_in(X) * c(X,DNa02)/C_in(DNa02), both DNa02 summed; '
             'aminergic presynaptic edges excluded (v3 policy)')

    # -- inputs to PPL1 (punishment ingress candidates) -----------------------------
    into_ppl1 = {}
    for t in sorted(set(ctype[ppl1])):
        post_mask = (ctype == t) & ppl1
        s = post_mask[j]
        cls = Counter(); typ = Counter()
        for pre, cnt in zip(i[s], c[s]):
            cls[str(nodes.superclass.iat[pre])] += int(cnt)
            typ[ctype[pre] or 'untyped'] += int(cnt)
        tot = int(c[s].sum())
        into_ppl1[t] = dict(input_contacts=tot,
                            by_superclass={k: round(v / tot, 3) for k, v in cls.most_common(6)},
                            top_presynaptic_types=dict(typ.most_common(8)))
    report['ppl1_inputs'] = into_ppl1

    # -- v3 policy effect on the circuit --------------------------------------------
    mod = np.isin(nt, MODULATORY)
    report['v3_policy'] = dict(
        modulatory_mb_neurons={t: int(v) for t, v in Counter(ctype[mb_core & mod]).most_common()},
        note='out-edges of these neurons carry zero fast weight under v3-modulatory-only')

    if args.synapses is not None:
        report['synapse_rois'] = synapse_rois(args.synapses, ids, ctype, kc, mbon, pam | ppl1)
        if args.points is not None:
            report['compartments'] = compartment_split(args.synapses, args.points, ids, ctype, kc, mbon,
                                                       pam | ppl1, nodes.instance.to_numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=int) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
