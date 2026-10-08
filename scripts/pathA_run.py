#!/usr/bin/env python3
"""Path A, phase 3: run the FROZEN known-circuit battery on the existing v3 engine.

Reads ``qualification/pathA/contract.json`` and refuses to run if its sha256
differs from the value given with ``--contract-sha256`` (the value recorded in
the commit that froze it).  Nothing in the engine, its equations or the graph
weights is changed:

* Activation: each activated cell receives independent Poisson events at the
  condition rate (Shiu et al. 2024 method).  An event is delivered through the
  existing per-neuron current channel as ONE 0.1 ms tick of drive
  ``pulse_drive`` (mV-equivalent), which drives the membrane to the declared
  ceiling and so forces a spike unless the cell is refractory -- the same
  effect as Shiu's Poisson synapse of 250 x W_syn.
* Silencing: a constant drive ``silence_drive`` holds the cell at the declared
  floor E_inh for the whole run (the engine's existing clamp), so it never
  spikes.  No weight is changed.
* Shuffled-input counterfactual (LABELLED OFFLINE COUNTERFACTUAL, NOT NeuroFly
  wiring): the targets of every out-edge of the activated cells are permuted
  uniformly over all neurons, in memory, with the condition's seed; weights
  and every other edge are untouched.  The pinned graph file is never written.

* Silencing counterfactual (``--cf-silence-prereg``; LABELLED OFFLINE
  COUNTERFACTUAL, NOT a fix): a frozen preregistration names frozen contract
  conditions and extra cells to clamp with the SAME silencing method above.
  The graph, weights and edges are untouched; rows get the prereg's ids,
  readouts and seeds, and its sha256 in their provenance.  Each such row also
  records an in-run audit: the post-reset state is at rest with an empty delay
  queue, and the sha256 of the exact input train.  A prereg may also name an
  edge-zeroing reference (outgoing weights of named cells set to 0 in an
  in-memory copy, like the shuffled counterfactual) to check the clamp against
  zero-outgoing-edge semantics in the real network.  The graph file is
  re-hashed at the end and must be unchanged.
* Direct-edge deletion (prereg ``direct_edges`` + per-condition ``zero_edges``):
  the listed existing edges get weight 0 in an in-memory copy of the v3 weight
  array.  The runner first checks that the listed edges are EXACTLY the
  edges from ``direct_edges_from`` to ``direct_edges_to`` in the pinned graph,
  with the pinned body ids, and refuses otherwise; after copying it checks that
  only those weights changed.  ``weight_copy`` with no edges exercises the same
  path as a sham.  Each row records the sha256 of the weight array used, that the
  parent weight array is unchanged, a canonical digest of the full post-reset
  state and of the Poisson RNG state.  ``--save-reset-fixture`` writes the
  post-reset state of a freshly built brain (no step) as an immutable fixture.

Output: one JSON line per (condition, seed) with per-readout-set mean rates
and per-cell counts, plus a sparse full-network spike-count file per run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import pathA_provenance as prov  # noqa: E402


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_v3_arrays(gdir: Path, cdir: Path, neurons_sha256: str):
    """Graph arrays with the v3 transmitter policy applied.  The transmitter table must be
    the contract's neurons.feather (sha256 checked BEFORE any label is applied)."""
    from brainlab.transmitter_policy import apply_policy, load_transmitters
    if sha256_file(Path(cdir) / 'normalized/neurons.feather') != neurons_sha256:
        raise SystemExit('neurons.feather sha256 differs from the contract: refusing to apply its transmitter labels')
    with np.load(gdir / 'graph.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in ('ptr', 'post', 'weight', 'ids')}
    labels = load_transmitters(cdir)
    w3, _ = apply_policy(arrays['ptr'], arrays['post'], arrays['weight'], labels)
    arrays['weight'] = np.ascontiguousarray(w3, dtype=np.float32)
    return arrays


def shuffled_post(arrays, sources, seed):
    """Counterfactual: permute targets of the sources' out-edges (labelled offline only)."""
    ptr, post = arrays['ptr'], arrays['post']
    new = post.copy()
    rng = np.random.default_rng(seed)
    n = len(ptr) - 1
    for s in sources:
        a, b = int(ptr[s]), int(ptr[s + 1])
        new[a:b] = rng.integers(0, n, size=b - a, dtype=np.int64).astype(np.int32)
    return new


def cf_plan(contract, prereg):
    """Conditions of a silencing-counterfactual prereg: frozen contract conditions
    plus a clamp of the prereg's cells.  Returns (extra sets, condition list)."""
    by_id = {c['id']: c for c in contract['conditions']}
    clash = set(prereg['sets']) & set(contract['sets'])
    if clash:
        raise SystemExit(f'prereg sets shadow contract sets: {sorted(clash)}')
    conds = []
    for pc in prereg['conditions']:
        base = by_id[pc['base_condition']]
        if base.get('counterfactual'):
            raise SystemExit(f"{pc['id']}: base condition is itself a counterfactual")
        conds.append(dict(base, id=pc['id'], silence=list(base.get('silence', [])) + list(pc['add_silence']),
                          counterfactual=pc['counterfactual'], readouts=list(prereg['readouts']),
                          seeds=list(pc['seeds']), zero_outgoing=list(pc.get('zero_outgoing', [])),
                          zero_edges=[int(e) for e in pc.get('zero_edges', [])],
                          weight_copy=bool(pc.get('weight_copy') or pc.get('zero_edges'))))
    return prereg['sets'], conds


def zeroed_weight(arrays, sources):
    """Edge-zeroing reference: outgoing weights of ``sources`` set to 0 in a copy (graph file untouched)."""
    ptr, w = arrays['ptr'], arrays['weight'].copy()
    for s in sources:
        w[int(ptr[s]):int(ptr[s + 1])] = 0.0
    return w


def check_direct_edges(arrays, sets, prereg):
    """The prereg's ``direct_edges`` must be EXACTLY the from->to edges of the pinned graph,
    with matching body ids and nonzero weights; every condition's ``zero_edges`` must be a
    subset.  Returns the sorted listed indices.  Refuses on any mismatch."""
    ptr, post, ids, w = arrays['ptr'], arrays['post'], arrays['ids'], arrays['weight']
    src, dst = sets[prereg['direct_edges_from']], sets[prereg['direct_edges_to']]
    actual = np.sort(np.concatenate([int(ptr[s]) + np.flatnonzero(np.isin(post[int(ptr[s]):int(ptr[s + 1])], dst))
                                     for s in src]))
    rows = prereg['direct_edges']
    listed = np.sort(np.array([int(r['edge_index']) for r in rows], dtype=np.int64))
    if len(listed) != int(prereg['direct_edge_count']) or not np.array_equal(listed, actual):
        raise SystemExit(f'direct_edges {listed.tolist()} are not exactly the {prereg["direct_edges_from"]}->'
                         f'{prereg["direct_edges_to"]} edges {actual.tolist()} of this graph')
    owner = np.searchsorted(ptr, listed, side='right') - 1
    for r, i, o in zip(sorted(rows, key=lambda r: r['edge_index']), listed, owner):
        if (int(ids[o]), int(ids[post[i]])) != (int(r['pre_body']), int(r['post_body'])) or w[i] == 0:
            raise SystemExit(f'direct edge {int(i)}: bodies {(int(ids[o]), int(ids[post[i]]))} or weight do not match the prereg')
    return listed


def deleted_weight(arrays, edges):
    """Copy of the weight array with exactly ``edges`` set to 0; refuses if anything else changed."""
    w = arrays['weight'].copy()
    idx = np.sort(np.asarray(edges, dtype=np.int64))
    w[idx] = 0.0
    changed = np.flatnonzero(w != arrays['weight'])
    if not np.array_equal(changed, idx):
        raise SystemExit(f'weight copy changed {changed.tolist()}, expected exactly {idx.tolist()}')
    return w


STATE_SCALARS = ('cursor', 'total_spikes', 'sim_ms')


def state_digest(brain) -> str:
    """Canonical sha256 of the full mutable state: name, dtype, shape and bytes of every state
    array, then the clock scalars.  A different dtype, shape, byte or scalar gives a new digest."""
    h = hashlib.sha256(b'neurofly.state/1\0' + str(brain.dynamics).encode() + b'\0')
    for name in brain._state_arrays():
        a = np.ascontiguousarray(getattr(brain, name))
        h.update(f'{name}|{a.dtype.str}|{a.shape}|'.encode() + a.tobytes() + b'\0')
    for name in STATE_SCALARS:
        h.update(f'{name}={getattr(brain, name)!r}\0'.encode())
    return h.hexdigest()


def rng_digest(seed_seq) -> str:
    """sha256 of the initial bit-generator state of the run's Poisson RNG."""
    st = np.random.default_rng(seed_seq).bit_generator.state
    return hashlib.sha256(json.dumps(st, sort_keys=True, default=int).encode()).hexdigest()


def save_reset_fixture(brain, path):
    """Immutable post-reset snapshot (no step taken): every state array plus the clock scalars."""
    brain.reset_state()
    arrays = {f'state__{k}': np.ascontiguousarray(getattr(brain, k)) for k in brain._state_arrays()}
    arrays.update({f'scalar__{k}': np.array(getattr(brain, k)) for k in STATE_SCALARS})
    np.savez(path, dynamics=np.array(str(brain.dynamics)), **arrays)
    return state_digest(brain)


def state_at_rest(brain, v_rest=-52.0):
    """True iff every transient (membrane, conductances, refractory, delay queue, active set, clocks) is at rest."""
    ok = bool(np.all(brain.v == v_rest)) and brain.cursor == 0 and brain.sim_ms == 0.0
    for name in brain._state_arrays():
        if name != 'v':
            ok &= not np.any(getattr(brain, name))
    return ok


def run_one(brain, stim, rate_hz, silence, readouts, contract, seed, audit=None):
    proto = contract['protocol']
    dt = 0.1
    ticks = int(round(proto['duration_ms'] / dt))
    pulse = float(proto['pulse_drive'])
    base = np.zeros(brain.n, dtype=np.float32)
    if len(silence):
        base[silence] = float(proto['silence_drive'])
    seed_seq = [seed, int(round(rate_hz * 10)), len(stim)]
    rng = np.random.default_rng(seed_seq)
    p = rate_hz * dt * 1e-3
    if p > 0 and len(stim):
        events = rng.random((ticks, len(stim))) < p
        ev_ticks = np.flatnonzero(events.any(axis=1))
    else:
        events = None
        ev_ticks = np.zeros(0, dtype=np.int64)
    brain.reset_state()
    if audit is not None:
        h = hashlib.sha256(np.asarray(stim, np.int64).tobytes())
        h.update(np.packbits(events).tobytes() if events is not None else b'none')
        audit.update(initial_state_clean=state_at_rest(brain), input_sha256=h.hexdigest(),
                     state_sha256=state_digest(brain), rng_seed_seq=seed_seq, rng_state_sha256=rng_digest(seed_seq))
    total = np.zeros(brain.n, dtype=np.int64)
    t = 0
    clock = time.perf_counter()
    for et in list(ev_ticks) + [ticks]:
        if et > t:
            c, _ = brain.step(base, (et - t) * dt)
            total += c
            t = et
        if et >= ticks:
            break
        d = base.copy()
        d[stim[events[et]]] = pulse
        c, _ = brain.step(d, dt)
        total += c
        t = et + 1
    wall = time.perf_counter() - clock
    dur_s = proto['duration_ms'] / 1000.0
    out = {}
    for name, idx in readouts.items():
        cnt = total[idx]
        out[name] = dict(mean_rate_hz=float(cnt.mean() / dur_s) if len(idx) else None,
                         counts=[int(x) for x in cnt])
    stim_rate = float(total[stim].mean() / dur_s) if len(stim) else 0.0
    return out, total, wall, stim_rate, int(len(ev_ticks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', default=str(ROOT / 'qualification/pathA/contract.json'))
    ap.add_argument('--contract-sha256', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default=None, help='comma list of condition ids (default all)')
    ap.add_argument('--timing-probe', action='store_true',
                    help='harness timing only: stimulate the contract timing set, print wall time, record nothing')
    ap.add_argument('--cf-silence-prereg', default=None,
                    help='LABELLED OFFLINE COUNTERFACTUAL: frozen prereg adding a clamp to frozen conditions')
    ap.add_argument('--cf-silence-prereg-sha256', default=None)
    ap.add_argument('--save-reset-fixture', default=None,
                    help='write the post-reset state of a freshly built brain (no step) to this .npz and exit')
    args = ap.parse_args()
    cpath = Path(args.contract)
    if sha256_file(cpath) != args.contract_sha256:
        raise SystemExit('Contract sha256 mismatch: refusing to run an unfrozen protocol')
    contract = json.loads(cpath.read_text())
    conditions, extra_sets, cf_sha = contract['conditions'], {}, None
    if args.cf_silence_prereg:
        cf_sha = sha256_file(args.cf_silence_prereg)
        if cf_sha != args.cf_silence_prereg_sha256:
            raise SystemExit('Counterfactual prereg sha256 mismatch: refusing to run an unfrozen prereg')
        prereg = json.loads(Path(args.cf_silence_prereg).read_text())
        if prereg['frozen_contract_sha256'] != args.contract_sha256:
            raise SystemExit('Counterfactual prereg was written for a different contract')
        extra_sets, conditions = cf_plan(contract, prereg)
    gdir = Path(os.environ['NEUROFLY_GRAPH_DIR'])
    cdir = Path(os.environ['NEUROFLY_CONNECTOME_DIR'])
    graph_sha = sha256_file(gdir / 'graph.npz')
    if graph_sha != contract['data']['graph_npz_sha256']:
        raise SystemExit('graph.npz sha256 differs from the contract')
    # Immutable run provenance (scripts/pathA_provenance.py): bound into every row.
    base = prov.expected(args.contract_sha256, graph_sha, contract['protocol']['dynamics'],
                         contract['protocol']['backend'], prov.engine_sha256(), prov.code_sha())
    if cf_sha:
        base['cf_prereg_sha256'] = cf_sha

    from brainlab.brain import Brain
    arrays = load_v3_arrays(gdir, cdir, contract['data']['neurons_feather_sha256'])
    sets = {k: np.array(v['node_index'], dtype=np.int64) for k, v in dict(contract['sets'], **extra_sets).items()}
    brain = Brain(arrays=arrays, validate=True, dynamics=contract['protocol']['dynamics'],
                  backend=contract['protocol']['backend'])
    if args.save_reset_fixture:
        print(json.dumps(dict(fixture=args.save_reset_fixture, state_sha256=save_reset_fixture(brain, args.save_reset_fixture))))
        return
    listed_edges = check_direct_edges(arrays, sets, prereg) if cf_sha and prereg.get('direct_edges') else None
    for cond in conditions:
        if cond.get('zero_edges') and (listed_edges is None or not set(cond['zero_edges']) <= set(listed_edges.tolist())):
            raise SystemExit(f"{cond['id']}: zero_edges not among the verified direct_edges")
    base_w_sha = hashlib.sha256(arrays['weight'].tobytes()).hexdigest()
    if args.timing_probe:
        stim = sets[contract['protocol']['timing_probe_set']]
        _, _, wall, _, nev = run_one(brain, stim, 200.0, np.zeros(0, np.int64), {}, contract, 0)
        print(json.dumps(dict(timing_probe_wall_s=wall, event_ticks=nev)))
        return
    out = Path(args.out)
    (out / 'counts').mkdir(parents=True, exist_ok=True)
    log = out / 'runs.jsonl'
    manifest = out / 'provenance.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != base:
            raise SystemExit(f'{manifest} records a different contract/engine/graph/source: refusing to resume')
    else:
        manifest.write_text(json.dumps(base, indent=1) + '\n')
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            try:
                prov.check_row(r, base)
            except prov.ProvenanceError as exc:
                raise SystemExit(f'refusing to resume: {exc}')
            done.add((r['condition'], r['seed']))
    only = set(args.only.split(',')) if args.only else None
    cf_brain = {}
    for cond in conditions:
        if only and cond['id'] not in only:
            continue
        stim = np.concatenate([sets[s] for s in cond['activate']]) if cond['activate'] else np.zeros(0, np.int64)
        silence = np.concatenate([sets[s] for s in cond.get('silence', [])]) if cond.get('silence') else np.zeros(0, np.int64)
        readouts = {r: sets[r] for r in cond.get('readouts') or contract['tests'][cond['test']]['readouts']}
        for seed in cond.get('seeds') or contract['protocol']['seeds']:
            if (cond['id'], seed) in done:
                continue
            b = brain
            if cond.get('weight_copy'):
                b = Brain(arrays=dict(arrays, weight=deleted_weight(arrays, cond['zero_edges'])), validate=True,
                          dynamics=contract['protocol']['dynamics'], backend=contract['protocol']['backend'])
            if cond.get('zero_outgoing'):
                zo = np.concatenate([sets[z] for z in cond['zero_outgoing']])
                b = Brain(arrays=dict(arrays, weight=zeroed_weight(arrays, zo)), validate=True,
                          dynamics=contract['protocol']['dynamics'], backend=contract['protocol']['backend'])
            if cond.get('counterfactual') == 'shuffled_input':
                b = Brain(arrays=dict(arrays, post=shuffled_post(arrays, stim, 10_000 + seed)),
                          validate=True, dynamics=contract['protocol']['dynamics'],
                          backend=contract['protocol']['backend'])
            audit = {} if cf_sha else None
            res, total, wall, stim_rate, nev = run_one(b, stim, cond['rate_hz'], silence, readouts, contract, seed, audit)
            nz = np.flatnonzero(total)
            np.savez_compressed(out / 'counts' / f"{cond['id']}_s{seed}.npz", node_index=nz.astype(np.int32),
                                counts=total[nz].astype(np.int32))
            rec = dict(provenance=prov.stamp(base, seed), condition=cond['id'], test=cond['test'], seed=seed, rate_hz=cond['rate_hz'],
                       activate=cond['activate'], silence=cond.get('silence', []),
                       counterfactual=cond.get('counterfactual'), wall_s=wall,
                       activated_cells=int(len(stim)), activated_mean_rate_hz=stim_rate,
                       total_spikes=int(total.sum()), firing_neurons=int(len(nz)), readouts=res)
            if audit is not None:
                audit.update(weight_sha256=hashlib.sha256(b.weight.tobytes()).hexdigest(),
                             parent_weight_unchanged=hashlib.sha256(arrays['weight'].tobytes()).hexdigest() == base_w_sha)
                rec.update(audit=audit, zero_outgoing=cond.get('zero_outgoing', []), zero_edges=cond.get('zero_edges', []))
            with open(log, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
            print(cond['id'], seed, f'{wall:.1f}s', {k: round(v['mean_rate_hz'], 2) for k, v in res.items()}, flush=True)
            del b
    if sha256_file(gdir / 'graph.npz') != graph_sha:
        raise SystemExit('graph.npz changed during the run')


if __name__ == '__main__':
    main()
