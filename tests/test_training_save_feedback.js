'use strict';
// CARD75: a Save-brain (or other training command) outcome stays visible for the displayed owner
// generation, including on graph runs whose poll refresh used to overwrite it, and never leaks
// across an owner or replay boundary. Success is shown only from a final applied acknowledgement.
const test = require('node:test'), assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/training.js'), 'utf8');
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const cut = (a, b) => app.slice(app.indexOf(a), app.indexOf(b) + b.length);
const identityHelper = cut('function identityRejection(', 'window.neuroflyIdentityRejection = identityRejection;');
const capsHelper = cut('const GRAPH_BACKENDS =', 'window.neuroflyGraphPanelCapabilities = graphPanelCapabilities;');
const CHECKPOINT = '/srv/neurofly/out/checkpoints/checkpoint_open-arena_instrument_1791363559769466031.json';

function harness(backend = 'connectome-fixed') {
    const elements = {}, listeners = {};
    const identity = {daemon_run_id: 'daemon-a', activation: 1, run_id: 'run-a', instance_id: 'instance-a', assay: 'open-arena', backend,
        label: backend === 'modular' ? 'modular' : 'MaleCNS fixed'};
    let reply = async () => ({status: 'ok', checkpoint: CHECKPOINT, ack: {applied: true, identity: {...identity}}});
    const sent = [];
    const bridge = {connected: true, activeUrl: 'http://intended', replayMode: false, readOnly: false,
        lastOrderedPacket: {identity, brain_id: 'brain-a', run_id: 'daemon-a'},
        sendCommand: async (action, params, queued) => { sent.push(action); queued?.(); return reply(action, params); }};
    const hud = {daemonBridge: bridge, arena: {activeParadigmId: 'open-arena'}};
    function el(id) {
        return elements[id] ||= {style: {}, attrs: {}, textContent: '', innerHTML: '', firstChild: {textContent: ''}, classes: new Set(),
            classList: {toggle(c, v) { v ? this.owner.classes.add(c) : this.owner.classes.delete(c); }},
            addEventListener() {}, setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; }};
    }
    const data = () => ({brain: {brain_id: 'brain-a', paradigm: 'open-arena', seed: 1, steps: 4, trials: 0, learning_enabled: false, probe: {discrimination: 0},
        weight_change_l2: 0, history: [], weights: Array.from({length: 120}, () => [1, 1])},
        telemetry: {identity: {...identity}, brain_id: 'brain-a'}, brains: [], status: {stream: {}, sim_speed: 1}});
    const c = {window: {app: {hud}, addEventListener(k, f) { listeners[k] = f; }},
        document: {getElementById: (id) => { const e = el(id); e.classList.owner = e; return e; }, createElement() { return {click() {}}; }},
        AbortSignal: {timeout: () => ({})}, fetch: async () => ({ok: true, json: async () => data()}), Blob, URL: {createObjectURL: () => '', revokeObjectURL() {}},
        setTimeout() {}, setInterval() {}, Date, console};
    vm.createContext(c);
    vm.runInContext(identityHelper, c);
    vm.runInContext(capsHelper, c);
    vm.runInContext(source.replace(/\}\)\(\);\s*$/, 'window.testOwner={refresh,reconcileTrainingOwner};})();'), c);
    const msg = () => el('trainingMessage');
    return {el, msg, identity, bridge, listeners, sent, api: c.window.testOwner, reply(fn) { reply = fn; },
        async polls(n = 3) { for (let i = 0; i < n; i++) await c.window.testOwner.refresh(); }};
}
const GENERIC = /Teach, Reverse, Probe and learning controls are disabled/;

test('graph success shows the checkpoint basename and survives poll refreshes', async () => {
    const h = harness();
    await h.api.refresh();
    assert.match(h.msg().textContent, GENERIC);
    await h.el('trainingSave').onclick();
    assert.deepEqual(h.sent, ['save_checkpoint']);
    for (let i = 0; i < 3; i++) {
        assert.equal(h.msg().textContent, 'Applied: save checkpoint · checkpoint_open-arena_instrument_1791363559769466031.json');
        assert.equal(h.msg().classes.has('error'), false);
        await h.polls(1);
    }
    // Learning controls stay disabled on the fixed graph; only Save/Export remain usable.
    for (const k of ['Teach', 'Reverse', 'Probe', 'Freeze']) assert.equal(h.el('training' + k).disabled, true, k);
    assert.equal(h.el('trainingSave').disabled, false);
});

test('graph error stays visible as an error across poll refreshes', async () => {
    const h = harness();
    await h.api.refresh();
    h.reply(async () => ({status: 'error', message: 'Checkpoint not saved: disk full', ack: {applied: false}}));
    await h.el('trainingSave').onclick();
    await h.polls(3);
    assert.equal(h.msg().textContent, 'Checkpoint not saved: disk full');
    assert.equal(h.msg().classes.has('error'), true);
});

test('no final acknowledgement is reported as unknown, never as success', async () => {
    for (const reply of [async () => null, async () => ({status: 'ok'}), async () => ({status: 'error', timed_out: true, message: 'Command timed out; outcome unknown.'})]) {
        const h = harness();
        await h.api.refresh();
        h.reply(reply);
        await h.el('trainingSave').onclick();
        await h.polls(2);
        assert.doesNotMatch(h.msg().textContent, /Applied/);
        assert.equal(h.msg().classes.has('error'), true);
    }
});

test('queued status shows while the command is pending and buttons stay disabled', async () => {
    const h = harness();
    await h.api.refresh();
    let resolve;
    h.reply(() => new Promise((r) => { resolve = r; }));
    const task = h.el('trainingSave').onclick();
    assert.match(h.msg().textContent, /Queued · awaiting final durable acknowledgement/);
    assert.equal(h.el('trainingSave').disabled, true);
    resolve({status: 'ok', checkpoint: CHECKPOINT, ack: {applied: true, identity: {...h.identity}}});
    await task;
    assert.match(h.msg().textContent, /^Applied: save checkpoint · /);
});

test('an owner change clears the outcome', async () => {
    const h = harness();
    await h.api.refresh();
    await h.el('trainingSave').onclick();
    assert.match(h.msg().textContent, /^Applied/);
    h.identity.activation++;
    h.api.reconcileTrainingOwner();
    assert.doesNotMatch(h.msg().textContent, /Applied/);
    await h.polls(2);
    assert.match(h.msg().textContent, GENERIC);
});

test('entering replay clears the outcome and leaving replay does not restore it', async () => {
    const h = harness();
    await h.api.refresh();
    h.reply(async () => ({status: 'error', message: 'Checkpoint not saved: disk full'}));
    await h.el('trainingSave').onclick();
    assert.equal(h.msg().textContent, 'Checkpoint not saved: disk full');
    h.bridge.replayMode = true;
    h.listeners['neurofly-replay-mode-change']();
    assert.doesNotMatch(h.msg().textContent, /disk full/);
    h.bridge.replayMode = false;
    h.listeners['neurofly-replay-mode-change']();
    await h.polls(2);
    assert.doesNotMatch(h.msg().textContent, /disk full|Applied/);
    assert.match(h.msg().textContent, GENERIC);
});

test('a stale completion from an older owner generation is not shown', async () => {
    for (const outcome of ['success', 'error']) {
        const h = harness();
        await h.api.refresh();
        let resolve;
        h.reply(() => new Promise((r) => { resolve = r; }));
        const task = h.el('trainingSave').onclick();
        h.identity.activation++;
        h.api.reconcileTrainingOwner();
        resolve(outcome === 'success'
            ? {status: 'ok', checkpoint: CHECKPOINT, ack: {applied: true, identity: {...h.identity, activation: 1}}}
            : {status: 'error', message: 'Checkpoint not saved: old owner'});
        await task;
        await h.polls(2);
        assert.doesNotMatch(h.msg().textContent, /Applied|old owner/, outcome);
    }
});

test('modular controller keeps the outcome too', async () => {
    const h = harness('modular');
    await h.api.refresh();
    await h.el('trainingSave').onclick();
    await h.polls(2);
    assert.equal(h.msg().textContent, 'Applied: save checkpoint · checkpoint_open-arena_instrument_1791363559769466031.json');
});
