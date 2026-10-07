'use strict';
// Split mode: the web process stays up while the simulation process is down. A paused,
// stale display must never mask sim_process_down; a restarted (paused) simulation shows
// its own new run identity.
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');

function harness() {
    const elements = {}, listeners = {}, fetches = [];
    for (const id of ['identRun', 'identFault', 'statAchieved', 'arenaRunState', 'btnPauseToggle',
                      'statDataAge', 'statStepAge', 'identBackend', 'identLabel', 'identAssists', 'identMotor',
                      'identAssistance', 'identOptoMap'])
        elements[id] = {textContent: '', title: '', style: {}, disabled: false};
    const c = {window: {NeuroFlyObservationRenderer: require('../web/observation_renderer.js')},
        document: {getElementById: id => elements[id] || null, querySelector: () => null},
        performance: {now: () => 1000}, console, EXPERIMENT_GUIDES: {optomotor: {}},
        fetch: (...args) => { fetches.push(args); return Promise.reject(new Error('no network in tests')); },
        EventSource: class { constructor() {} addEventListener(name, fn) { listeners[name] = fn; } close() {} }};
    vm.createContext(c);
    vm.runInContext(app.slice(app.indexOf('const DAEMON_FRAME_OFFSET'), app.indexOf('const ASSAY_CONFIGS ='))
        + '\nthis.B=DaemonBridgeClient;', c);
    // Panels outside this repair (graph readouts, 3D, buffers) accept anything.
    const loose = () => new Proxy(function () {}, {
        get: (t, k) => (k in t ? t[k] : (t[k] = loose())), apply: () => undefined,
        set: (t, k, v) => { Reflect.set(t, k, v); return true; }});
    const arena = Object.assign(loose(), {activeParadigmId: 'optomotor', remoteDriven: true, awaitingDaemon: false,
        telemetryBuffer: []});
    const bridge = Object.create(c.B.prototype);
    Object.assign(bridge, {arena, hud: loose(), activeUrl: 'http://test-only', staleAfterMs: 5000, lastValidDataTime: 1000,
        statusPill: {textContent: '', title: '', style: {}}, replayMode: false, switchPending: false,
        commandAckCache: new Map(), pendingCommands: new Map(),
        updatePersistenceBanner() {}, renderTiming() {}, renderDeliveryIdentity() {}, showDaemonAddress() {},
        fetchManifest() {}, scheduleReconnect() {}});
    const identity = run => ({run_id: run, daemon_run_id: run, instance_id: `i-${run}`, activation: 0,
        backend: 'connectome-fixed', assay: 'optomotor'});
    const packet = (run, step, paused) => ({type: 'telemetry', run_id: run, paradigm: 'optomotor', step, paused,
        identity: identity(run), fly: {state: 'still', x: 0, y: 0, heading: 0}, trial: 1});
    bridge.onDaemonConnected({status: 'online', total_steps: 70, paused: true, active_paradigm: 'optomotor',
        liveness: {state: 'paused'}});
    bridge.handleDaemonPacket(packet('run-old', 70, true));
    return {bridge, elements, fetches, packet,
        beat(data) { listeners.heartbeat({data: JSON.stringify(data)}); }};
}

// Exactly what SimProxy.health() + the stream heartbeat send while the simulation process is down.
const DOWN = {server_time: 1, seq: 70, identity: null, step: 70, paused: false, status: 'error', halted: true,
    error: 'simulation process is down (connection closed by the simulation process)',
    liveness: {state: 'sim_process_down', sim_process_connected: false, sim_thread_alive: false},
    error_detail: {kind: 'simulation_process', message: 'simulation process is down'}};

test('baseline: the paused run is shown as paused with its run id and Resume', () => {
    const {bridge, elements} = harness();
    assert.match(bridge.statusPill.textContent, /DAEMON CONNECTED · PAUSED/);
    assert.match(elements.identRun.textContent, /run-old/);
    assert.equal(elements.btnPauseToggle.textContent, 'Resume');
});

test('paused -> simulation process down: disconnected, stale run cleared, commands disabled', async () => {
    const {bridge, elements, fetches, beat} = harness();
    beat(DOWN);
    assert.match(bridge.statusPill.textContent, /SIMULATION PROCESS DOWN/);
    assert.doesNotMatch(bridge.statusPill.textContent, /CONNECTED|PAUSED/);
    assert.equal(elements.identRun.textContent, 'run --');
    assert.equal(elements.btnPauseToggle.disabled, true);
    assert.notEqual(elements.btnPauseToggle.textContent, 'Resume');
    assert.match(elements.arenaRunState.textContent, /SIMULATION PROCESS DOWN/);
    assert.equal(await bridge.sendCommand('set_paused', {paused: false}), null);
    assert.equal(fetches.length, 0);
    // Repeated heartbeats (more than 169 s of them) keep saying so.
    for (let i = 0; i < 5; i++) beat(DOWN);
    assert.match(bridge.statusPill.textContent, /SIMULATION PROCESS DOWN/);
    // Unresponsive (connected, no status yet) is not shown as paused either.
    beat({...DOWN, liveness: {state: 'sim_process_unresponsive'}});
    assert.match(bridge.statusPill.textContent, /SIMULATION PROCESS NOT RESPONDING/);
});

test('a page opened while the simulation process is down starts disconnected, not paused', () => {
    const {bridge} = harness();
    bridge.onDaemonConnected({...DOWN, total_steps: 70});
    assert.match(bridge.statusPill.textContent, /SIMULATION PROCESS DOWN/);
});

test('paused restart: the new run identity is shown and commands are enabled again', () => {
    const {bridge, elements, beat, packet} = harness();
    beat(DOWN);
    // The restarted process is connected; its heartbeat carries the new run's identity.
    beat({server_time: 2, seq: null, step: 0, paused: true, status: 'online', halted: false, error: null,
          identity: packet('run-new', 0, true).identity, liveness: {state: 'paused'}});
    assert.equal(bridge.simProcessDown, null);
    // Its first published frame (seq 1 again, step 0, paused) is applied, not dropped as old.
    assert.notEqual(bridge.handleDaemonPacket(packet('run-new', 0, true)), false);
    assert.match(elements.identRun.textContent, /run-new/);
    assert.doesNotMatch(elements.identRun.textContent, /run-old/);
    assert.equal(elements.btnPauseToggle.textContent, 'Resume');
    assert.equal(elements.btnPauseToggle.disabled, false);
    bridge.updateFreshness();
    assert.match(bridge.statusPill.textContent, /DAEMON CONNECTED · PAUSED/);
});
