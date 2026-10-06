'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');

function harness() {
    const elements = {};
    for (const id of ['daemonAddress', 'identDevice', 'deliveryBadge', 'statDataAge', 'statStepAge']) {
        elements[id] = {textContent: '', title: '', style: {}};
    }
    const context = {window: {}, performance: {now: () => 1000},
        document: {getElementById: id => elements[id] || null, querySelector: () => null},
        EXPERIMENT_GUIDES: {'t-maze': {}, 'y-maze': {}, 'open-arena': {}}};
    vm.createContext(context);
    const start = app.indexOf('function deliveryBuildState(');
    const end = app.indexOf('window.neuroflyValidateRequestedSpeed', start);
    vm.runInContext(app.slice(start, end)
        + app.slice(app.indexOf('class DaemonBridgeClient'), app.indexOf('const ASSAY_CONFIGS ='))
        + '\nthis.Bridge=DaemonBridgeClient;', context);
    const bridge = Object.create(context.Bridge.prototype);
    Object.assign(bridge, {activeUrl: 'http://daemon.example.invalid:8769', staleAfterMs: 5000,
        lastValidDataTime: 1000, arena: {activeParadigmId: 't-maze'},
        statusPill: {textContent: '', title: '', style: {}}, addressBadge: elements.daemonAddress,
        updatePersistenceBanner() {}, renderTiming() {}, startStreaming() {}});
    bridge.onDaemonConnected({active_paradigm: 't-maze', total_steps: 101, uptime_sec: 2.5,
        version: 'test', backend: 'modular', compute: {device: 'cpu', detail: 'Initial test CPU'}});
    return {bridge, elements};
}

test('live connection tooltip has no assay, step or uptime snapshot through assay changes', () => {
    const {bridge} = harness();
    const title = `Connected to the learning daemon at ${bridge.activeUrl}`;
    assert.equal(bridge.statusPill.title, title);
    for (const [assay, step] of [['open-arena', 500], ['y-maze', 600]]) {
        bridge.arena.activeParadigmId = assay;
        bridge.noteStep(step, `new-run|${assay}|2`);
        bridge.updateFreshness();
        assert.equal(bridge.lastStepSeen, step);
        assert.equal(bridge.statusPill.title, title);
        assert.doesNotMatch(bridge.statusPill.title, /assay:|\bsteps\b|uptime|t-maze|101|2\.5/);
    }
});

test('device and address snapshot labels survive pause and live recovery', () => {
    const {bridge, elements} = harness();
    const address = elements.daemonAddress.title;
    const device = elements.identDevice.textContent;
    assert.match(address, /Initial connection snapshot: backend modular/);
    assert.match(device, /^Initial connection snapshot: cpu$/);
    assert.match(elements.identDevice.title, /^Initial connection snapshot: Initial test CPU$/);
    bridge.daemonPaused = true;
    bridge.updateFreshness();
    assert.match(bridge.statusPill.textContent, /PAUSED/);
    bridge.daemonPaused = false;
    bridge.updateFreshness();
    assert.equal(bridge.statusPill.textContent, '● LIVE DAEMON');
    assert.equal(bridge.statusPill.title, `Connected to the learning daemon at ${bridge.activeUrl}`);
    assert.equal(elements.daemonAddress.title, address);
    assert.equal(elements.identDevice.textContent, device);
    assert.match(elements.identDevice.title, /^Initial connection snapshot:/);
});

test('fault and watchdog tooltips retain current diagnostics before endpoint-only recovery', () => {
    const {bridge} = harness();
    bridge.daemonHalt = {error: 'test fault', detail: {paradigm: 'y-maze', step: 700}};
    bridge.updateFreshness();
    assert.match(bridge.statusPill.title, /test fault.*assay y-maze, step 700/);
    bridge.daemonHalt = null;
    bridge.daemonLiveness = {state: 'dead'};
    bridge.updateFreshness();
    assert.match(bridge.statusPill.textContent, /NOT ADVANCING/);
    assert.match(bridge.statusPill.title, /simulation thread in the daemon has stopped/);
    bridge.daemonLiveness = {state: 'advancing'};
    bridge.updateFreshness();
    assert.equal(bridge.statusPill.title, `Connected to the learning daemon at ${bridge.activeUrl}`);
});
