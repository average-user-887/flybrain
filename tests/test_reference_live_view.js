'use strict';
// Reference fly (not connectome) in the live dashboard: selector, identity bar, readout.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
const LABEL = 'Reference fly (not connectome) — illustrative reference controller';
function slice(start, end) { const a = app.indexOf(start), b = app.indexOf(end, a); assert.ok(a >= 0 && b > a); return app.slice(a, b); }
function ctx(extra = {}) { const c = {window: {}, ...extra}; vm.createContext(c); vm.runInContext(slice('const SELECTABLE_BACKENDS =', 'function identityRejection('), c); return c; }
const refIdentity = {backend: 'reference-flygym', label: LABEL, run_id: 'reference-abc', instance_id: 'reference-flygym:reference-abc',
    daemon_run_id: 'daemon', activation: 2, graph_sha256: null, is_connectome: false};
const refPacket = {type: 'telemetry', run_id: 'daemon', step: 50, paradigm: 'flat-ground-walking', paused: false, identity: refIdentity,
    timing: {achieved_speed: 0.6, requested_speed: 1},
    reference_fly: {display_label: LABEL, assay: 'flat-ground-walking', run_id: 'reference-abc', sim_time_s: 0.1,
        pos_mm: [1.4, 0.1], yaw_rad: 0.05, forward_speed_mm_s: 14, contacts: [1, 0, 1, 0, 1, 0],
        trail_mm: [[0, 0], [0.7, 0.05], [1.4, 0.1]], store: '<repo>/outputs/reference_fly/live-x'}};
const bridge = {connected: true, activeUrl: 'http://127.0.0.1:8969', readOnly: false, switchPending: false};

test('the selector shows and allows the reference fly while its packet is live', () => {
    const state = ctx().window.neuroflyBackendSelectorState;
    // Reference packets leave the assay arena waiting: the selector still has a verified identity.
    const s = state({remoteDriven: false, awaitingDaemon: true, remotePacket: null, referencePacket: refPacket}, bridge);
    assert.equal(s.backend, 'reference-flygym'); assert.equal(s.allowed, true);
    const off = state({remoteDriven: false, awaitingDaemon: true, remotePacket: null, referencePacket: refPacket}, {...bridge, connected: false});
    assert.equal(off.backend, ''); assert.equal(off.allowed, false);
});

test('the reference option is labelled, listed after the connectome and never preselected', () => {
    const sel = html.slice(html.indexOf('id="selectBackend"'), html.indexOf('</select>', html.indexOf('id="selectBackend"')));
    assert.match(sel, /<option value="reference-flygym">Reference fly \(not connectome\) — illustrative reference controller<\/option>/);
    assert.doesNotMatch(sel, /value="reference-flygym"[^>]*selected/);
    assert.ok(sel.indexOf('connectome-fixed') < sel.indexOf('reference-flygym'));
    assert.match(html, /id="referenceFlyView"[^>]*hidden/);
    assert.doesNotMatch(html + app, /what a real fly would do/i);
});

test('packet classification and readout carry the label; connectome packets are not reference', () => {
    const c = ctx();
    assert.equal(c.window.neuroflyIsReferencePacket(refPacket), true);
    assert.equal(c.window.neuroflyIsReferencePacket({identity: {backend: 'connectome-fixed'}, reference_fly: {}}), false);
    assert.equal(c.window.neuroflyIsReferencePacket({identity: {backend: 'reference-flygym'}}), false);
    const text = c.window.neuroflyReferenceReadout(refPacket);
    assert.match(text, /^Reference fly \(not connectome\) — illustrative reference controller · walking/);
    assert.match(text, /outputs\/reference_fly\/live-x/);
});

test('the identity bar and banner name the reference fly and never claim Assistance OFF', () => {
    const els = {};
    const el = id => (els[id] ||= {textContent: '', title: '', style: {}});
    const c = {window: {}, document: {getElementById: el}};
    vm.createContext(c);
    vm.runInContext(slice('const MOTOR_SOURCE_NOTES =', 'function graphPanelCapabilities(')
        + slice('function renderIdentity(pkt) {', '// -----------------------------------------------------------------------------\n// Structured error')
        + '\nwindow.renderIdentity=renderIdentity;', c);
    c.window.renderIdentity(refPacket);
    assert.equal(els.identBackend.textContent, 'reference-flygym');
    assert.equal(els.identLabel.textContent, LABEL);
    assert.equal(els.identAssistance.textContent, 'n/a (not connectome)');
    assert.match(els.identityBanner.textContent, /REFERENCE FLY \(NOT CONNECTOME\).*not connectome evidence.*14 assays are unavailable/);
    assert.equal(els.identityBanner.style.display, 'block');
    c.window.renderIdentity({identity: {backend: 'connectome-fixed', label: 'MaleCNS'}, motor: {}});
    assert.equal(els.identBackend.textContent, 'connectome-fixed');
    assert.doesNotMatch(els.identityBanner.textContent, /REFERENCE/);
});

test('the reference canvas draws the walked trail on its own canvas', () => {
    const calls = [];
    const ctx2d = new Proxy({}, {get: (t, k) => k in t ? t[k] : (...a) => calls.push(k), set: (t, k, v) => (t[k] = v, true)});
    const canvas = {clientWidth: 300, clientHeight: 200, width: 0, height: 0, getContext: () => ctx2d};
    const out = ctx().window.neuroflyRenderReferenceFly(canvas, refPacket.reference_fly);
    assert.equal(out.points, 4);
    assert.ok(calls.includes('lineTo') && calls.includes('fillText'));
});

test('packet validation accepts reference packets without an assay pose and rejects broken ones', () => {
    const c = {window: {}}; vm.createContext(c);
    vm.runInContext(slice('function validateDaemonPacket(pkt) {', 'class DaemonBridgeClient {') + '\nwindow.v=validateDaemonPacket;', c);
    assert.equal(c.window.v(refPacket), null);
    assert.match(c.window.v({...refPacket, reference_fly: {...refPacket.reference_fly, pos_mm: [NaN, 0]}}), /reference fly position/);
    assert.match(c.window.v({type: 'telemetry', step: 1, paradigm: 'open-arena', identity: {backend: 'connectome-fixed'}}), /fly position/);
});

test('pause the connectome, select the reference, then Resume sends paused:false (reference packet state)', () => {
    const start = app.indexOf("    const btnPause = document.getElementById('btnPauseToggle');");
    const end = app.indexOf('    // Expose convenient top-level app handle', start);
    assert.ok(start >= 0 && end > start);
    const sent = [];
    let click = null;
    const button = {addEventListener: (name, fn) => { click = fn; }, classList: {toggle() {}}, textContent: ''};
    const arena = {remotePacket: {paused: true, identity: {backend: 'connectome-fixed'}}};
    const hud = {daemonBridge: {connected: true, replayMode: false, sendCommand: (a, p) => sent.push([a, p])}};
    const c = {arena, hud, window: {}, document: {getElementById: id => (id === 'btnPauseToggle' ? button : null)}};
    vm.createContext(c);
    vm.runInContext('let isPaused = false;\n' + app.slice(start, end), c);
    // Reference view: the assay packet is cleared, its own (paused) packet is live.
    arena.remotePacket = null;
    arena.referencePacket = {...refPacket, paused: true};
    click();
    assert.equal(JSON.stringify(sent.at(-1)), JSON.stringify(['set_paused', {paused: false}]));
    arena.referencePacket = {...refPacket, paused: false};
    click();
    assert.equal(JSON.stringify(sent.at(-1)), JSON.stringify(['set_paused', {paused: true}]));
});

test('header clock and assay fields show the reference clock, never the stale graph clock or trial', () => {
    const c = {window: {}}; vm.createContext(c);
    vm.runInContext('const TRIAL_CLOCK_REASONS = {};\n' + slice('function clockReadouts(view, packet, remote, replay) {', 'class ScientificBioArena {') + '\nwindow.clockReadouts=clockReadouts;', c);
    const stale = {simTime: 99.5, stepCount: 4975, paradigmElapsedSec: 12, currentTrial: 3};
    const graphPacket = {clocks: {graph: {elapsed_s: 400, step: 20000}, trial: {}}};
    const ref = c.window.clockReadouts({...stale, referencePacket: {...refPacket, step: 50}}, null, false, false);
    assert.equal(ref.timeLabel, 'Reference time'); assert.equal(ref.simTime, '0.10s'); assert.equal(ref.step, '50');
    assert.equal(ref.graphVisible, false); assert.equal(ref.trial, 'n/a (reference)'); assert.equal(ref.elapsed, 'n/a (reference)');
    assert.equal(ref.assay, 'n/a (reference)');
    const live = c.window.clockReadouts(stale, graphPacket, true, false);
    assert.equal(live.timeLabel, 'Session time'); assert.equal(live.graphVisible, true); assert.equal(live.assay, undefined);
});
