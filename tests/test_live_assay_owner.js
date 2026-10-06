'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const app = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const live = fs.readFileSync(path.join(__dirname, '../web/live_assays.js'), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));

function packet(backend = 'modular', activation = 1) {
    return {paradigm: 'open-arena', brain_id: 'experiment-record-' + backend, step: 1,
        identity: {daemon_run_id: 'daemon', run_id: 'run-' + backend, instance_id: 'controller-' + backend,
            backend, activation, label: backend === 'modular' ? 'Modular controller' : 'Fixed graph controller'},
        segment_id: 'segment-' + backend, fly: {speed: 1}, descending: {dna02_yaw: 0},
        live_assay: {model_version: 'test-assay', limitation: 'Test modular assay note',
            parameters: [{name: 'airflow', label: 'Airflow', value: 1, min: 0, max: 40, step: 1, unit: 'mm/s'}],
            actions: [{name: 'food', label: 'Place food'}]}};
}
function harness() {
    const panel = {style: {display: 'none'}, mounts: 0, elements: {},
        set innerHTML(html) {
            this.html = html; this.mounts++; this.elements = {};
            for (const match of html.matchAll(/<(\w+)[^>]*id="([^"]+)"/g)) {
                this.elements[match[2]] = {tagName: match[1], style: {}, listeners: {},
                    addEventListener(name, fn) {this.listeners[name] = fn;}};
            }
        },
        querySelector(selector) {return this.elements[selector.slice(1)] || null;},
        querySelectorAll() {return Object.values(this.elements).filter(e => ['button', 'input'].includes(e.tagName));}};
    const document = {activeElement: null, getElementById(id) {return id === 'assayToolsPanel' ? panel : panel.elements[id] || null;}};
    const context = {window: {}, document, GRAPH_BACKENDS: ['connectome-fixed', 'connectome-plastic', 'connectome-with-trained-readout']};
    vm.createContext(context);
    vm.runInContext(app.slice(app.indexOf('function liveAssayMountKey('), app.indexOf('const SELECTABLE_BACKENDS ='))
        + app.slice(app.indexOf('function graphPanelCapabilities('), app.indexOf('/** Select DN display evidence'))
        + app.slice(app.indexOf('class ScientificHUD'), app.indexOf('// 9. APPLICATION INITIALIZATION'))
        + '\nthis.HUD=ScientificHUD;', context);
    vm.runInContext(live, context);
    const hud = Object.create(context.HUD.prototype), commands = [];
    Object.assign(hud, {arena: {remoteDriven: true, activeParadigmId: 'open-arena', remotePacket: packet(),
        getObservationDisplay() {return this.remotePacket;}}, daemonBridge: {connected: true,
        sendCommand: async (...args) => {commands.push(args); return {status: 'ok'};}}, renderObservationPanels() {}});
    hud.renderAssayTools('open-arena');
    return {hud, panel, document, commands};
}

test('same-assay backend switches remount actual header, limitation and action callbacks in both directions', async () => {
    const {hud, panel, commands} = harness();
    assert.match(panel.html, /CONNECTED ASSAY · Modular controller/);
    const graph = packet('connectome-fixed', 2);
    graph.live_assay.actions = [{name: 'graph-action', label: 'Graph action'}];
    hud.arena.remotePacket = graph; hud.updateAssayTools();
    assert.equal(panel.mounts, 2);
    assert.match(panel.html, /CONNECTED ASSAY · Fixed graph controller/);
    assert.match(panel.html, /Legacy modular assay note \(not a graph measurement or capability\)/);
    assert.equal(panel.querySelector('#live_action_food'), null);
    await panel.querySelector('#live_action_graph-action').listeners.click();
    assert.equal(commands[0][0], 'assay_action'); assert.equal(commands[0][1].name, 'graph-action');
    hud.arena.remotePacket = packet('modular', 3); hud.updateAssayTools();
    assert.equal(panel.mounts, 3); assert.match(panel.html, /CONNECTED ASSAY · Modular controller/);
    assert.doesNotMatch(panel.html, /Legacy modular assay note/);
    assert.equal(panel.querySelector('#live_action_graph-action'), null);
    await panel.querySelector('#live_action_food').listeners.click();
    assert.equal(commands[1][1].name, 'food');
});

test('new accepted owner and capability declarations remount without waiting for assay changes', () => {
    for (const field of ['daemon_run_id', 'run_id', 'instance_id', 'activation', 'backend', 'label']) {
        const {hud, panel} = harness(); const next = clone(hud.arena.remotePacket);
        next.identity[field] = field === 'activation' ? 2 : 'next-' + field;
        hud.arena.remotePacket = next; hud.updateAssayTools(); assert.equal(panel.mounts, 2, field);
    }
    const {hud, panel} = harness(); const next = clone(hud.arena.remotePacket);
    next.live_assay.parameters[0].max = 20;
    next.live_assay.parameters[0].label = 'Revised airflow';
    hud.arena.remotePacket = next; hud.updateAssayTools();
    assert.equal(panel.mounts, 2); assert.match(panel.html, /Revised airflow/); assert.match(panel.html, /max="20"/);
});

test('ordinary parameter samples preserve mounted inputs, focus and pending/read-only fencing', () => {
    const {hud, panel, document} = harness();
    const input = panel.querySelector('#live_param_airflow'); input.value = 'editing'; document.activeElement = input;
    const next = clone(hud.arena.remotePacket); next.step = 2; next.live_assay.parameters[0].value = 5;
    hud.arena.remotePacket = next; hud.updateAssayTools();
    assert.equal(panel.mounts, 1); assert.equal(panel.querySelector('#live_param_airflow'), input);
    assert.equal(input.value, 'editing'); assert.equal(panel.querySelector('#live_value_airflow').textContent, 5);
    hud.daemonBridge.switchPending = true; hud.updateAssayTools(); assert.equal(input.disabled, true);
    hud.daemonBridge.switchPending = false; hud.daemonBridge.readOnly = true;
    hud.updateAssayTools(); assert.equal(input.disabled, true);
    hud.daemonBridge.readOnly = false; document.activeElement = null;
    hud.updateAssayTools(); assert.equal(input.disabled, false); assert.equal(input.value, 5);
});
